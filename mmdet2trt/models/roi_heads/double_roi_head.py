from mmdet2trt.models.builder import register_wraper

from .standard_roi_head import StandardRoIHeadWraper


@register_wraper('mmdet.models.roi_heads.double_roi_head.DoubleHeadRoIHead')
class DoubleHeadRoIHeadWraper(StandardRoIHeadWraper):

    def __init__(self, module, wrap_config):
        super(DoubleHeadRoIHeadWraper, self).__init__(module, wrap_config)

        self.reg_roi_scale_factor = self.module.reg_roi_scale_factor

    def _bbox_forward(self, x, rois):

        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.shared_head is not None:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)

        # rcnn
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results
