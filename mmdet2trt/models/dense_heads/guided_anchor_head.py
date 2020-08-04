import torch
from torch import nn
from mmdet2trt.models.builder import register_warper, build_warper
import mmdet2trt.ops.util_ops as mm2trt_util

from mmdet2trt.core.post_processing.batched_nms import BatchedNMS


class GuidedAnchorHeadWarper(nn.Module):

    def __init__(self, module):
        super(GuidedAnchorHeadWarper, self).__init__()
        self.module = module
        self.loc_filter_thr = module.loc_filter_thr
        self.num_anchors = module.num_anchors
        self.square_anchor_generator = build_warper(self.module.square_anchor_generator)
        self.anchor_coder = build_warper(self.module.anchor_coder)
        self.bbox_coder = build_warper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.use_sigmoid_cls = self.module.use_sigmoid_cls
        # self.rcnn_nms = BatchedNMS(module.test_cfg.score_thr, module.test_cfg.nms.iou_threshold, backgroundLabelId = self.num_classes)

    def get_anchors(self,
                    cls_scores,
                    shape_preds,
                    loc_preds,
                    use_loc_filter=False):

        multi_level_squares = self.square_anchor_generator(cls_scores, device = cls_scores[0].device)
        num_levels = len(cls_scores)

        guided_anchors_list = []
        loc_mask_list = []
        for i in range(num_levels):
            squares = multi_level_squares[i]
            shape_pred = shape_preds[i]
            loc_pred = loc_preds[i]
            guided_anchors, loc_mask = self._get_guided_anchors(squares, shape_pred, loc_pred, use_loc_filter)
            guided_anchors_list.append(guided_anchors)
            loc_mask_list.append(loc_mask)
        
        return multi_level_squares, guided_anchors_list, loc_mask_list


    def _get_guided_anchors(self,
                            squares,
                            shape_pred,
                            loc_pred,
                            use_loc_filter=False):
        
        loc_pred = loc_pred.sigmoid()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(0, 2, 3, 1).float().expand(-1, -1, -1, self.num_anchors)
        mask = mask.view(mask.shape[0], -1)
        # calculate guided anchors
        squares = squares.unsqueeze(0)
        anchor_deltas = shape_pred.permute(0, 2, 3, 1).contiguous().view(
            shape_pred.shape[0], -1, 2)
        zeros = squares[:, :, :2]*0.
        bbox_deltas = torch.cat([zeros, anchor_deltas], dim=2)
        guided_anchors = self.anchor_coder.decode(
            squares, bbox_deltas, wh_ratio_clip=1e-6)

        return guided_anchors, mask


    def forward(self, feat, x):
        pass