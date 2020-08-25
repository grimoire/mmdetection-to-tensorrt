import torch
from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn

from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
import mmdet2trt.ops.util_ops as mm2trt_util

@register_warper("mmdet.models.RPNHead")
class RPNHeadWarper(nn.Module):
    def __init__(self, module):
        super(RPNHeadWarper, self).__init__()
        self.module = module
        self.anchor_generator = build_warper(self.module.anchor_generator)
        self.bbox_coder = build_warper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        self.rpn_nms = BatchedNMS(0.0, self.test_cfg.nms_thr, -1)


    def forward(self, feat, x):
        module = self.module

        cls_scores, bbox_preds = module(feat)
        
        num_levels = len(cls_scores)
        mlvl_anchors = self.anchor_generator(cls_scores, device = cls_scores[0].device)

        mlvl_scores = []
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx] #.squeeze()
            rpn_bbox_pred = bbox_preds[idx] #.squeeze()
            anchors = mlvl_anchors[idx]
            
            scores, proposals = self.bbox_coder(rpn_cls_score, 
                                                rpn_bbox_pred, 
                                                anchors, 
                                                min_num_bboxes = self.test_cfg.nms_pre, 
                                                num_classes = 1,
                                                use_sigmoid_cls = True, 
                                                input_x = x
                                                )

            _, proposals, scores, _ = self.rpn_nms(scores, proposals, self.test_cfg.nms_pre, self.test_cfg.nms_post)
            
            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)
            
        scores = torch.cat(mlvl_scores, dim=1)
        proposals = torch.cat(mlvl_proposals, dim=1)
        
        _, topk_inds = scores.topk(self.test_cfg.max_num, dim=1)
        proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
        
        return proposals
