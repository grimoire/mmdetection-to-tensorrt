import torch
from mmdet2trt.models.builder import register_wraper
from torch import nn


def yolodecoder_batched(bboxes, pred_bboxes, stride):
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    # Get outputs x, y
    x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
    y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
    w_pred = torch.exp(pred_bboxes[..., 2]) * w
    h_pred = torch.exp(pred_bboxes[..., 3]) * h

    decoded_bboxes = torch.stack(
        (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
         x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
        dim=-1)

    return decoded_bboxes


@register_wraper('mmdet.core.bbox.coder.YOLOBBoxCoder')
class YOLOBBoxCoderWraper(nn.Module):

    def __init__(self, module):
        super(YOLOBBoxCoderWraper, self).__init__()

    def forward(self,
                cls_scores,
                bbox_preds,
                anchors,
                min_num_bboxes,
                num_classes,
                use_sigmoid_cls,
                input_x=None):
        pass

    def decode(self, bboxes, pred_bboxes, stride):
        return yolodecoder_batched(bboxes, pred_bboxes, stride)
