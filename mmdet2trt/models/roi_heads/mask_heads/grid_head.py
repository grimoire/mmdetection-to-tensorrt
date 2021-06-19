import torch
from mmdet2trt.models.builder import register_wraper
from torch import nn


@register_wraper('mmdet.models.roi_heads.mask_heads.grid_head.GridHead')
class GridHeadWraper(nn.Module):

    def __init__(self, module, test_cfg):
        super(GridHeadWraper, self).__init__()
        self.module = module
        self.test_cfg = test_cfg
        self.module.test_mode = True

    def forward(self, x):
        return self.module(x)

    def get_bboxes(self, cls_scores, det_bboxes, grid_pred, img_shape=None):
        module = self.module
        grid_pred = grid_pred.sigmoid()

        R, c, h, w = grid_pred.shape

        grid_pred = grid_pred.view(R, c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=2)
        pred_scores = pred_scores.view(-1)
        xs = pred_position % w
        ys = pred_position // w

        sub_region_xs = [
            module.sub_regions[i][0] for i in range(module.grid_points)
        ]
        sub_region_ys = [
            module.sub_regions[i][1] for i in range(module.grid_points)
        ]
        sub_region_xs = xs.new_tensor(sub_region_xs)
        sub_region_ys = ys.new_tensor(sub_region_ys)

        xs = xs + sub_region_xs
        ys = ys + sub_region_ys
        xs = xs.view(-1)
        ys = ys.view(-1)

        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, c), [pred_scores, xs, ys]))

        # get expanded pos_bboxes
        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = (det_bboxes[:, 0, None] - widths / 2)
        y1 = (det_bboxes[:, 1, None] - heights / 2)

        # map the grid point to the absolute coordinates
        abs_xs = (xs.float() + 0.5) / (w * 1.0) * widths + x1
        abs_ys = (ys.float() + 0.5) / (h * 1.0) * heights + y1

        x1_inds = [i for i in range(module.grid_size)]
        y1_inds = [i * module.grid_size for i in range(module.grid_size)]
        x2_inds = [
            module.grid_points - module.grid_size + i
            for i in range(module.grid_size)
        ]
        y2_inds = [(i + 1) * module.grid_size - 1
                   for i in range(module.grid_size)]

        # voting of all grid points on some boundary
        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x1_inds].sum(dim=1, keepdim=True))
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y1_inds].sum(dim=1, keepdim=True))
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x2_inds].sum(dim=1, keepdim=True))
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y2_inds].sum(dim=1, keepdim=True))

        if img_shape is not None:
            bboxes_x1 = bboxes_x1.clamp(min=0, max=img_shape[1])
            bboxes_x2 = bboxes_x2.clamp(min=0, max=img_shape[1])
            bboxes_y1 = bboxes_y1.clamp(min=0, max=img_shape[0])
            bboxes_y2 = bboxes_y2.clamp(min=0, max=img_shape[0])

        bbox_res = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2],
                             dim=1)

        return cls_scores, bbox_res
