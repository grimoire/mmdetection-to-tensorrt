import torch
from mmdet2trt.models.builder import register_wrapper
from torch import nn


@register_wrapper('mmdet.models.utils.SinePositionalEncoding')
class SinePositionalEncodingWraper(nn.Module):

    def __init__(self, module):
        super(SinePositionalEncodingWraper, self).__init__()
        self.module = module

    def forward(self, mask):
        module = self.module
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if module.normalize:
            y_embed = (y_embed + module.offset) / \
                      (y_embed[:, -1:, :] + module.eps) * module.scale
            x_embed = (x_embed + module.offset) / \
                      (x_embed[:, :, -1:] + module.eps) * module.scale
        dim_t = torch.arange(
            module.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = module.temperature**(2 * (dim_t // 2) / module.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        C = pos_x.shape[-1]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, C)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, C)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
