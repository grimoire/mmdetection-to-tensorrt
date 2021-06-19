import torch
from mmdet2trt.core.bbox.transforms import batched_bbox_cxcywh_to_xyxy
from mmdet2trt.models.builder import register_wraper
from torch import nn
from torch.nn import functional as F


@register_wraper('mmdet.models.dense_heads.TransformerHead')
class TransformerHeadWraper(nn.Module):

    def __init__(self, module):
        super(TransformerHeadWraper, self).__init__()
        self.module = module
        self.test_cfg = module.test_cfg

    def module_forward(self, feats, x):
        module = self.module
        batch_size, _, input_img_h, input_img_w = x.shape
        masks = feats[0].new_zeros((batch_size, input_img_h, input_img_w))

        cls_scores = []
        bbox_preds = []
        for feat in feats:
            feat = module.input_proj(feat)
            masks_interp = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            pos_embed = module.positional_encoding(
                masks_interp)  # [bs, embed_dim, h, w]
            # outs_dec: [nb_dec, bs, num_query, embed_dim]
            outs_dec, _ = module.transformer(feat, masks_interp,
                                             module.query_embedding.weight,
                                             pos_embed)

            all_cls_scores = module.fc_cls(outs_dec)
            all_bbox_preds = module.fc_reg(
                module.activate(module.reg_ffn(outs_dec))).sigmoid()

            cls_scores.append(all_cls_scores)
            bbox_preds.append(all_bbox_preds)

        return cls_scores, bbox_preds

    def forward(self, feats, x):
        img_shape = x.shape[2:]

        cls_scores, bbox_preds = self.module_forward(feats, x)

        cls_score = cls_scores[-1][0]
        bbox_pred = bbox_preds[-1][0]

        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        bbox_pred = batched_bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred.clamp(min=0, max=1)
        bbox_pred0 = bbox_pred[:, :, 0] * img_shape[1]
        bbox_pred1 = bbox_pred[:, :, 1] * img_shape[0]
        bbox_pred2 = bbox_pred[:, :, 2] * img_shape[1]
        bbox_pred3 = bbox_pred[:, :, 3] * img_shape[0]
        bbox_pred = torch.stack(
            [bbox_pred0, bbox_pred1, bbox_pred2, bbox_pred3], dim=-1)

        num_dets = (scores[:, :1] * 0).int() + scores.shape[1]

        return num_dets, bbox_pred, scores, det_labels
