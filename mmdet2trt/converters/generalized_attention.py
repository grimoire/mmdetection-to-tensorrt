import math

import mmdet2trt
import numpy as np
import torch
import torch.nn.functional as F
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter


def get_position_embedding(self,
                           x_q,
                           x_kv,
                           q_stride,
                           kv_stride,
                           feat_dim,
                           wave_length=1000):
    h_idxs = mmdet2trt.ops.util_ops.arange_by_input(x_q, 2)
    h_idxs = h_idxs.unsqueeze(1) * q_stride
    w_idxs = mmdet2trt.ops.util_ops.arange_by_input(x_q, 3)
    w_idxs = w_idxs.unsqueeze(1) * q_stride

    h_kv_idxs = mmdet2trt.ops.util_ops.arange_by_input(x_kv, 2)
    h_kv_idxs = h_kv_idxs.unsqueeze(1) * kv_stride
    w_kv_idxs = mmdet2trt.ops.util_ops.arange_by_input(x_kv, 3)
    w_kv_idxs = w_kv_idxs.unsqueeze(1) * kv_stride

    # (h, h_kv, 1)
    h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
    h_diff *= self.position_magnitude

    # (w, w_kv, 1)
    w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
    w_diff *= self.position_magnitude

    feat_range = torch.arange(0, feat_dim / 4, device=x_q.device)

    dim_mat = x_q.new_tensor([wave_length])
    dim_mat = dim_mat**((4. / feat_dim) * feat_range)
    dim_mat = dim_mat.view((1, 1, -1))

    embedding_x = torch.cat(
        ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

    embedding_y = torch.cat(
        ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

    return embedding_x, embedding_y


@tensorrt_converter(
    'mmcv.cnn.bricks.GeneralizedAttention.forward', is_real=False)
def convert_GeneralizeAttention(ctx):
    self = ctx.method_args[0]
    x_input = ctx.method_args[1]
    output = ctx.method_return

    num_heads = self.num_heads

    # use empirical_attention
    if self.q_downsample is not None:
        x_q = self.q_downsample(x_input)
    else:
        x_q = x_input
    n, _, h, w = x_q.shape

    if self.kv_downsample is not None:
        x_kv = self.kv_downsample(x_input)
    else:
        x_kv = x_input
    _, _, h_kv, w_kv = x_kv.shape

    if self.attention_type[0] or self.attention_type[1]:
        proj_query = self.query_conv(x_q).view(
            (n, num_heads, self.qk_embed_dim, h * w))
        proj_query = proj_query.permute(0, 1, 3, 2)

    if self.attention_type[0] or self.attention_type[2]:
        proj_key = self.key_conv(x_kv).view(
            (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

    if self.attention_type[1] or self.attention_type[3]:
        position_embed_x, position_embed_y = get_position_embedding(
            self, x_q, x_kv, self.q_stride, self.kv_stride,
            self.position_embedding_dim)
        # (n, num_heads, w, w_kv, dim)
        position_feat_x = self.appr_geom_fc_x(position_embed_x).\
            view(1, w, w_kv, num_heads, self.qk_embed_dim).\
            permute(0, 3, 1, 2, 4)

        # (n, num_heads, h, h_kv, dim)
        position_feat_y = self.appr_geom_fc_y(position_embed_y).\
            view(1, h, h_kv, num_heads, self.qk_embed_dim).\
            permute(0, 3, 1, 2, 4)

        position_feat_x /= math.sqrt(2)
        position_feat_y /= math.sqrt(2)

        # accelerate for saliency only
    if (np.sum(self.attention_type) == 1) and self.attention_type[2]:
        appr_bias = self.appr_bias.\
            view(1, num_heads, 1, self.qk_embed_dim)

        energy = torch.matmul(appr_bias, proj_key).\
            view(n, num_heads, 1, h_kv * w_kv)

        h = 1
        w = 1
    else:
        # (n, num_heads, h*w, h_kv*w_kv), query before key, 540mb for
        if not self.attention_type[0]:
            energy = x_input.new_zeros(n, num_heads, h, w, h_kv, w_kv)

        # attention_type[0]: appr - appr
        # attention_type[1]: appr - position
        # attention_type[2]: bias - appr
        # attention_type[3]: bias - position
        if self.attention_type[0] or self.attention_type[2]:
            if self.attention_type[0] and self.attention_type[2]:
                appr_bias = self.appr_bias.\
                    view(1, num_heads, 1, self.qk_embed_dim)
                energy = torch.matmul(proj_query + appr_bias, proj_key).\
                    view(n, num_heads, h, w, h_kv, w_kv)

            elif self.attention_type[0]:
                energy = torch.matmul(proj_query, proj_key).\
                    view(n, num_heads, h, w, h_kv, w_kv)

            elif self.attention_type[2]:
                appr_bias = self.appr_bias.\
                    view(1, num_heads, 1, self.qk_embed_dim)

                energy += torch.matmul(appr_bias, proj_key).\
                    view(n, num_heads, 1, 1, h_kv, w_kv)

        if self.attention_type[1] or self.attention_type[3]:
            if self.attention_type[1] and self.attention_type[3]:
                geom_bias = self.geom_bias.\
                    view(1, num_heads, 1, self.qk_embed_dim)

                proj_query_reshape = (proj_query + geom_bias).\
                    view(n, num_heads, h, w, self.qk_embed_dim)

                energy_x = torch.matmul(
                    proj_query_reshape.permute(0, 1, 3, 2, 4),
                    position_feat_x.permute(0, 1, 2, 4, 3))
                energy_x = energy_x.\
                    permute(0, 1, 3, 2, 4).unsqueeze(4)

                energy_y = torch.matmul(proj_query_reshape,
                                        position_feat_y.permute(0, 1, 2, 4, 3))
                energy_y = energy_y.unsqueeze(5)

                energy += energy_x + energy_y

            elif self.attention_type[1]:
                proj_query_reshape = proj_query.\
                    view(n, num_heads, h, w, self.qk_embed_dim)
                proj_query_reshape = proj_query_reshape.\
                    permute(0, 1, 3, 2, 4)
                position_feat_x_reshape = position_feat_x.\
                    permute(0, 1, 2, 4, 3)
                position_feat_y_reshape = position_feat_y.\
                    permute(0, 1, 2, 4, 3)

                energy_x = torch.matmul(proj_query_reshape,
                                        position_feat_x_reshape)
                energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)

                energy_y = torch.matmul(proj_query_reshape,
                                        position_feat_y_reshape)
                energy_y = energy_y.unsqueeze(5)

                energy += energy_x + energy_y

            elif self.attention_type[3]:
                geom_bias = self.geom_bias.\
                    view(1, num_heads, self.qk_embed_dim, 1)

                position_feat_x_reshape = position_feat_x.\
                    view(n, num_heads, w*w_kv, self.qk_embed_dim)

                position_feat_y_reshape = position_feat_y.\
                    view(n, num_heads, h * h_kv, self.qk_embed_dim)

                energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
                energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)

                energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
                energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)

                energy += energy_x + energy_y

        energy = energy.view(n, num_heads, h * w, h_kv * w_kv)

    if self.spatial_range >= 0:
        cur_local_constraint_map = \
            self.local_constraint_map[:h, :w, :h_kv, :w_kv].\
            contiguous().\
            view(1, 1, h*w, h_kv*w_kv)

        energy = energy.masked_fill_(cur_local_constraint_map, float('-inf'))

    attention = F.softmax(energy, 3)

    proj_value = self.value_conv(x_kv)
    proj_value_reshape = proj_value.\
        view((n, num_heads, self.v_dim, h_kv * w_kv)).\
        permute(0, 1, 3, 2)

    out = torch.matmul(attention, proj_value_reshape).\
        permute(0, 1, 3, 2).\
        contiguous().\
        view(n, self.v_dim * self.num_heads, h, w)

    out = self.proj_conv(out)

    # output is downsampled, upsample back to input size
    if self.q_downsample is not None:
        out = F.interpolate(
            out, size=x_input.shape[2:], mode='bilinear', align_corners=False)

    out = self.gamma * out + x_input

    output._trt = out._trt
    ctx.method_return = output
