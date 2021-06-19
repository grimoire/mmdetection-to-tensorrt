import torch
from torch import nn
from torchvision.ops import nms as tv_nms


class BatchedNMS(nn.Module):

    def __init__(self, scoreThreshold, iouThreshold, backgroundLabelId=-1):
        super(BatchedNMS, self).__init__()
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.backgroundLabelId = backgroundLabelId

    def forward(self, scores, bboxes, topK, keepTopK):
        topK = min(scores.shape[1], topK)
        keepTopK = min(scores.shape[1], keepTopK)
        num_detections = []
        nmsed_bboxes = []
        nmsed_scores = []
        nmsed_classes = []

        for batch_idx in range(scores.shape[0]):
            num_detection = scores.new_tensor([0], dtype=torch.int32)
            nmsed_bbox = []
            nmsed_score = []
            nmsed_class = []
            for cls_idx in range(scores.shape[2]):
                if cls_idx == self.backgroundLabelId:
                    continue

                # bbox = bboxes[batch_idx, :, cls_idx, :]
                if bboxes.shape[2] != 1:
                    bbox = bboxes[batch_idx, :, cls_idx, :]
                else:
                    bbox = bboxes[batch_idx, :, 0, :]
                score = scores[batch_idx, :, cls_idx]

                if self.scoreThreshold > 0.:
                    score_mask = score > self.scoreThreshold
                    if not score_mask.any():
                        continue
                    bbox = bbox[score_mask, :]
                    score = score[score_mask]

                if len(score.shape) == 2:
                    score = score.squeeze(1)
                score, topk_inds = score.topk(min(score.shape[0], topK))
                # bbox = bbox[topk_inds, :]
                bbox = bbox.index_select(0, topk_inds)
                # score = score[topk_inds]
                # score[score < self.scoreThreshold] = 0.

                nms_idx = tv_nms(bbox, score, self.iouThreshold)
                # nms_idx = ops_nms(torch.cat([bbox,score.unsqueeze(1)], dim=1)
                # , self.iouThreshold)
                num_detection += nms_idx.shape[0]
                bbox = bbox.index_select(0, nms_idx)
                score = score.index_select(0, nms_idx)

                nmsed_bbox.append(bbox)
                # score = score[nms_idx]
                nmsed_score.append(score)
                nmsed_class.append(torch.ones_like(score) * cls_idx)
            if len(nmsed_score) > 0:
                nmsed_bbox = torch.cat(nmsed_bbox, dim=0)
                nmsed_score = torch.cat(nmsed_score, dim=0)
                nmsed_class = torch.cat(nmsed_class, dim=0)
            else:
                nmsed_bbox = scores.new_empty((0, 4))
                nmsed_score = scores.new_empty((0, ))
                nmsed_class = scores.new_empty((0, ))

            num_nms = nmsed_score.shape[0]

            if scores.shape[2] != 1:
                _, topk_inds = nmsed_score.topk(min(num_nms, keepTopK))
                # nmsed_bbox = nmsed_bbox[topk_inds,:]
                nmsed_bbox = nmsed_bbox.index_select(0, topk_inds)
                # nmsed_score = nmsed_score[topk_inds]
                nmsed_score = nmsed_score.index_select(0, topk_inds)
                # nmsed_class = nmsed_class[topk_inds]
                nmsed_class = nmsed_class.index_select(0, topk_inds)

            if num_nms < keepTopK:
                nmsed_bbox = torch.nn.functional.pad(
                    nmsed_bbox,
                    pad=(0, 0, 0, keepTopK - num_nms),
                    mode='constant',
                    value=0)
                nmsed_score = torch.nn.functional.pad(
                    nmsed_score,
                    pad=(0, keepTopK - num_nms),
                    mode='constant',
                    value=0)
                nmsed_class = torch.nn.functional.pad(
                    nmsed_class,
                    pad=(0, keepTopK - num_nms),
                    mode='constant',
                    value=-1)
            else:
                if num_detection > keepTopK:
                    num_detection = num_detection.new_tensor([keepTopK])
                nmsed_bbox = nmsed_bbox[:keepTopK, :]
                nmsed_score = nmsed_score[:keepTopK]
                nmsed_class = nmsed_class[:keepTopK]

            num_detections.append(num_detection)
            nmsed_bboxes.append(nmsed_bbox)
            nmsed_scores.append(nmsed_score)
            nmsed_classes.append(nmsed_class)

        num_detections = torch.stack(num_detections, dim=0)
        nmsed_bboxes = torch.stack(nmsed_bboxes, dim=0)
        nmsed_scores = torch.stack(nmsed_scores, dim=0)
        nmsed_classes = torch.stack(nmsed_classes, dim=0)

        return num_detections, nmsed_bboxes, nmsed_scores, nmsed_classes
