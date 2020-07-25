import torch
import numpy as np

def convert_to_mmdet_result(results, num_classes=80):
    results = [r.cpu().detach() for r in results]
    batch_size = results[0].shape[0]

    bbox_scores = torch.cat([results[1],results[2].unsqueeze(-1)], dim=-1)

    mmdet_results = []
    for batch_id in range(batch_size):
        mmdet_result = [[] for _ in range(num_classes)]
        num_detected = results[0][batch_id].item()
        for n in range(num_detected):
            cls_id = int(results[3][batch_id][n])
            bbox = bbox_scores[batch_id][n]
            mmdet_result[cls_id].append(bbox)
        mmdet_result = [torch.stack(bb).numpy() if len(bb)>0 else np.empty((0,5)) for bb in mmdet_result]
        mmdet_results.append(mmdet_result)

    return mmdet_results



