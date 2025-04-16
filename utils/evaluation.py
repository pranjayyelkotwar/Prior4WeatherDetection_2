import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# def calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
#     if len(gt_boxes) == 0:
#         return 1.0 if len(pred_boxes) == 0 else 0.0

#     if len(pred_boxes) == 0:
#         return 0.0

#     sorted_indices = torch.argsort(pred_scores, descending=True)
#     pred_boxes = pred_boxes[sorted_indices]
#     pred_labels = pred_labels[sorted_indices]

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     ious = box_iou(pred_boxes.to(device), gt_boxes.to(device))
#     match_indices = (ious >= iou_threshold).nonzero(as_tuple=False)

#     iou_values = ious[match_indices[:, 0], match_indices[:, 1]]
#     sorted_match_indices = match_indices[torch.argsort(iou_values, descending=True)]

#     matched_preds = set()
#     matched_gts = set()
#     tp = torch.zeros(len(pred_boxes))
#     fp = torch.ones(len(pred_boxes))

#     for pi, gi in sorted_match_indices:
#         if pi.item() in matched_preds or gi.item() in matched_gts:
#             continue
#         if pred_labels[pi] == gt_labels[gi]:
#             tp[pi] = 1
#             fp[pi] = 0
#             matched_preds.add(pi.item())
#             matched_gts.add(gi.item())

#     tp_cumsum = torch.cumsum(tp, dim=0)
#     fp_cumsum = torch.cumsum(fp, dim=0)
#     recalls = tp_cumsum / len(gt_boxes)
#     precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

#     precisions = torch.cat([torch.tensor([1.0]), precisions])
#     recalls = torch.cat([torch.tensor([0.0]), recalls])

#     ap = torch.trapz(precisions, recalls).item()
#     return ap

# def evaluate_detection(model, dataloader, device, num_classes=10, iou_threshold=0.5):
#     model.eval()
#     all_preds = [defaultdict(list) for _ in range(num_classes)]
#     all_gts = [defaultdict(list) for _ in range(num_classes)]

#     with torch.no_grad():
#         for images, prior_images, targets in tqdm(dataloader):
#             images = [img.to(device) for img in images]
#             images_tensor = torch.stack(images)

#             predictions = model(images_tensor, None, None)

#             for i, (pred, target) in enumerate(zip(predictions, targets)):

#                 pred_boxes = pred['boxes'].to(device).cpu()
#                 pred_labels = pred['labels'].to(device).cpu()
#                 pred_scores = pred['scores'].to(device).cpu()

#                 gt_boxes = target['boxes'].to(device).cpu()
#                 gt_labels = target['labels'].to(device).cpu()

#                 for c in range(1, num_classes):
#                     mask_pred = pred_labels == c
#                     mask_gt = gt_labels == c

#                     all_preds[c]['boxes'].append(pred_boxes[mask_pred])
#                     all_preds[c]['scores'].append(pred_scores[mask_pred])
#                     all_preds[c]['labels'].append(pred_labels[mask_pred])

#                     all_gts[c]['boxes'].append(gt_boxes[mask_gt])
#                     all_gts[c]['labels'].append(gt_labels[mask_gt])

#     def compute_ap_for_class(c):
#         if not all_preds[c]['boxes']:
#             return c, 0.0
        
#         pred_boxes = torch.cat(all_preds[c]['boxes']) if all_preds[c]['boxes'] else torch.empty((0, 4))
#         pred_scores = torch.cat(all_preds[c]['scores']) if all_preds[c]['scores'] else torch.empty((0,))
#         pred_labels = torch.cat(all_preds[c]['labels']) if all_preds[c]['labels'] else torch.empty((0,), dtype=torch.int64)

#         gt_boxes = torch.cat(all_gts[c]['boxes']) if all_gts[c]['boxes'] else torch.empty((0, 4))
#         gt_labels = torch.cat(all_gts[c]['labels']) if all_gts[c]['labels'] else torch.empty((0,), dtype=torch.int64)

#         ap = calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold)
#         return c, ap

#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(compute_ap_for_class, range(1, num_classes)))

# #     ap_per_class = dict(results)
# #     mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0

# #     return {
# #         'mAP': mAP,
# #         'AP_per_class': ap_per_class
# #     }


# # def evaluate_detection(model, dataloader, device, num_classes=10, iou_threshold=0.5, batch_size=4):
# #     """
# #     Evaluates detection performance with reduced memory usage by processing predictions in batches.
    
# #     Args:
# #         model: The detection model
# #         dataloader: Validation dataloader
# #         device: Computing device (CPU/GPU)
# #         num_classes: Number of object classes
# #         iou_threshold: IoU threshold for considering a detection correct
# #         batch_size: Number of predictions to process at once for AP calculation
# #     """
# #     model.eval()
# #     all_preds = [defaultdict(list) for _ in range(num_classes)]
# #     all_gts = [defaultdict(list) for _ in range(num_classes)]

# #     # First pass: collect all predictions and ground truths
# #     with torch.no_grad():
# #         for images, prior_images, targets in tqdm(dataloader, desc="Collecting predictions"):
# #             # Move only the current batch to GPU
# #             images = [img.to(device) for img in images]
# #             images_tensor = torch.stack(images)

# #             predictions = model(images_tensor, None, None)

# #             # Immediately move results to CPU to free GPU memory
# #             for i, (pred, target) in enumerate(zip(predictions, targets)):
# #                 pred_boxes = pred['boxes'].cpu()
# #                 pred_labels = pred['labels'].cpu()
# #                 pred_scores = pred['scores'].cpu()

# #                 gt_boxes = target['boxes'].cpu()
# #                 gt_labels = target['labels'].cpu()

# #                 for c in range(1, num_classes):
# #                     mask_pred = pred_labels == c
# #                     mask_gt = gt_labels == c

# #                     all_preds[c]['boxes'].append(pred_boxes[mask_pred])
# #                     all_preds[c]['scores'].append(pred_scores[mask_pred])
# #                     all_preds[c]['labels'].append(pred_labels[mask_pred])

# #                     all_gts[c]['boxes'].append(gt_boxes[mask_gt])
# #                     all_gts[c]['labels'].append(gt_labels[mask_gt])
            
# #             # Clear GPU cache after each batch
# #             if torch.cuda.is_available():
# #                 torch.cuda.empty_cache()
# #     # Second pass: compute AP for each class in batches
# #     def compute_ap_for_class(c):
# #         if not all_preds[c]['boxes']:
# #             return c, 0.0
        
# #         # Concatenate all predictions and ground truths for this class
# #         pred_boxes = torch.cat(all_preds[c]['boxes']) if all_preds[c]['boxes'] else torch.empty((0, 4))
# #         pred_scores = torch.cat(all_preds[c]['scores']) if all_preds[c]['scores'] else torch.empty((0,))
# #         pred_labels = torch.cat(all_preds[c]['labels']) if all_preds[c]['labels'] else torch.empty((0,), dtype=torch.int64)

# #         gt_boxes = torch.cat(all_gts[c]['boxes']) if all_gts[c]['boxes'] else torch.empty((0, 4))
# #         gt_labels = torch.cat(all_gts[c]['labels']) if all_gts[c]['labels'] else torch.empty((0,), dtype=torch.int64)
        
# #         # Process in smaller batches if the number of predictions is large
# #         if len(pred_boxes) > batch_size:
# #             # Sort by scores first (required for AP calculation)
# #             sorted_indices = torch.argsort(pred_scores, descending=True)
# #             pred_boxes = pred_boxes[sorted_indices]
# #             pred_labels = pred_labels[sorted_indices]
# #             pred_scores = pred_scores[sorted_indices]
            
# #             # Process all GTs but batch the predictions
# #             tp = torch.zeros(len(pred_boxes))
# #             fp = torch.ones(len(pred_boxes))
            
# #             # Process predictions in batches
# #             for i in range(0, len(pred_boxes), batch_size):
# #                 end_idx = min(i + batch_size, len(pred_boxes))
# #                 batch_pred_boxes = pred_boxes[i:end_idx]
# #                 batch_pred_labels = pred_labels[i:end_idx]
                
# #                 # Compute IoUs for this batch
# #                 device_for_iou = 'cpu'  # Using CPU for IoU calculation to save GPU memory
# #                 ious = box_iou(batch_pred_boxes.to(device_for_iou), gt_boxes.to(device_for_iou))
                
# #                 # Mark TP/FP for this batch
# #                 for j in range(len(batch_pred_boxes)):
# #                     pred_idx = i + j
# #                     # Find max IoU with matching class
# #                     max_iou = 0
# #                     max_gt_idx = -1
                    
# #                     for gt_idx in range(len(gt_boxes)):
# #                         if batch_pred_labels[j] == gt_labels[gt_idx] and ious[j, gt_idx] > max_iou:
# #                             max_iou = ious[j, gt_idx]
# #                             max_gt_idx = gt_idx
                    
# #                     # If we found a match above threshold
# #                     if max_iou >= iou_threshold:
# #                         tp[pred_idx] = 1
# #                         fp[pred_idx] = 0
            
# #             # Calculate AP from TP/FP
# #             tp_cumsum = torch.cumsum(tp, dim=0)
# #             fp_cumsum = torch.cumsum(fp, dim=0)
# #             recalls = tp_cumsum / (len(gt_boxes) if len(gt_boxes) > 0 else 1)
# #             precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
# #             # Add starting point for AP calculation
# #             precisions = torch.cat([torch.tensor([1.0]), precisions])
# #             recalls = torch.cat([torch.tensor([0.0]), recalls])
            
# #             ap = torch.trapz(precisions, recalls).item()
# #             return c, ap
# #         else:
# #             # For small number of predictions, use the original function
# #             ap = calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold)
# #             return c, ap


#     # Compute AP for each class sequentially to save memory
#     ap_per_class = {}
#     for c in range(1, num_classes):
#         class_id, ap = compute_ap_for_class(c)
#         ap_per_class[class_id] = ap
#         # Clear memory after each class
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0

#     return {
#         'mAP': mAP,
#         'AP_per_class': ap_per_class
#     }


import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from collections import defaultdict
import gc


def chunked_box_iou(boxes1, boxes2, chunk_size=64):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if boxes1.size(1) != 4 or boxes2.size(1) != 4:
        raise ValueError("Input tensors must have shape [N, 4] representing bounding boxes.")
    if boxes1.size(0) == 0 or boxes2.size(0) == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))  # Return empty IoU matrix if inputs are empty

    device = boxes1.device  # Ensure tensors are on the same device
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)

    iou_matrix = torch.zeros((boxes1.size(0), boxes2.size(0)), device=device)
    for i in range(0, boxes1.size(0), chunk_size):
        chunk = boxes1[i:i + chunk_size]
        iou_matrix[i:i + chunk_size] = box_iou(chunk, boxes2)
    return iou_matrix


def calculate_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0

    if len(pred_boxes) == 0:
        return 0.0

    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]

    ious = chunked_box_iou(pred_boxes, gt_boxes)
    match_indices = (ious >= iou_threshold).nonzero(as_tuple=False)

    iou_values = ious[match_indices[:, 0], match_indices[:, 1]]
    sorted_match_indices = match_indices[torch.argsort(iou_values, descending=True)]

    matched_preds = set()
    matched_gts = set()
    tp = torch.zeros(len(pred_boxes))
    fp = torch.ones(len(pred_boxes))

    for pi, gi in sorted_match_indices:
        if pi.item() in matched_preds or gi.item() in matched_gts:
            continue
        if pred_labels[pi] == gt_labels[gi]:
            tp[pi] = 1
            fp[pi] = 0
            matched_preds.add(pi.item())
            matched_gts.add(gi.item())

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])

    ap = torch.trapz(precisions, recalls).item()
    return ap


def evaluate_detection_streaming(model, dataloader, device, num_classes=8, iou_threshold=0.5, amp=True):
    """
    Evaluate detection performance with memory-efficient streaming evaluation and mixed precision
    """
    model.eval()
    ap_sums = defaultdict(float)
    ap_counts = defaultdict(int)

    with torch.no_grad():
        for images, prior_images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device, non_blocking=True) for img in images]

            # Use mixed precision for inference
            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(torch.stack(images), None, None)

            for pred, target in zip(preds, targets):
                pred_boxes = pred['boxes'].detach()
                pred_labels = pred['labels'].detach()
                pred_scores = pred['scores'].detach()
                gt_boxes = target['boxes'].detach()
                gt_labels = target['labels'].detach()

                # Calculate AP per class
                for c in range(1, num_classes):  # Skip background class (0)
                    if not ((pred_labels == c).any() or (gt_labels == c).any()):
                        continue

                    mask_pred = pred_labels == c
                    mask_gt = gt_labels == c
                    
                    # Skip if no predictions or ground truth for this class
                    if not mask_pred.any() and not mask_gt.any():
                        continue
                    
                    # Skip empty ground truth
                    if not mask_gt.any():
                        ap_sums[c] += 0.0
                        ap_counts[c] += 1
                        continue
                    
                    # Handle empty predictions separately
                    if not mask_pred.any():
                        ap_sums[c] += 0.0
                        ap_counts[c] += 1
                        continue
                    
                    ap = calculate_ap(
                        pred_boxes[mask_pred],
                        pred_scores[mask_pred],
                        pred_labels[mask_pred],
                        gt_boxes[mask_gt],
                        gt_labels[mask_gt],
                        iou_threshold
                    )
                    ap_sums[c] += ap
                    ap_counts[c] += 1

                # Manually release per-sample GPU memory
                del pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
                torch.cuda.empty_cache()

            del preds, images
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate final AP per class and mAP
    final_ap = {c: ap_sums[c] / ap_counts[c] if ap_counts[c] > 0 else 0.0 for c in range(1, num_classes)}
    mAP = np.mean(list(final_ap.values())) if final_ap else 0.0
    return {'mAP': mAP, 'AP_per_class': final_ap}
