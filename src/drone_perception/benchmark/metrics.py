"""mAP (mean Average Precision) computation for object detection.

Extracted from scripts/eval_mAP.py into a reusable module.
Computes COCO-style mAP@50 and mAP@50:95 with per-class breakdowns.

Usage:
    from drone_perception.benchmark.metrics import compute_map, load_yolo_labels

    predictions = {"img001": (boxes, scores, class_ids), ...}
    ground_truths = {"img001": (gt_boxes, gt_class_ids), ...}
    accuracy = compute_map(predictions, ground_truths, n_classes=10)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


# VisDrone class names (default)
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]


@dataclass
class AccuracyResult:
    """Result from mAP computation."""
    map50: float
    map50_95: float
    per_class_ap50: np.ndarray   # shape (n_classes,)
    per_class_map50_95: np.ndarray  # shape (n_classes,)
    n_classes: int
    class_names: list[str]


def load_yolo_labels(label_path: str, img_w: int, img_h: int):
    """Load YOLO format labels into pixel-coordinate bounding boxes.

    Args:
        label_path: Path to .txt label file (class x_center y_center width height).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple of (boxes_xyxy, class_ids) where boxes_xyxy has shape (N, 4)
        and class_ids has shape (N,). Returns empty arrays if file missing.
    """
    boxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return np.empty((0, 4)), np.empty(0, dtype=int)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)
    if not boxes:
        return np.empty((0, 4)), np.empty(0, dtype=int)
    return np.array(boxes, dtype=np.float64), np.array(class_ids, dtype=int)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using 11-point interpolation (PASCAL VOC style)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = mrec[1:] >= t
        if mask.any():
            ap += mpre[1:][mask].max()
    return ap / 11.0


def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between all pairs of boxes_a (N,4) and boxes_b (M,4).

    Returns (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])  # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _match_class_predictions(
    gt_boxes_cls: np.ndarray,
    pred_boxes_cls: np.ndarray,
    pred_scores_cls: np.ndarray,
    iou_thresh: float,
    n_gt_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to ground truths for one class at one IoU threshold.

    Returns (scores_sorted, tp_sorted) arrays sorted by confidence descending.
    Uses vectorized IoU computation for speed.
    """
    if n_gt_total == 0 and len(pred_boxes_cls) == 0:
        return np.array([]), np.array([])
    if len(pred_boxes_cls) == 0:
        return np.array([]), np.array([])
    if n_gt_total == 0:
        # All predictions are false positives
        order = pred_scores_cls.argsort()[::-1]
        return pred_scores_cls[order], np.zeros(len(pred_scores_cls))

    # Sort by confidence descending
    order = pred_scores_cls.argsort()[::-1]
    pred_boxes_cls = pred_boxes_cls[order]
    pred_scores_cls = pred_scores_cls[order]

    # Compute all pairwise IoU (N_pred x N_gt)
    iou_matrix = _batch_iou(pred_boxes_cls, gt_boxes_cls)

    gt_matched = np.zeros(len(gt_boxes_cls), dtype=bool)
    tp = np.zeros(len(pred_boxes_cls))

    for i in range(len(pred_boxes_cls)):
        # Find best unmatched GT
        iou_row = iou_matrix[i].copy()
        iou_row[gt_matched] = 0  # zero out matched GTs
        best_j = np.argmax(iou_row)
        best_iou = iou_row[best_j]
        if best_iou >= iou_thresh:
            tp[i] = 1
            gt_matched[best_j] = True

    return pred_scores_cls, tp


def compute_map(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truths: dict[str, tuple[np.ndarray, np.ndarray]],
    n_classes: int = 10,
    class_names: list[str] | None = None,
    max_dets_per_class: int = 300,
) -> AccuracyResult:
    """Compute mAP@50 and mAP@50:95 across all images.

    Args:
        predictions: {img_id: (boxes_xyxy, scores, class_ids)}
                     boxes_xyxy shape (N, 4), scores shape (N,), class_ids shape (N,)
        ground_truths: {img_id: (boxes_xyxy, class_ids)}
                       boxes_xyxy shape (M, 4), class_ids shape (M,)
        n_classes: Number of classes.
        class_names: Optional list of class name strings.
        max_dets_per_class: Keep only top-K detections per class per image
                           before mAP matching. Matches COCO eval convention.
                           Low-confidence detections beyond this limit have
                           negligible impact on mAP.

    Returns:
        AccuracyResult with mAP50, mAP50:95, and per-class breakdowns.
    """
    if class_names is None:
        class_names = VISDRONE_CLASSES[:n_classes]

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 10 thresholds: 0.5, 0.55, ..., 0.95
    ap_per_class = np.zeros((n_classes, len(iou_thresholds)))

    for cls_id in range(n_classes):
        for t_idx, iou_thresh in enumerate(iou_thresholds):
            # Collect all predictions and ground truths for this class
            score_list = []
            tp_list = []
            n_gt_total = 0

            for img_id in sorted(ground_truths.keys()):
                gt_boxes, gt_classes = ground_truths[img_id]
                gt_mask = gt_classes == cls_id
                gt_boxes_cls = gt_boxes[gt_mask] if gt_mask.any() else np.empty((0, 4))
                n_gt = len(gt_boxes_cls)
                n_gt_total += n_gt

                pred_boxes, pred_scores, pred_classes = predictions.get(
                    img_id, (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))
                )
                pred_mask = pred_classes == cls_id
                pred_boxes_cls = pred_boxes[pred_mask] if pred_mask.any() else np.empty((0, 4))
                pred_scores_cls = pred_scores[pred_mask] if pred_mask.any() else np.empty(0)

                # Keep only top-K detections per class (matches COCO eval convention)
                if len(pred_scores_cls) > max_dets_per_class:
                    top_k_idx = np.argsort(pred_scores_cls)[::-1][:max_dets_per_class]
                    pred_boxes_cls = pred_boxes_cls[top_k_idx]
                    pred_scores_cls = pred_scores_cls[top_k_idx]

                if n_gt == 0 and len(pred_boxes_cls) == 0:
                    continue

                scores, tps = _match_class_predictions(
                    gt_boxes_cls, pred_boxes_cls, pred_scores_cls,
                    iou_thresh, n_gt,
                )
                score_list.extend(scores.tolist())
                tp_list.extend(tps.tolist())

            if n_gt_total == 0:
                ap_per_class[cls_id, t_idx] = float('nan')
                continue

            if len(score_list) == 0:
                ap_per_class[cls_id, t_idx] = 0.0
                continue

            score_arr = np.array(score_list)
            tp_arr = np.array(tp_list)
            order = score_arr.argsort()[::-1]
            tp_arr = tp_arr[order]

            cum_tp = np.cumsum(tp_arr)
            cum_fp = np.cumsum(1 - tp_arr)
            recall = cum_tp / n_gt_total
            precision = cum_tp / (cum_tp + cum_fp)

            ap_per_class[cls_id, t_idx] = compute_ap(recall, precision)

    # mAP@50 = mean of AP at IoU=0.5
    ap50 = ap_per_class[:, 0]
    # mAP@50:95 = mean across all classes and all IoU thresholds
    ap50_clean = np.nan_to_num(ap50, nan=0.0)
    ap_all_clean = np.nan_to_num(ap_per_class, nan=0.0)

    map50 = float(ap50_clean.mean())
    map50_95 = float(ap_all_clean.mean())
    per_class_ap50 = ap50_clean
    per_class_map50_95 = ap_all_clean.mean(axis=1)

    return AccuracyResult(
        map50=map50,
        map50_95=map50_95,
        per_class_ap50=per_class_ap50,
        per_class_map50_95=per_class_map50_95,
        n_classes=n_classes,
        class_names=class_names,
    )