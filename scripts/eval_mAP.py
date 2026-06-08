"""Standalone mAP evaluation — thin CLI wrapper around drone_perception.benchmark.metrics.

Computes mAP@50 and mAP@50:95 per-class and overall using ONNX Runtime.

Usage:
    python scripts/eval_mAP.py
    python scripts/eval_mAP.py --conf 0.001 --iou 0.65
    python scripts/eval_mAP.py --max-images 50
"""

import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
import onnxruntime as ort

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.drone_perception.benchmark.metrics import compute_map, load_yolo_labels

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]
NC = 10


def letterbox(img, imgsz):
    """Resize + pad image to imgsz×imgsz, maintaining aspect ratio."""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(img, (new_w, new_h))
    dh, dw = imgsz - new_h, imgsz - new_w
    top, bottom = dh // 2, imgsz - new_h - dh // 2
    left, right = dw // 2, imgsz - new_w - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, (r, left, top)


def preprocess(frame, imgsz):
    """Letterbox + HWC→CHW + normalize to [0,1]."""
    img, pad_info = letterbox(frame, imgsz)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = img[np.newaxis, ...]  # add batch dim
    return img, pad_info


def postprocess(output, pad_info, conf_thresh, iou_thresh):
    """YOLO output → (boxes_xyxy, scores, class_ids) in original image coords."""
    ratio, pad_w, pad_h = pad_info
    if output.ndim == 3:
        output = output[0]
    predictions = output.T  # (anchors, 4+nc)

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(int)

    mask = scores >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # xywh → xyxy
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Per-class NMS
    keep = nms(boxes_xyxy, scores, class_ids, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Remove padding, scale to original image coords
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio

    return boxes_xyxy, scores, class_ids


def nms(boxes, scores, class_ids, iou_thresh):
    """Per-class greedy NMS."""
    keep_all = []
    for cls in np.unique(class_ids):
        mask = class_ids == cls
        idx = np.where(mask)[0]
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        order = cls_scores.argsort()[::-1]
        keep = []
        x1, y1, x2, y2 = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            remaining = np.where(iou <= iou_thresh)[0]
            order = order[remaining + 1]
        for k in keep:
            keep_all.append(idx[k])
    return keep_all


@click.command()
@click.option("--weights", default="runs/detect/outputs/train/yolov8n_960/weights/best.onnx",
              type=click.Path(exists=True), help="ONNX weights path.")
@click.option("--data-dir", default="data/visdrone_yolo",
              type=click.Path(exists=True), help="YOLO dataset root directory.")
@click.option("--imgsz", default=960, type=int, help="Model input size.")
@click.option("--conf", default=0.001, type=float, help="Confidence threshold (use 0.001 for mAP).")
@click.option("--iou", default=0.65, type=float, help="NMS IoU threshold.")
@click.option("--max-images", default=0, type=int, help="Max images to evaluate (0=all).")
def main(weights, data_dir, imgsz, conf, iou, max_images):
    """Standalone mAP evaluation using ONNX Runtime — no ultralytics dependency."""
    data_path = Path(data_dir)
    img_dir = data_path / "images" / "val"
    label_dir = data_path / "labels" / "val"

    # Collect all val images
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.png"))
    if max_images > 0:
        img_files = img_files[:max_images]

    click.echo(f"Standalone mAP Evaluation")
    click.echo(f"  Model:  {weights}")
    click.echo(f"  Data:   {data_dir} ({len(img_files)} images)")
    click.echo(f"  imgsz={imgsz}, conf={conf}, iou={iou}")
    click.echo()

    # Init ONNX Runtime
    sess = ort.InferenceSession(weights, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    for _ in range(3):
        sess.run(None, {input_name: dummy})

    all_predictions = {}
    all_ground_truths = {}

    total_inference_ms = 0
    total_postprocess_ms = 0

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem
        frame = cv2.imread(str(img_path))
        if frame is None:
            click.echo(f"[!] Cannot read {img_path}")
            continue
        h, w = frame.shape[:2]

        # Load ground truth
        label_path = label_dir / f"{stem}.txt"
        gt_boxes, gt_classes = load_yolo_labels(str(label_path), w, h)
        all_ground_truths[stem] = (gt_boxes, gt_classes)

        # Inference
        input_tensor, pad_info = preprocess(frame, imgsz)
        t0 = time.perf_counter()
        output = sess.run(None, {input_name: input_tensor})[0]
        inf_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        pred_boxes, pred_scores, pred_classes = postprocess(output, pad_info, conf, iou)
        post_ms = (time.perf_counter() - t1) * 1000

        total_inference_ms += inf_ms
        total_postprocess_ms += post_ms
        all_predictions[stem] = (pred_boxes, pred_scores, pred_classes)

        if (idx + 1) % 50 == 0 or idx == len(img_files) - 1:
            click.echo(f"  [{idx+1}/{len(img_files)}] {stem}: "
                       f"{len(pred_boxes)} det, {inf_ms:.1f}ms inf, {post_ms:.1f}ms post")

    n_images = len(img_files)
    click.echo(f"\n  Avg inference: {total_inference_ms/n_images:.1f}ms, "
               f"Avg postprocess: {total_postprocess_ms/n_images:.1f}ms")

    # Compute mAP using shared module
    click.echo(f"\nComputing mAP...")
    result = compute_map(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        n_classes=NC,
        class_names=CLASS_NAMES,
    )

    click.echo(f"\n{'='*65}")
    click.echo(f"  Standalone mAP Evaluation Results")
    click.echo(f"  Model: {weights}")
    click.echo(f"  Dataset: {data_dir} ({n_images} images)")
    click.echo(f"  imgsz={imgsz}, conf={conf}, iou={iou}")
    click.echo(f"{'='*65}")
    click.echo(f"\n  mAP@50:     {result.map50:.4f}")
    click.echo(f"  mAP@50:95:  {result.map50_95:.4f}")
    click.echo(f"\n  Per-class AP@50:")
    click.echo(f"  {'Class':<20s} {'AP50':>8s} {'mAP50:95':>10s}")
    click.echo(f"  {'-'*40}")
    for cls_id, name in enumerate(CLASS_NAMES):
        click.echo(f"  {name:<20s} {result.per_class_ap50[cls_id]:>8.4f} "
                   f"{result.per_class_map50_95[cls_id]:>10.4f}")
    click.echo(f"  {'-'*40}")
    click.echo(f"  {'MEAN':<20s} {result.map50:>8.4f} {result.map50_95:>10.4f}")
    click.echo()


if __name__ == "__main__":
    main()