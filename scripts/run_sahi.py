"""CLI: Run SAHI (Slicing Aided Hyper Inference) with before/after comparison.

Slices the image into overlapping tiles for better small-object detection,
then shows detection count with vs without SAHI.

Usage:
    python scripts/run_sahi.py --weights best.onnx --source image.jpg
    python scripts/run_sahi.py --weights best.pt --backend pytorch --source video.mp4
    python scripts/run_sahi.py --weights best.onnx --source image.jpg --slice-size 480 --slice-overlap 0.3
"""

import click
import cv2
import numpy as np
from pathlib import Path

from src.drone_perception.config import Config
from src.drone_perception.inference.sahi_backend import SahiBackend

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def _build_backend(backend_name: str, weights: str, imgsz: int, conf: float, iou: float):
    """Construct the requested inference backend."""
    if backend_name == "pytorch":
        from src.drone_perception.inference.pytorch_backend import PyTorchBackend
        return PyTorchBackend(weights=weights, imgsz=imgsz, conf=conf, iou=iou)
    elif backend_name == "onnx":
        from src.drone_perception.inference.onnx_backend import ONNXBackend
        return ONNXBackend(onnx_path=weights, imgsz=imgsz, conf=conf, iou=iou)
    elif backend_name == "tensorrt":
        from src.drone_perception.inference.tensorrt_backend import TensorRTBackend
        return TensorRTBackend(engine_path=weights, imgsz=imgsz, conf=conf, iou=iou)


def _annotate(frame: np.ndarray, result, draw_label: bool = True) -> np.ndarray:
    """Draw detection boxes on a frame copy."""
    annotated = frame.copy()
    for i in range(result.num_detections):
        x1, y1, x2, y2 = result.boxes[i].astype(int)
        score = result.scores[i]
        cls_id = result.class_ids[i]
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if draw_label:
            cv2.putText(annotated, f"{cls_name} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


@click.command()
@click.option("--weights", required=True, type=click.Path(exists=True),
              help="Path to model weights (.pt, .onnx, or .engine).")
@click.option("--backend", default="onnx", type=click.Choice(["pytorch", "onnx", "tensorrt"]),
              help="Inference backend.")
@click.option("--source", required=True, type=click.Path(exists=True),
              help="Path to image or video file.")
@click.option("--model", default="yolov8n", help="Model config name.")
@click.option("--imgsz", default=None, type=int, help="Inference image size (overrides config).")
@click.option("--conf", default=0.25, type=float, help="Confidence threshold.")
@click.option("--iou", default=0.45, type=float, help="NMS IoU threshold.")
@click.option("--slice-size", default=640, type=int, help="SAHI tile size in pixels.")
@click.option("--slice-overlap", default=0.25, type=float, help="SAHI overlap ratio (0–0.5).")
@click.option("--merge-iou", default=0.45, type=float, help="IoU threshold for merging slice detections.")
@click.option("--output", default=None, type=click.Path(), help="Output directory.")
def main(weights, backend, source, model, imgsz, conf, iou, slice_size, slice_overlap, merge_iou, output):
    """Run SAHI inference with before/after comparison."""
    config = Config.from_args(model=model)

    # Use CLI imgsz override if provided, otherwise use config default
    inference_size = imgsz if imgsz is not None else config.input_size

    # Build base backend
    base_backend = _build_backend(backend, str(weights), inference_size, conf, iou)

    # Build SAHI wrapper
    sahi_backend = SahiBackend(base_backend, slice_size=slice_size,
                               slice_overlap=slice_overlap, merge_iou=merge_iou)

    # Warmup
    click.echo(f"[→] Warming up {sahi_backend.name}...")
    sahi_backend.warmup(n=5)

    source_path = Path(source)
    is_video = source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    out_dir = Path(output) if output else config.output_dir / "sahi"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_video:
        # Single image — show before/after comparison
        click.echo(f"[→] Running SAHI comparison on {source_path.name}...")
        frame = cv2.imread(str(source_path))

        # Without SAHI (use base backend directly)
        t0 = cv2.getTickCount()
        result_base = base_backend.predict(frame)
        t1 = cv2.getTickCount()
        base_ms = (t1 - t0) / cv2.getTickFrequency() * 1000

        # With SAHI
        result_sahi = sahi_backend.predict(frame)

        # Per-class breakdown
        click.echo(f"\n{'='*50}")
        click.echo(f"  SAHI Comparison: {source_path.name}")
        click.echo(f"  Image size: {frame.shape[1]}x{frame.shape[0]}")
        click.echo(f"  SAHI config: slice={slice_size}, overlap={slice_overlap}")
        click.echo(f"{'='*50}")
        click.echo(f"  {'Metric':<20} {'Without SAHI':>12} {'With SAHI':>12} {'Δ':>8}")
        click.echo(f"  {'-'*52}")
        click.echo(f"  {'Detections':<20} {result_base.num_detections:>12} {result_sahi.num_detections:>12} {'+' if result_sahi.num_detections > result_base.num_detections else ''}{result_sahi.num_detections - result_base.num_detections:>7}")
        click.echo(f"  {'Latency (ms)':<20} {base_ms:>12.1f} {result_sahi.inference_ms:>12.1f} {result_sahi.inference_ms / max(1, base_ms):>7.1f}x")

        # Per-class counts
        base_counts = {}
        sahi_counts = {}
        for cls_id in range(len(CLASS_NAMES)):
            base_counts[cls_id] = int((result_base.class_ids == cls_id).sum())
            sahi_counts[cls_id] = int((result_sahi.class_ids == cls_id).sum())

        click.echo(f"\n  Per-class breakdown:")
        click.echo(f"  {'Class':<18} {'Without':>8} {'With':>8} {'Δ':>6}")
        click.echo(f"  {'-'*40}")
        for cls_id, name in enumerate(CLASS_NAMES):
            b, s = base_counts[cls_id], sahi_counts[cls_id]
            delta = s - b
            click.echo(f"  {name:<18} {b:>8} {s:>8} {'+' if delta > 0 else ''}{delta:>5}")

        # Save side-by-side comparison
        annotated_base = _annotate(frame, result_base)
        annotated_sahi = _annotate(frame, result_sahi)

        # Add labels
        cv2.putText(annotated_base, f"Without SAHI: {result_base.num_detections} det",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_sahi, f"With SAHI: {result_sahi.num_detections} det",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stack horizontally
        comparison = np.hstack([annotated_base, annotated_sahi])
        comp_path = out_dir / f"{source_path.stem}_sahi_compare.jpg"
        cv2.imwrite(str(comp_path), comparison)
        click.echo(f"\n[✓] Comparison saved to {comp_path}")

    else:
        # Video processing with SAHI
        click.echo(f"[→] Processing video {source_path.name} with SAHI...")
        cap = cv2.VideoCapture(str(source_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = out_dir / f"{source_path.stem}_sahi.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        det_counts = []
        latencies = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = sahi_backend.predict(frame)
            det_counts.append(result.num_detections)
            latencies.append(result.inference_ms)

            annotated = _annotate(frame, result)
            cv2.putText(annotated,
                        f"SAHI | {result.num_detections} det | {result.inference_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 50 == 0:
                click.echo(f"    Frame {frame_idx}/{total}: "
                           f"avg {np.mean(det_counts[-50:]):.1f} det, "
                           f"{1000 / np.mean(latencies[-50:]):.1f} FPS")

        cap.release()
        writer.release()

        click.echo(f"\n[✓] SAHI video saved to {out_path}")
        click.echo(f"    {frame_idx} frames, avg {np.mean(det_counts):.1f} det/frame, "
                   f"{1000 / np.mean(latencies):.1f} FPS")


if __name__ == "__main__":
    main()