"""CLI: Run inference on an image, video, or live stream with a specified backend.

Usage:
    python scripts/run_inference.py --weights best.pt --backend pytorch --source test.jpg
    python scripts/run_inference.py --weights best.onnx --backend onnx --source video.mp4
    python scripts/run_inference.py --weights best.engine --backend tensorrt --source video.mp4
    python scripts/run_inference.py --weights best.engine --backend tensorrt --source 0          # webcam
    python scripts/run_inference.py --weights best.engine --backend tensorrt --source rtsp://... # IP camera
"""

import click
import cv2
import numpy as np
from pathlib import Path

from src.drone_perception.config import Config

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def annotate_frame(frame, result):
    """Draw bounding boxes + labels on a frame."""
    for i in range(result.num_detections):
        x1, y1, x2, y2 = result.boxes[i].astype(int)
        score = result.scores[i]
        cls_id = result.class_ids[i]
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


@click.command()
@click.option("--weights", required=True, type=click.Path(exists=True),
              help="Path to model weights (.pt, .onnx, or .engine).")
@click.option("--backend", default="pytorch", type=click.Choice(["pytorch", "onnx", "tensorrt"]),
              help="Inference backend.")
@click.option("--source", required=True,
              help="Image/video file path, camera index (e.g. 0), or RTSP URL.")
@click.option("--model", default="yolov8n", help="Model config name.")
@click.option("--conf", default=0.25, type=float, help="Confidence threshold.")
@click.option("--iou", default=0.45, type=float, help="NMS IoU threshold.")
@click.option("--sahi/--no-sahi", default=False, help="Enable SAHI slicing for small objects.")
@click.option("--slice-size", default=640, type=int, help="SAHI tile size in pixels.")
@click.option("--slice-overlap", default=0.25, type=float, help="SAHI overlap ratio (0–0.5).")
@click.option("--output", default=None, type=click.Path(), help="Output path (image or video).")
@click.option("--imgsz", default=None, type=int, help="Input image size (default: from model config).")
@click.option("--show/--no-show", default=False, help="Display live window (ESC to quit).")
def main(weights, backend, source, model, conf, iou, sahi, slice_size, slice_overlap, output, imgsz, show):
    """Run inference on an image, video, or live stream."""
    config = Config.from_args(model=model, overrides={
        "inference": {"conf": conf, "iou": iou}
    })
    input_size = imgsz or config.input_size

    # Initialize backend
    weights_path = Path(weights)

    if backend == "pytorch":
        from src.drone_perception.inference.pytorch_backend import PyTorchBackend
        infer = PyTorchBackend(weights=str(weights_path), imgsz=input_size, conf=conf, iou=iou)
    elif backend == "onnx":
        from src.drone_perception.inference.onnx_backend import ONNXBackend
        infer = ONNXBackend(onnx_path=str(weights_path), imgsz=input_size, conf=conf, iou=iou)
    elif backend == "tensorrt":
        from src.drone_perception.inference.tensorrt_backend import TensorRTBackend
        infer = TensorRTBackend(engine_path=str(weights_path), imgsz=input_size, conf=conf, iou=iou)

    # Wrap with SAHI if enabled
    if sahi:
        from src.drone_perception.inference.sahi_backend import SahiBackend
        infer = SahiBackend(infer, slice_size=slice_size, slice_overlap=slice_overlap)

    # Warmup
    click.echo(f"[→] Warming up {infer.name} backend...")
    infer.warmup(n=5)

    # Determine source type
    is_stream = source.isdigit() or source.startswith("rtsp://") or source.startswith("http://")
    source_path = Path(source) if not is_stream else None
    is_video = source_path is not None and source_path.suffix.lower() in VIDEO_EXTS

    # ─── Single Image ────────────────────────────────────
    if source_path is not None and not is_video:
        click.echo(f"[→] Running inference on {source_path.name}...")
        frame = cv2.imread(str(source_path))
        result = infer.predict(frame)
        annotated = annotate_frame(frame, result)

        output_path = output or f"outputs/inference/{source_path.stem}_{backend}.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        click.echo(f"[✓] {result.num_detections} detections in {result.inference_ms:.1f}ms → {output_path}")

    # ─── Video File ──────────────────────────────────────
    elif is_video:
        click.echo(f"[→] Processing video {source_path.name}...")
        cap = cv2.VideoCapture(str(source_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = output or f"outputs/inference/{source_path.stem}_{backend}.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        latencies = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = infer.predict(frame)
            latencies.append(result.inference_ms)
            frame_count += 1

            annotated = annotate_frame(frame, result)
            writer.write(annotated)

            if show:
                cv2.imshow("Detection", annotated)
                if cv2.waitKey(1) == 27:  # ESC
                    click.echo("    [!] Stopped by user (ESC)")
                    break

            if frame_count % 100 == 0:
                avg_fps = 1000.0 / (sum(latencies[-100:]) / 100)
                click.echo(f"    Frame {frame_count}: {result.num_detections} det, {avg_fps:.1f} FPS")

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = 1000.0 / np.mean(latencies)
        click.echo(f"[✓] {frame_count} frames at {avg_fps:.1f} FPS → {output_path}")

    # ─── Live Stream (webcam / RTSP) ─────────────────────
    else:
        cam_index = int(source) if source.isdigit() else source
        click.echo(f"[→] Opening stream: {cam_index}")
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            click.echo(f"[✗] Cannot open stream: {cam_index}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        click.echo(f"    Resolution: {w}x{h}")

        writer = None
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            fps = 25.0  # default for live streams
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
            click.echo(f"    Recording to: {output}")

        latencies = []
        frame_count = 0
        click.echo("    Press ESC or Q in window to stop (or Ctrl+C)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    click.echo("[✗] Stream ended or cannot read frame")
                    break

                result = infer.predict(frame)
                latencies.append(result.inference_ms)
                frame_count += 1

                annotated = annotate_frame(frame, result)

                # Overlay FPS + detection count
                if len(latencies) >= 10:
                    recent_fps = 1000.0 / (sum(latencies[-10:]) / 10)
                    cv2.putText(annotated, f"{recent_fps:.1f} FPS | {result.num_detections} det",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if writer:
                    writer.write(annotated)

                cv2.imshow("Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    break

        except KeyboardInterrupt:
            click.echo("\n    [!] Stopped by Ctrl+C")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if latencies:
            avg_fps = 1000.0 / np.mean(latencies)
            click.echo(f"[✓] {frame_count} frames at {avg_fps:.1f} FPS avg")


if __name__ == "__main__":
    main()