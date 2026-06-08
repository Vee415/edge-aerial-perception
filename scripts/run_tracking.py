"""CLI: Run detection + ByteTrack tracking on a video.

Usage:
    python scripts/run_tracking.py --weights best.pt --source video.mp4
    python scripts/run_tracking.py --weights best.engine --backend tensorrt --source video.mp4
"""

import click

from src.drone_perception.config import Config
from src.drone_perception.track.tracker import DroneTracker


@click.command()
@click.option("--weights", required=True, type=click.Path(exists=True),
              help="Path to model weights (.pt or .engine).")
@click.option("--source", required=True, type=click.Path(exists=True),
              help="Path to input video file.")
@click.option("--model", default="yolov8n", help="Model config name.")
@click.option("--dataset", default="visdrone", help="Dataset config name.")
@click.option("--backend", default="pytorch", type=click.Choice(["pytorch", "onnx", "tensorrt"]),
              help="Inference backend for tracking.")
@click.option("--mode", default="native", type=click.Choice(["native", "standalone"]),
              help="Tracking mode: native (model.track) or standalone (custom backend + BYTETracker).")
@click.option("--sahi/--no-sahi", default=False, help="Enable SAHI slicing for small objects.")
@click.option("--slice-size", default=640, type=int, help="SAHI tile size in pixels.")
@click.option("--slice-overlap", default=0.25, type=float, help="SAHI overlap ratio (0–0.5).")
def main(weights, source, model, dataset, backend, mode, sahi, slice_size, slice_overlap):
    """Run detection + ByteTrack tracking on a video."""
    config = Config.from_args(model=model, dataset=dataset)
    tracker = DroneTracker(config)

    if mode == "native":
        click.echo(f"[→] Running native tracking (model.track) on {source}...")
        results = tracker.track_native(weights=weights, video_path=source, save=True)
        click.echo(f"[✓] Tracked {len(results)} frames")
    else:
        click.echo(f"[→] Running standalone tracking ({backend} + BYTETracker) on {source}...")

        # Initialize inference backend
        if backend == "pytorch":
            from src.drone_perception.inference.pytorch_backend import PyTorchBackend
            infer = PyTorchBackend(weights=str(weights), imgsz=config.input_size)
        elif backend == "onnx":
            from src.drone_perception.inference.onnx_backend import ONNXBackend
            infer = ONNXBackend(onnx_path=str(weights), imgsz=config.input_size)
        elif backend == "tensorrt":
            from src.drone_perception.inference.tensorrt_backend import TensorRTBackend
            infer = TensorRTBackend(engine_path=str(weights), imgsz=config.input_size)

        # Wrap with SAHI if enabled
        if sahi:
            from src.drone_perception.inference.sahi_backend import SahiBackend
            infer = SahiBackend(infer, slice_size=slice_size, slice_overlap=slice_overlap)

        results = tracker.track_video(video_path=source, backend=infer, save=True)
        click.echo(f"[✓] Tracked {len(results)} frames")


if __name__ == "__main__":
    main()