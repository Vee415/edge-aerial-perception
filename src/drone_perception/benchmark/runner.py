"""Performance benchmarking across inference backends.

Produces the benchmark table from the project spec:

| Backend    | Precision | FPS | p50 Latency | p95 Latency | Memory | Power Mode | Notes |
|------------|-----------|-----|-------------|-------------|---------|------------|-------|
| PyTorch    | FP32      | TBD | TBD         | TBD         | TBD     | TBD        | ...   |
| ONNX RT    | FP32      | TBD | TBD         | TBD         | TBD     | TBD        | ...   |
| TensorRT   | FP16      | TBD | TBD         | TBD         | TBD     | TBD        | ...   |
| TensorRT   | INT8      | TBD | TBD         | TBD         | TBD     | TBD        | ...   |
"""

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from ..config import Config
from ..inference.backend import InferenceBackend, FrameResult
from ..inference.pytorch_backend import PyTorchBackend
from ..inference.onnx_backend import ONNXBackend


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single backend."""
    backend: str
    precision: str
    fps: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_memory_mb: float
    peak_memory_mb: float
    power_mode: str = "N/A"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "Backend": self.backend,
            "Precision": self.precision,
            "FPS": round(self.fps, 1),
            "p50 Latency (ms)": round(self.p50_latency_ms, 1),
            "p95 Latency (ms)": round(self.p95_latency_ms, 1),
            "Avg Memory (MB)": round(self.avg_memory_mb, 1),
            "Peak Memory (MB)": round(self.peak_memory_mb, 1),
            "Power Mode": self.power_mode,
            "Notes": self.notes,
        }


class BenchmarkRunner:
    """Benchmark inference backends for FPS, latency, and memory usage.

    Usage:
        runner = BenchmarkRunner(config)
        results = runner.benchmark_all(video_path="test.mp4")
        runner.print_table(results)
    """

    def __init__(self, config: Config):
        self.config = config
        self.benchmark_cfg = config.benchmark
        self.output_dir = config.output_dir / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def benchmark_backend(
        self,
        backend: InferenceBackend,
        video_path: str | Path,
    ) -> BenchmarkResult:
        """Run benchmark on a single backend over a video.

        Args:
            backend: Inference backend to benchmark.
            video_path: Path to test video file.

        Returns:
            BenchmarkResult with FPS, latency, memory stats.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        warmup = self.benchmark_cfg.get("warmup", 10)
        iterations = self.benchmark_cfg.get("iterations", 300)

        # Warmup
        print(f"[→] Warming up {backend.name} ({warmup} frames)...")
        backend.warmup(n=warmup)

        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Benchmark
        print(f"[→] Benchmarking {backend.name} ({iterations} frames)...")
        latencies = []
        mem_readings = []
        frame_count = 0

        while frame_count < iterations:
            ret, frame = cap.read()
            if not ret:
                # Loop video if shorter than iterations
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = backend.predict(frame)
            latencies.append(result.inference_ms)
            mem_readings.append(process.memory_info().rss / 1024 / 1024)
            frame_count += 1

        cap.release()

        # Compute stats
        latencies = np.array(latencies)
        fps = 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        avg_memory = np.mean(mem_readings)
        peak_memory = np.max(mem_readings)

        # Power mode (Jetson only)
        power_mode = self._get_power_mode()

        result = BenchmarkResult(
            backend=backend.name,
            precision=backend.precision,
            fps=fps,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            power_mode=power_mode,
            notes=f"{iterations} frames, {video_path.name}",
        )

        print(f"[✓] {backend.name}: {fps:.1f} FPS, "
              f"p50={p50:.1f}ms, p95={p95:.1f}ms, "
              f"mem={avg_memory:.0f}MB avg")

        return result

    def benchmark_all(
        self,
        video_path: str | Path,
        weights_pt: str | Path | None = None,
        weights_onnx: str | Path | None = None,
        weights_engine: str | Path | None = None,
        backends: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark multiple backends and compare.

        Args:
            video_path: Path to test video.
            weights_pt: Path to PyTorch .pt weights.
            weights_onnx: Path to ONNX .onnx model.
            weights_engine: Path to TensorRT .engine model.
            backends: List of backends to benchmark. Default: ["pytorch", "onnx"].

        Returns:
            List of BenchmarkResult for each backend.
        """
        if backends is None:
            backends = ["pytorch", "onnx"]

        imgsz = self.config.input_size
        conf = self.config.inference.get("conf", 0.25)
        iou = self.config.inference.get("iou", 0.45)

        results = []

        if "pytorch" in backends and weights_pt:
            pt_backend = PyTorchBackend(
                weights=str(weights_pt),
                imgsz=imgsz, conf=conf, iou=iou,
            )
            results.append(self.benchmark_backend(pt_backend, video_path))

        if "onnx" in backends and weights_onnx:
            onnx_backend = ONNXBackend(
                onnx_path=str(weights_onnx),
                imgsz=imgsz, conf=conf, iou=iou,
            )
            results.append(self.benchmark_backend(onnx_backend, video_path))

        if "tensorrt" in backends and weights_engine:
            try:
                from ..inference.tensorrt_backend import TensorRTBackend
                trt_backend = TensorRTBackend(
                    engine_path=str(weights_engine),
                    imgsz=imgsz, conf=conf, iou=iou,
                )
                results.append(self.benchmark_backend(trt_backend, video_path))
            except ImportError:
                print("[!] TensorRT not available — skipping TensorRT backend")

        # Save results
        self._save_results(results)

        return results

    def print_table(self, results: list[BenchmarkResult]) -> None:
        """Print benchmark results as a formatted table matching the project spec."""
        df = pd.DataFrame([r.to_dict() for r in results])

        # Print markdown table
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)
        print(df.to_markdown(index=False))
        print("=" * 100 + "\n")

        # Also save as CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"[✓] Results saved to {csv_path}")

    def _get_power_mode(self) -> str:
        """Get Jetson power mode if running on Jetson, else 'N/A'."""
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "N/A"

    def measure_power(self, duration_s: int = 30) -> dict:
        """Measure power consumption on Jetson using tegrastats.

        Args:
            duration_s: Duration to measure in seconds.

        Returns:
            Dict with power stats: avg_w, peak_w, module_name.
        """
        try:
            # Start tegrastats logging
            log_path = self.output_dir / "tegrastats.log"
            proc = subprocess.Popen(
                ["tegrastats", "--start", str(log_path)],
                stdout=subprocess.DEVNULL,
            )
            time.sleep(duration_s)
            proc.terminate()

            # Parse tegrastats log for power readings
            powers = []
            with open(log_path, "r") as f:
                for line in f:
                    if "VDD_GPU_SOC" in line or "milliwatt" in line.lower():
                        # Extract power values
                        import re
                        matches = re.findall(r"(\d+) mW", line)
                        if matches:
                            powers.append(sum(int(m) for m in matches))

            if powers:
                return {
                    "avg_power_w": np.mean(powers) / 1000,
                    "peak_power_w": np.max(powers) / 1000,
                    "duration_s": duration_s,
                }
        except FileNotFoundError:
            print("[!] tegrastats not available — not running on Jetson")

        return {"avg_power_w": "N/A", "peak_power_w": "N/A", "duration_s": duration_s}

    def _save_results(self, results: list[BenchmarkResult]) -> None:
        """Save benchmark results as JSON."""
        data = {
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "gpu": self._get_gpu_name(),
            },
            "results": [r.to_dict() for r in results],
        }
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return "Unknown"