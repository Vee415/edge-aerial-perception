"""Unified benchmark suite: latency + mAP + memory + power in one pass.

Runs inference over the VisDrone val set, collecting per-frame latency
and predictions for mAP computation, plus background memory and power
monitoring (Jetson only). Uses the shared InferenceBackend abstraction
so PyTorch, ONNX, TRT, and SAHI-wrapped backends all work transparently.

Usage:
    from drone_perception.inference.onnx_backend import ONNXBackend
    from drone_perception.benchmark.suite import BenchmarkSuite

    backend = ONNXBackend(onnx_path="best.onnx", imgsz=960, conf=0.001, iou=0.65)
    suite = BenchmarkSuite(data_dir="data/visdrone_yolo", imgsz=960)
    result = suite.run(backend, scenario="onnx_fp32_960")
    print(f"mAP50={result.accuracy.map50:.4f}, FPS={result.latency.fps:.1f}")
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..inference.backend import InferenceBackend
from .metrics import compute_map, load_yolo_labels, VISDRONE_CLASSES
from .memory import MemoryMonitor, is_jetson as is_jetson_memory
from .power import JetsonPowerMonitor, is_jetson as is_jetson_power, get_jetson_power_mode
from .results import (
    AccuracyMetrics,
    BenchmarkResult,
    LatencyMetrics,
    MemoryMetrics,
    PowerMetrics,
)


class BenchmarkSuite:
    """Run full benchmarks: latency + mAP + memory + power.

    Single-pass over the val set: runs inference on each image,
    collects predictions for mAP and per-frame timing, plus
    background memory/power monitoring.

    Args:
        data_dir: Path to YOLO-format dataset root (with images/val and labels/val).
        imgsz: Model input size (e.g., 960).
        n_classes: Number of object classes.
        class_names: List of class name strings.
        warmup: Number of warmup frames before timing starts.
        on_jetson: Force Jetson mode for power/RAM measurement. None=auto-detect.
    """

    def __init__(
        self,
        data_dir: str | Path,
        imgsz: int = 960,
        n_classes: int = 10,
        class_names: list[str] | None = None,
        warmup: int = 10,
        on_jetson: bool | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.imgsz = imgsz
        self.n_classes = n_classes
        self.class_names = class_names or VISDRONE_CLASSES[:n_classes]
        self.warmup = warmup
        self._on_jetson = on_jetson if on_jetson is not None else is_jetson_power()

        # Validate paths — handle both YOLO split layout and flat layout
        # YOLO split: data_dir/images/val, data_dir/labels/val
        # Flat:       data_dir/images, data_dir/labels
        self.img_dir = self.data_dir / "images" / "val"
        self.label_dir = self.data_dir / "labels" / "val"
        if not self.img_dir.exists():
            # Try flat layout (Jetson convention: VisDrone2019-DET-val/images/)
            flat_img = self.data_dir / "images"
            if flat_img.exists():
                self.img_dir = flat_img
                self.label_dir = self.data_dir / "labels"
            else:
                raise FileNotFoundError(
                    f"Val images not found at {self.data_dir / 'images' / 'val'} "
                    f"or {flat_img}"
                )

    def run(
        self,
        backend: InferenceBackend,
        scenario: str = "",
        max_images: int = 0,
        conf_threshold: float | None = None,
        iou_threshold: float = 0.65,
        measure_power: bool | None = None,
        measure_memory: bool = True,
        output_dir: str | Path | None = None,
    ) -> BenchmarkResult:
        """Run benchmark: latency + mAP + memory + power in one pass.

        Args:
            backend: Inference backend to benchmark.
            scenario: Scenario name for the result.
            max_images: Max number of val images (0 = all).
            conf_threshold: Override backend confidence threshold for mAP eval.
            iou_threshold: IoU threshold for mAP computation.
            measure_power: Enable power measurement (Jetson only). None=auto.
            measure_memory: Enable memory measurement.
            output_dir: Directory to save results. None = don't save.

        Returns:
            BenchmarkResult with all collected metrics.
        """
        # Collect val image paths
        img_files = sorted(self.img_dir.glob("*.jpg"))
        if not img_files:
            img_files = sorted(self.img_dir.glob("*.png"))
        if max_images > 0:
            img_files = img_files[:max_images]

        print(f"\n{'='*70}")
        print(f"  Benchmark Suite: {backend.name}")
        print(f"  Scenario: {scenario or 'unnamed'}")
        print(f"  Images: {len(img_files)}, imgsz: {self.imgsz}")
        print(f"  Jetson: {self._on_jetson}")
        print(f"{'='*70}\n")

        # Load all ground truths
        ground_truths = {}
        for img_path in img_files:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            # Read image to get dimensions
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            gt_boxes, gt_classes = load_yolo_labels(str(label_path), w, h)
            ground_truths[img_path.stem] = (gt_boxes, gt_classes)

        # Start memory monitor
        mem_monitor = MemoryMonitor(on_jetson=self._on_jetson) if measure_memory else None
        if mem_monitor:
            mem_monitor.start()

        # Start power monitor (Jetson only)
        pwr_monitor = None
        if measure_power is None:
            measure_power = self._on_jetson
        if measure_power and self._on_jetson:
            pwr_monitor = JetsonPowerMonitor(interval_ms=100)
            pwr_started = pwr_monitor.start()
            if pwr_started:
                print("[→] Power monitoring started (tegrastats)")
            else:
                print("[!] Power monitoring failed to start")

        # Warmup
        print(f"[→] Warming up {backend.name} ({self.warmup} frames)...")
        backend.warmup(n=self.warmup)

        # Inference loop
        predictions = {}
        latencies = []
        total_images = len(img_files)

        print(f"[→] Running inference on {total_images} images...")
        for idx, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            result = backend.predict(frame)

            # Store predictions for mAP
            predictions[img_path.stem] = (result.boxes, result.scores, result.class_ids)
            latencies.append(result.inference_ms)

            if (idx + 1) % 50 == 0 or idx == total_images - 1:
                print(f"  [{idx+1}/{total_images}] {img_path.stem}: "
                      f"{result.num_detections} det, {result.inference_ms:.1f}ms")

        # Stop memory monitor
        memory_metrics: MemoryMetrics | None = None
        if mem_monitor:
            memory_metrics = mem_monitor.stop()
            print(f"[✓] Memory: avg={memory_metrics.avg_rss_mb:.0f}MB, "
                  f"peak={memory_metrics.peak_rss_mb:.0f}MB"
                  + (f", sys_used={memory_metrics.avg_system_used_mb:.0f}MB"
                     if memory_metrics.avg_system_used_mb else ""))

        # Stop power monitor
        power_metrics: PowerMetrics | None = None
        if pwr_monitor:
            power_metrics = pwr_monitor.stop()
            if power_metrics:
                print(f"[✓] Power: avg={power_metrics.avg_total_w:.2f}W, "
                      f"peak={power_metrics.peak_total_w:.2f}W "
                      f"(mode={power_metrics.power_mode})")

        # Compute latency stats
        latencies_arr = np.array(latencies)
        fps = 1000.0 / np.mean(latencies_arr) if np.mean(latencies_arr) > 0 else 0
        latency_metrics = LatencyMetrics(
            fps=fps,
            avg_latency_ms=float(np.mean(latencies_arr)),
            p50_latency_ms=float(np.percentile(latencies_arr, 50)),
            p95_latency_ms=float(np.percentile(latencies_arr, 95)),
            min_latency_ms=float(np.min(latencies_arr)),
            max_latency_ms=float(np.max(latencies_arr)),
            num_frames=len(latencies),
        )
        print(f"\n[✓] Latency: {fps:.1f} FPS, "
              f"p50={latency_metrics.p50_latency_ms:.1f}ms, "
              f"p95={latency_metrics.p95_latency_ms:.1f}ms")

        # Compute mAP
        print(f"\n[→] Computing mAP...")
        acc_result = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            n_classes=self.n_classes,
            class_names=self.class_names,
        )
        accuracy_metrics = AccuracyMetrics(
            map50=acc_result.map50,
            map50_95=acc_result.map50_95,
            per_class_ap50={
                name: float(acc_result.per_class_ap50[i])
                for i, name in enumerate(self.class_names)
            },
            per_class_map50_95={
                name: float(acc_result.per_class_map50_95[i])
                for i, name in enumerate(self.class_names)
            },
            num_images=len(predictions),
            conf_threshold=conf_threshold or 0.001,
            iou_threshold=iou_threshold,
        )
        print(f"[✓] mAP@50: {accuracy_metrics.map50:.4f}, "
              f"mAP@50:95: {accuracy_metrics.map50_95:.4f}")

        # Per-class summary
        print(f"\n  {'Class':<20s} {'AP@50':>8s} {'mAP@50:95':>10s}")
        print(f"  {'-'*40}")
        for name in self.class_names:
            ap50 = accuracy_metrics.per_class_ap50.get(name, 0)
            map50_95 = accuracy_metrics.per_class_map50_95.get(name, 0)
            print(f"  {name:<20s} {ap50:>8.4f} {map50_95:>10.4f}")
        print(f"  {'-'*40}")
        print(f"  {'MEAN':<20s} {accuracy_metrics.map50:>8.4f} "
              f"{accuracy_metrics.map50_95:>10.4f}")

        # Get system info
        gpu_name = self._get_gpu_name()
        power_mode = power_metrics.power_mode if power_metrics else (
            get_jetson_power_mode() if self._on_jetson else "N/A"
        )

        # Assemble result
        result = BenchmarkResult(
            scenario=scenario,
            backend_name=backend.name,
            precision=backend.precision,
            imgsz=self.imgsz,
            conf_threshold=conf_threshold or 0.001,
            iou_threshold=iou_threshold,
            sahi_config=getattr(backend, '_sahi_config', None),
            latency=latency_metrics,
            accuracy=accuracy_metrics,
            memory=memory_metrics,
            power=power_metrics,
            platform=platform.platform(),
            gpu=gpu_name,
            python_version=platform.python_version(),
        )

        # Save if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            result.to_json(output_dir / f"{scenario or 'benchmark'}_results.json")
            print(f"\n[✓] Results saved to {output_dir}")

        return result

    def run_deployment_latency(
        self,
        backend: InferenceBackend,
        video_path: str | Path,
        scenario: str = "",
        iterations: int = 300,
    ) -> LatencyMetrics:
        """Run latency-only benchmark at deployment conf threshold.

        This measures realistic FPS at conf=0.25 over a video,
        not mAP. Useful for reporting deployment-ready latency numbers.

        Args:
            backend: Inference backend.
            video_path: Path to test video.
            scenario: Scenario name.
            iterations: Number of frames to benchmark.

        Returns:
            LatencyMetrics with FPS and percentile latencies.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Warmup
        print(f"[→] Warming up {backend.name}...")
        backend.warmup(n=self.warmup)

        # Measure
        print(f"[→] Benchmarking {backend.name} ({iterations} frames)...")
        latencies = []
        frame_count = 0

        while frame_count < iterations:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = backend.predict(frame)
            latencies.append(result.inference_ms)
            frame_count += 1

        cap.release()

        latencies_arr = np.array(latencies)
        fps = 1000.0 / np.mean(latencies_arr) if np.mean(latencies_arr) > 0 else 0
        return LatencyMetrics(
            fps=fps,
            avg_latency_ms=float(np.mean(latencies_arr)),
            p50_latency_ms=float(np.percentile(latencies_arr, 50)),
            p95_latency_ms=float(np.percentile(latencies_arr, 95)),
            min_latency_ms=float(np.min(latencies_arr)),
            max_latency_ms=float(np.max(latencies_arr)),
            num_frames=len(latencies),
        )

    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return "Unknown"