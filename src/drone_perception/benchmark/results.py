"""Unified benchmark result dataclasses.

Every benchmark run produces a single BenchmarkResult containing
all four metric categories: accuracy, latency, memory, and power.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class AccuracyMetrics:
    """Detection accuracy metrics computed from val-set predictions."""
    map50: float
    map50_95: float
    per_class_ap50: dict[str, float]
    per_class_map50_95: dict[str, float]
    num_images: int
    conf_threshold: float
    iou_threshold: float


@dataclass
class LatencyMetrics:
    """Inference latency statistics."""
    fps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    num_frames: int


@dataclass
class MemoryMetrics:
    """Memory usage statistics.

    On Jetson (unified memory), system fields capture total RAM
    including GPU allocations. On desktop, only process RSS is available.
    """
    avg_rss_mb: float
    peak_rss_mb: float
    # Jetson unified memory (None on desktop)
    avg_system_used_mb: Optional[float] = None
    peak_system_used_mb: Optional[float] = None
    system_total_mb: Optional[float] = None


@dataclass
class PowerMetrics:
    """Power consumption metrics (Jetson only).

    Rail names differ by platform:
    - Orin Nano Super (JP6.x): VDD_IN (total), VDD_CPU_GPU_CV (combined), VDD_SOC
      Separate GPU/CPU is NOT available; avg_gpu_w stores VDD_CPU_GPU_CV,
      avg_cpu_w is 0.0.
    - Orin (other): POM_5V_IN (total), POM_5V_GPU, POM_5V_CPU
    - Nano T210: VDD_IN (total), VDD_GPU_SOC, VDD_CPU
    """
    avg_total_w: float           # Total module power average
    peak_total_w: float          # Total module power peak
    avg_gpu_w: float             # GPU rail avg (or VDD_CPU_GPU_CV on Orin Nano Super)
    peak_gpu_w: float            # GPU rail peak
    avg_cpu_w: float             # CPU rail avg (0.0 if only combined rail available)
    peak_cpu_w: float            # CPU rail peak
    power_mode: str              # e.g. "MAXN_SUPER", "15W", "25W", "7W"
    sample_count: int
    has_separate_gpu_cpu: bool = True  # False on Orin Nano Super


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single backend configuration.

    Contains latency, accuracy, memory, and (on Jetson) power metrics
    from a single pass over the val set.
    """
    # Scenario identification
    scenario: str
    backend_name: str
    precision: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    sahi_config: Optional[dict] = None  # {"slice_size": 640, "slice_overlap": 0.25}

    # Metric categories
    latency: LatencyMetrics = field(default=None)  # type: ignore[assignment]
    accuracy: AccuracyMetrics = field(default=None)  # type: ignore[assignment]
    memory: MemoryMetrics = field(default=None)  # type: ignore[assignment]
    power: Optional[PowerMetrics] = None

    # System info
    platform: str = ""
    gpu: str = ""
    python_version: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to flat dict suitable for JSON/CSV."""
        d = {
            "scenario": self.scenario,
            "backend": self.backend_name,
            "precision": self.precision,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "sahi_config": self.sahi_config or "None",
        }
        if self.latency:
            d.update({
                "fps": round(self.latency.fps, 1),
                "avg_latency_ms": round(self.latency.avg_latency_ms, 2),
                "p50_latency_ms": round(self.latency.p50_latency_ms, 2),
                "p95_latency_ms": round(self.latency.p95_latency_ms, 2),
                "min_latency_ms": round(self.latency.min_latency_ms, 2),
                "max_latency_ms": round(self.latency.max_latency_ms, 2),
                "num_frames": self.latency.num_frames,
            })
        if self.accuracy:
            d.update({
                "mAP50": round(self.accuracy.map50, 4),
                "mAP50_95": round(self.accuracy.map50_95, 4),
            })
            # Flatten per-class AP50
            for cls_name, ap in self.accuracy.per_class_ap50.items():
                d[f"ap50_{cls_name}"] = round(ap, 4)
        if self.memory:
            d.update({
                "avg_rss_mb": round(self.memory.avg_rss_mb, 1),
                "peak_rss_mb": round(self.memory.peak_rss_mb, 1),
            })
            if self.memory.avg_system_used_mb is not None:
                d["avg_system_used_mb"] = round(self.memory.avg_system_used_mb, 1)
                d["peak_system_used_mb"] = round(self.memory.peak_system_used_mb, 1)
                d["system_total_mb"] = round(self.memory.system_total_mb, 1)
        if self.power:
            d.update({
                "avg_total_power_w": round(self.power.avg_total_w, 2),
                "peak_total_power_w": round(self.power.peak_total_w, 2),
                "avg_gpu_power_w": round(self.power.avg_gpu_w, 2),
                "peak_gpu_power_w": round(self.power.peak_gpu_w, 2),
                "avg_cpu_power_w": round(self.power.avg_cpu_w, 2),
                "peak_cpu_power_w": round(self.power.peak_cpu_w, 2),
                "power_mode": self.power.power_mode,
                "has_separate_gpu_cpu": self.power.has_separate_gpu_cpu,
            })
        d.update({
            "platform": self.platform,
            "gpu": self.gpu,
            "python_version": self.python_version,
            "timestamp": self.timestamp,
        })
        return d

    def to_json(self, path: Path) -> None:
        """Save as JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        """Deserialize from flat dict."""
        latency = LatencyMetrics(
            fps=d.get("fps", 0),
            avg_latency_ms=d.get("avg_latency_ms", 0),
            p50_latency_ms=d.get("p50_latency_ms", 0),
            p95_latency_ms=d.get("p95_latency_ms", 0),
            min_latency_ms=d.get("min_latency_ms", 0),
            max_latency_ms=d.get("max_latency_ms", 0),
            num_frames=d.get("num_frames", 0),
        )
        accuracy = None
        if "mAP50" in d:
            per_class_ap50 = {}
            per_class_map50_95 = {}
            for key, val in d.items():
                if key.startswith("ap50_"):
                    per_class_ap50[key[5:]] = val
                if key.startswith("map50_95_"):
                    per_class_map50_95[key[9:]] = val
            accuracy = AccuracyMetrics(
                map50=d["mAP50"],
                map50_95=d["mAP50_95"],
                per_class_ap50=per_class_ap50,
                per_class_map50_95=per_class_map50_95,
                num_images=d.get("num_images", 0),
                conf_threshold=d.get("conf_threshold", 0.001),
                iou_threshold=d.get("iou_threshold", 0.65),
            )
        memory = None
        if "avg_rss_mb" in d:
            memory = MemoryMetrics(
                avg_rss_mb=d["avg_rss_mb"],
                peak_rss_mb=d["peak_rss_mb"],
                avg_system_used_mb=d.get("avg_system_used_mb"),
                peak_system_used_mb=d.get("peak_system_used_mb"),
                system_total_mb=d.get("system_total_mb"),
            )
        power = None
        if "avg_total_power_w" in d:
            power = PowerMetrics(
                avg_total_w=d["avg_total_power_w"],
                peak_total_w=d["peak_total_power_w"],
                avg_gpu_w=d["avg_gpu_power_w"],
                peak_gpu_w=d["peak_gpu_power_w"],
                avg_cpu_w=d["avg_cpu_power_w"],
                peak_cpu_w=d["peak_cpu_power_w"],
                power_mode=d.get("power_mode", "Unknown"),
                sample_count=d.get("power_sample_count", 0),
                has_separate_gpu_cpu=d.get("has_separate_gpu_cpu", True),
            )
        sahi_config = d.get("sahi_config")
        if isinstance(sahi_config, str) and sahi_config == "None":
            sahi_config = None
        return cls(
            scenario=d["scenario"],
            backend_name=d["backend"],
            precision=d["precision"],
            imgsz=d["imgsz"],
            conf_threshold=d["conf_threshold"],
            iou_threshold=d["iou_threshold"],
            sahi_config=sahi_config,
            latency=latency,
            accuracy=accuracy,
            memory=memory,
            power=power,
            platform=d.get("platform", ""),
            gpu=d.get("gpu", ""),
            python_version=d.get("python_version", ""),
            timestamp=d.get("timestamp", ""),
        )