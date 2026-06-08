"""Predefined benchmark scenarios for laptop and Jetson Orin Nano Super.

Each scenario defines the backend, precision, image size, SAHI config,
confidence threshold, and (for Jetson) power mode. These are used by
run_benchmark_suite.py to run standardized benchmarks.

Jetson Orin Nano Super power modes (from /etc/nvpmodel.conf):
  ID=0: 15W  (GPU 612MHz, CPU 1.5GHz)
  ID=1: 25W  (GPU 918MHz, CPU 1.34GHz) — default
  ID=2: MAXN_SUPER (uncapped)
  ID=3: 7W   (4 cores, GPU 408MHz, CPU 960MHz)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkScenario:
    """A single benchmark configuration."""
    name: str
    backend: str              # "pytorch", "onnx", "tensorrt"
    precision: str             # "FP32", "FP16"
    imgsz: int                 # Model input size
    conf_map: float            # Confidence threshold for mAP eval (typically 0.001)
    iou_map: float             # IoU threshold for mAP eval (typically 0.65)
    conf_deploy: float         # Confidence threshold for deployment FPS (typically 0.25)
    iou_deploy: float          # IoU threshold for deployment (typically 0.45)
    sahi_config: Optional[dict] = None  # {"slice_size": 640, "slice_overlap": 0.25}
    jetson_power_mode: Optional[str] = None  # "MAXN_SUPER", "25W", "15W", "7W"
    platform: str = "any"      # "laptop", "jetson", "any"

    @property
    def is_sahi(self) -> bool:
        return self.sahi_config is not None

    @property
    def display_name(self) -> str:
        parts = [self.backend, self.precision, f"{self.imgsz}"]
        if self.sahi_config:
            ss = self.sahi_config.get("slice_size", "?")
            so = self.sahi_config.get("slice_overlap", "?")
            parts.append(f"sahi_{ss}_o{so}")
        if self.jetson_power_mode:
            parts.append(self.jetson_power_mode.replace(" ", "").lower())
        return "_".join(parts)


# ─── Laptop Benchmarks (RTX 4060) ────────────────────────────────────

LAPTOP_SCENARIOS = [
    BenchmarkScenario(
        name="pytorch_fp32_960",
        backend="pytorch",
        precision="FP32",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        platform="laptop",
    ),
    BenchmarkScenario(
        name="onnx_fp32_960",
        backend="onnx",
        precision="FP32",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        platform="laptop",
    ),
    BenchmarkScenario(
        name="onnx_sahi_640_025",
        backend="onnx",
        precision="FP32",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        sahi_config={"slice_size": 640, "slice_overlap": 0.25},
        platform="laptop",
    ),
    BenchmarkScenario(
        name="onnx_sahi_480_025",
        backend="onnx",
        precision="FP32",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        sahi_config={"slice_size": 480, "slice_overlap": 0.25},
        platform="laptop",
    ),
    BenchmarkScenario(
        name="onnx_sahi_640_020",
        backend="onnx",
        precision="FP32",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        sahi_config={"slice_size": 640, "slice_overlap": 0.20},
        platform="laptop",
    ),
]

# ─── Jetson Orin Nano Super Benchmarks ────────────────────────────────
# Power modes: MAXN_SUPER (ID=2, uncapped), 25W (ID=1), 15W (ID=0), 7W (ID=3)

JETSON_SCENARIOS = [
    # MAXN_SUPER mode (full performance, uncapped)
    BenchmarkScenario(
        name="trt_fp16_960_maxn",
        backend="tensorrt",
        precision="FP16",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        jetson_power_mode="MAXN_SUPER",
        platform="jetson",
    ),
    BenchmarkScenario(
        name="trt_fp16_640_maxn",
        backend="tensorrt",
        precision="FP16",
        imgsz=640,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        jetson_power_mode="MAXN_SUPER",
        platform="jetson",
    ),
    # 15W mode (GPU 612MHz, CPU 1.5GHz)
    BenchmarkScenario(
        name="trt_fp16_960_15w",
        backend="tensorrt",
        precision="FP16",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        jetson_power_mode="15W",
        platform="jetson",
    ),
    # 7W mode (4 cores, GPU 408MHz, CPU 960MHz)
    BenchmarkScenario(
        name="trt_fp16_960_7w",
        backend="tensorrt",
        precision="FP16",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        jetson_power_mode="7W",
        platform="jetson",
    ),
    # SAHI on Jetson (MAXN_SUPER)
    BenchmarkScenario(
        name="trt_sahi_640_025_maxn",
        backend="tensorrt",
        precision="FP16",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        sahi_config={"slice_size": 640, "slice_overlap": 0.25},
        jetson_power_mode="MAXN_SUPER",
        platform="jetson",
    ),
    BenchmarkScenario(
        name="trt_sahi_480_025_maxn",
        backend="tensorrt",
        precision="FP16",
        imgsz=960,
        conf_map=0.001,
        iou_map=0.65,
        conf_deploy=0.25,
        iou_deploy=0.45,
        sahi_config={"slice_size": 480, "slice_overlap": 0.25},
        jetson_power_mode="MAXN_SUPER",
        platform="jetson",
    ),
]

ALL_SCENARIOS = LAPTOP_SCENARIOS + JETSON_SCENARIOS

SCENARIO_MAP = {s.name: s for s in ALL_SCENARIOS}