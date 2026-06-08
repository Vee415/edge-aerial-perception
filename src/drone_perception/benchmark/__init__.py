"""Benchmark module — latency, accuracy, memory, and power measurement."""

from .runner import BenchmarkRunner, BenchmarkResult as LegacyBenchmarkResult
from .results import (
    AccuracyMetrics,
    BenchmarkResult,
    LatencyMetrics,
    MemoryMetrics,
    PowerMetrics,
)
from .metrics import compute_map, load_yolo_labels, AccuracyResult
from .memory import MemoryMonitor
from .power import JetsonPowerMonitor, is_jetson, set_jetson_power_mode
from .suite import BenchmarkSuite
from .scenarios import (
    BenchmarkScenario,
    LAPTOP_SCENARIOS,
    JETSON_SCENARIOS,
    ALL_SCENARIOS,
    SCENARIO_MAP,
)
from .report import generate_json, generate_markdown

__all__ = [
    # Legacy (video-only benchmarking)
    "BenchmarkRunner",
    "LegacyBenchmarkResult",
    # Unified suite
    "BenchmarkSuite",
    "BenchmarkResult",
    "AccuracyMetrics",
    "LatencyMetrics",
    "MemoryMetrics",
    "PowerMetrics",
    # mAP computation
    "compute_map",
    "load_yolo_labels",
    "AccuracyResult",
    # Monitoring
    "MemoryMonitor",
    "JetsonPowerMonitor",
    "is_jetson",
    "set_jetson_power_mode",
    # Scenarios
    "BenchmarkScenario",
    "LAPTOP_SCENARIOS",
    "JETSON_SCENARIOS",
    "ALL_SCENARIOS",
    "SCENARIO_MAP",
    # Reporting
    "generate_json",
    "generate_markdown",
]