"""Memory monitoring for benchmarking.

On Jetson (unified memory), reads /proc/meminfo for total system RAM
including GPU allocations. On desktop, uses psutil for process RSS.

Usage:
    monitor = MemoryMonitor()
    monitor.start()
    # ... run inference ...
    metrics = monitor.stop()  # MemoryMetrics
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from .results import MemoryMetrics


def is_jetson() -> bool:
    """Check if running on a Jetson device."""
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            return True
        # Check for tegrastats
        import subprocess
        result = subprocess.run(
            ["which", "tegrastats"],
            capture_output=True, text=True, timeout=2,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
        return False


def read_proc_meminfo() -> dict:
    """Read /proc/meminfo for system-level memory stats.

    Returns dict with total_mb, used_mb, available_mb.
    On Jetson with unified memory, this includes GPU allocations.
    """
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(':')
                value_kb = int(parts[1])
                info[key] = value_kb
    except (OSError, IOError):
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0}

    total_kb = info.get("MemTotal", 0)
    available_kb = info.get("MemAvailable", info.get("MemFree", 0))
    buffers_kb = info.get("Buffers", 0)
    cached_kb = info.get("Cached", 0)
    # Used = total - available (available already accounts for buffers/cache)
    used_kb = total_kb - available_kb

    return {
        "total_mb": total_kb / 1024,
        "used_mb": used_kb / 1024,
        "available_mb": available_kb / 1024,
    }


@dataclass
class _MemorySample:
    """A single memory reading."""
    timestamp_s: float
    rss_mb: float           # Process RSS (psutil)
    system_used_mb: float    # System-wide used (from /proc/meminfo, includes GPU on Jetson)
    system_total_mb: float  # System total


class MemoryMonitor:
    """Background memory monitor using a sampling thread.

    Periodically samples process RSS and (on Jetson) system-wide memory.
    Sampling happens in a background thread to avoid adding overhead
    to the inference loop.

    Usage:
        monitor = MemoryMonitor()
        monitor.start()
        # ... run inference ...
        metrics = monitor.stop()
    """

    def __init__(self, interval_s: float = 0.05, on_jetson: bool | None = None):
        """Initialize memory monitor.

        Args:
            interval_s: Sampling interval in seconds (default 50ms).
            on_jetson: Force Jetson mode. None = auto-detect.
        """
        self.interval_s = interval_s
        self._on_jetson = on_jetson if on_jetson is not None else is_jetson()
        self._process = psutil.Process()
        self._samples: list[_MemorySample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0

    def start(self):
        """Start background memory sampling."""
        self._samples.clear()
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryMetrics:
        """Stop sampling and return memory statistics.

        Returns:
            MemoryMetrics with avg/peak RSS and (on Jetson) system memory.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if not self._samples:
            return MemoryMetrics(avg_rss_mb=0, peak_rss_mb=0)

        rss_values = [s.rss_mb for s in self._samples]
        avg_rss = sum(rss_values) / len(rss_values)
        peak_rss = max(rss_values)

        if self._on_jetson and any(s.system_used_mb > 0 for s in self._samples):
            sys_used = [s.system_used_mb for s in self._samples if s.system_used_mb > 0]
            sys_total = self._samples[0].system_total_mb if self._samples else 0
            return MemoryMetrics(
                avg_rss_mb=avg_rss,
                peak_rss_mb=peak_rss,
                avg_system_used_mb=sum(sys_used) / len(sys_used),
                peak_system_used_mb=max(sys_used),
                system_total_mb=sys_total,
            )

        return MemoryMetrics(avg_rss_mb=avg_rss, peak_rss_mb=peak_rss)

    def snapshot(self) -> _MemorySample:
        """Take a single memory reading (not for background use)."""
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        system_used_mb = 0.0
        system_total_mb = 0.0

        if self._on_jetson:
            meminfo = read_proc_meminfo()
            system_used_mb = meminfo.get("used_mb", 0.0)
            system_total_mb = meminfo.get("total_mb", 0.0)

        return _MemorySample(
            timestamp_s=time.perf_counter() - self._start_time,
            rss_mb=rss_mb,
            system_used_mb=system_used_mb,
            system_total_mb=system_total_mb,
        )

    def _sample_loop(self):
        """Background sampling loop."""
        while not self._stop_event.is_set():
            try:
                sample = self.snapshot()
                self._samples.append(sample)
            except (psutil.NoSuchProcess, OSError):
                break
            self._stop_event.wait(self.interval_s)