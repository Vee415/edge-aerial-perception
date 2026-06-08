"""Jetson power measurement via tegrastats subprocess.

Supports Jetson Orin Nano Super and other Jetson platforms.
Uses tegrastats background logging to capture power consumption
during inference benchmarks.

Rail names differ by platform:
- Orin Nano Super (JP6.x): VDD_IN, VDD_CPU_GPU_CV, VDD_SOC
  (combined CPU+GPU rail, no separate GPU/CPU readings)
- Orin (other): POM_5V_IN, POM_5V_GPU, POM_5V_CPU
- Nano (T210): VDD_IN, VDD_GPU_SOC, VDD_CPU

Usage:
    monitor = JetsonPowerMonitor(interval_ms=100)
    monitor.start()
    # ... run inference ...
    metrics = monitor.stop()  # PowerMetrics or None if not on Jetson
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .results import PowerMetrics


@dataclass
class PowerSample:
    """A single power reading."""
    timestamp_s: float
    total_mw: float       # VDD_IN or POM_5V_IN (total module power)
    gpu_mw: Optional[float] = None   # Separate GPU rail (None if combined)
    cpu_mw: Optional[float] = None   # Separate CPU rail (None if combined)
    cpu_gpu_mw: Optional[float] = None  # Combined CPU+GPU rail (Orin Nano Super)
    soc_mw: Optional[float] = None    # SoC rail (Orin Nano Super)
    ram_used_mb: float = 0.0


def is_jetson() -> bool:
    """Check if running on a Jetson device."""
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            return True
        # Check for tegrastats
        result = subprocess.run(
            ["which", "tegrastats"],
            capture_output=True, text=True, timeout=2,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_jetson_power_mode() -> str:
    """Get current Jetson power mode via nvpmodel.

    Returns:
        Power mode string (e.g., 'MAXN_SUPER', '15W', '25W', '7W') or 'N/A'.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            # Output like "NV Power Mode: MAXN_SUPER\n2"
            output = result.stdout.strip()
            # Extract the mode name from lines like "NV Power Mode: MAXN_SUPER"
            for line in output.splitlines():
                if "Power Mode" in line:
                    # "NV Power Mode: MAXN_SUPER" -> "MAXN_SUPER"
                    return line.split(":", 1)[1].strip()
            return output
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "N/A"


def set_jetson_power_mode(mode: str) -> bool:
    """Set Jetson power mode via nvpmodel.

    Args:
        mode: One of 'maxn_super', '25w', '15w', '7w', or nvpmodel mode number.
            For Orin Nano Super: 0=15W, 1=25W, 2=MAXN_SUPER, 3=7W

    Returns:
        True if successful, False otherwise.
    """
    # Orin Nano Super nvpmodel IDs from /etc/nvpmodel.conf
    mode_map = {
        "maxn_super": "2",
        "maxn": "2",
        "25w": "1",
        "15w": "0",
        "7w": "3",
        "10w": "3",   # approximate: 7W is closest to old 10W
        "maxq": "3",  # approximate
    }
    mode_num = mode_map.get(mode.lower(), mode)

    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-m", mode_num],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Also maximize clocks for MAXN_SUPER
            if mode_num == "2":
                subprocess.run(
                    ["sudo", "jetson_clocks"],
                    capture_output=True, text=True, timeout=10,
                )
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def maximize_jetson_clocks() -> bool:
    """Maximize Jetson clock frequencies for benchmarking."""
    try:
        subprocess.run(
            ["sudo", "jetson_clocks"],
            capture_output=True, text=True, timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Regex patterns for tegrastats output
# Orin Nano Super (JP6.x): VDD_IN 3325mW/3325mW VDD_CPU_GPU_CV 560mW/560mW VDD_SOC 1080mW/1080mW
_RE_ORIN_NANO_SUPER = re.compile(
    r'VDD_IN\s+(\d+)mW/(\d+)mW.*?'
    r'VDD_CPU_GPU_CV\s+(\d+)mW/(\d+)mW.*?'
    r'VDD_SOC\s+(\d+)mW/(\d+)mW',
    re.IGNORECASE,
)
# Orin (other): POM_5V_IN 2532mW/2698mW POM_5V_GPU 1234mW/1500mW POM_5V_CPU 567mW/800mW
_RE_ORIN_POWER = re.compile(
    r'POM_5V_IN\s+(\d+)mW/(\d+)mW.*?'
    r'POM_5V_GPU\s+(\d+)mW/(\d+)mW.*?'
    r'POM_5V_CPU\s+(\d+)mW/(\d+)mW',
    re.IGNORECASE,
)
# Nano T210: VDD_IN 2532/2698 VDD_GPU_SOC 1234/1500 VDD_CPU 567/800
_RE_NANO_POWER = re.compile(
    r'VDD_IN\s+(\d+)/(\d+).*?'
    r'VDD_GPU_SOC\s+(\d+)/(\d+).*?'
    r'VDD_CPU\s+(\d+)/(\d+)',
    re.IGNORECASE,
)
# RAM field: RAM 1146/3996MB or RAM 452/7620MB
_RE_RAM = re.compile(r'RAM\s+(\d+)/(\d+)(?:MB|kB|mB)?', re.IGNORECASE)


def parse_tegrastats_line(line: str) -> Optional[PowerSample]:
    """Parse a single tegrastats output line for power and RAM.

    Handles three tegrastats formats:
    - Orin Nano Super: VDD_IN, VDD_CPU_GPU_CV, VDD_SOC
    - Orin (other): POM_5V_IN, POM_5V_GPU, POM_5V_CPU
    - Nano T210: VDD_IN, VDD_GPU_SOC, VDD_CPU

    Args:
        line: A single tegrastats output line.

    Returns:
        PowerSample or None if line cannot be parsed.
    """
    gpu_mw = None
    cpu_mw = None
    cpu_gpu_mw = None
    soc_mw = None
    total_mw = 0.0

    # Try Orin Nano Super format first (most specific)
    match = _RE_ORIN_NANO_SUPER.search(line)
    if match:
        total_mw = float(match.group(1))    # VDD_IN current
        cpu_gpu_mw = float(match.group(3))  # VDD_CPU_GPU_CV current
        soc_mw = float(match.group(5))      # VDD_SOC current
    else:
        # Try Orin format
        match = _RE_ORIN_POWER.search(line)
        if match:
            total_mw = float(match.group(1))   # POM_5V_IN current
            gpu_mw = float(match.group(3))      # POM_5V_GPU current
            cpu_mw = float(match.group(5))       # POM_5V_CPU current
        else:
            # Try Nano T210 format
            match = _RE_NANO_POWER.search(line)
            if match:
                total_mw = float(match.group(1))   # VDD_IN
                gpu_mw = float(match.group(3))      # VDD_GPU_SOC
                cpu_mw = float(match.group(5))       # VDD_CPU
            else:
                return None

    # Parse RAM
    ram_match = _RE_RAM.search(line)
    ram_used_mb = float(ram_match.group(1)) if ram_match else 0.0

    return PowerSample(
        timestamp_s=0.0,  # Will be set relative to start
        total_mw=total_mw,
        gpu_mw=gpu_mw,
        cpu_mw=cpu_mw,
        cpu_gpu_mw=cpu_gpu_mw,
        soc_mw=soc_mw,
        ram_used_mb=ram_used_mb,
    )


class JetsonPowerMonitor:
    """Background power monitor using tegrastats on Jetson devices.

    Starts tegrastats as a subprocess that logs to a file, then parses
    the log when stopped to compute power statistics.

    Usage:
        monitor = JetsonPowerMonitor(interval_ms=100)
        monitor.start()
        # ... run inference ...
        metrics = monitor.stop()  # Returns PowerMetrics or None
    """

    def __init__(self, interval_ms: int = 100, log_dir: str | Path | None = None):
        """Initialize power monitor.

        Args:
            interval_ms: tegrastats sampling interval in milliseconds.
            log_dir: Directory for tegrastats log file. Defaults to /tmp.
        """
        self.interval_ms = interval_ms
        self.log_dir = Path(log_dir) if log_dir else Path("/tmp")
        self.log_path = self.log_dir / "tegrastats_benchmark.log"
        self._process: subprocess.Popen | None = None
        self._start_time: float = 0.0
        self._is_jetson = is_jetson()

    def start(self) -> bool:
        """Start tegrastats background logging.

        Returns:
            True if monitoring started, False if not on Jetson or tegrastats unavailable.
        """
        if not self._is_jetson:
            return False

        try:
            # Stop any existing tegrastats
            subprocess.run(
                ["tegrastats", "--stop"],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Start fresh tegrastats logging
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._process = subprocess.Popen(
                [
                    "tegrastats",
                    "--interval", str(self.interval_ms),
                    "--logfile", str(self.log_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._start_time = time.perf_counter()
            time.sleep(0.5)  # Let tegrastats stabilize
            return True
        except FileNotFoundError:
            self._process = None
            return False

    def stop(self) -> Optional[PowerMetrics]:
        """Stop tegrastats, parse log, and return power metrics.

        On Orin Nano Super, separate GPU/CPU power is not available.
        avg_gpu_w/avg_cpu_w will be set to 0.0 and the combined
        cpu+gpu power is reported via avg_cpu_gpu_w (stored in
        avg_gpu_w as the closest available metric).

        Returns:
            PowerMetrics if on Jetson and data was collected, None otherwise.
        """
        if self._process is None:
            return None

        # Stop tegrastats
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()

        # Also explicitly stop tegrastats daemon
        try:
            subprocess.run(
                ["tegrastats", "--stop"],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Parse the log
        if not self.log_path.exists():
            return None

        samples: list[PowerSample] = []
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = parse_tegrastats_line(line)
                if sample is not None:
                    sample.timestamp_s = 0.0  # Relative timestamps not critical
                    samples.append(sample)

        # Clean up log
        try:
            self.log_path.unlink()
        except OSError:
            pass

        if not samples:
            return None

        # Compute statistics
        total_w = [s.total_mw / 1000.0 for s in samples]

        # Determine if we have separate GPU/CPU or combined
        has_separate = samples[0].gpu_mw is not None
        has_combined = samples[0].cpu_gpu_mw is not None

        if has_separate:
            gpu_w = [s.gpu_mw / 1000.0 for s in samples]  # type: ignore[union-attr]
            cpu_w = [s.cpu_mw / 1000.0 for s in samples]  # type: ignore[union-attr]
            avg_gpu_w = sum(gpu_w) / len(gpu_w)
            peak_gpu_w = max(gpu_w)
            avg_cpu_w = sum(cpu_w) / len(cpu_w)
            peak_cpu_w = max(cpu_w)
        elif has_combined:
            # Orin Nano Super: no separate GPU/CPU, use combined rail
            cpu_gpu_w = [s.cpu_gpu_mw / 1000.0 for s in samples]  # type: ignore[union-attr]
            # Store combined in gpu_w field (represents compute power)
            avg_gpu_w = sum(cpu_gpu_w) / len(cpu_gpu_w)
            peak_gpu_w = max(cpu_gpu_w)
            # CPU alone is unknown; estimate as total - soc - cpu_gpu
            # but that's imprecise, so just report 0 for separate CPU
            avg_cpu_w = 0.0
            peak_cpu_w = 0.0
        else:
            avg_gpu_w = 0.0
            peak_gpu_w = 0.0
            avg_cpu_w = 0.0
            peak_cpu_w = 0.0

        power_mode = get_jetson_power_mode()

        return PowerMetrics(
            avg_total_w=sum(total_w) / len(total_w),
            peak_total_w=max(total_w),
            avg_gpu_w=avg_gpu_w,
            peak_gpu_w=peak_gpu_w,
            avg_cpu_w=avg_cpu_w,
            peak_cpu_w=peak_cpu_w,
            power_mode=power_mode,
            sample_count=len(samples),
            has_separate_gpu_cpu=has_separate,
        )

    def read_sysfs_power(self) -> Optional[dict]:
        """Read power from INA3221 sysfs (Jetson direct read, no tegrastats).

        Returns:
            Dict with VDD_IN, VDD_GPU, VDD_CPU power in mW, or None.
        """
        # INA3221 sysfs paths vary by Jetson model
        # Common base paths
        sysfs_paths = [
            "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/",
            "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/",
        ]

        for base in sysfs_paths:
            try:
                vdd_in = self._read_sysfs(f"{base}in_power0_input")
                vdd_gpu = self._read_sysfs(f"{base}in_power1_input")
                vdd_cpu = self._read_sysfs(f"{base}in_power2_input")

                if vdd_in is not None and vdd_gpu is not None and vdd_cpu is not None:
                    return {
                        "VDD_IN_mW": vdd_in,
                        "VDD_GPU_mW": vdd_gpu,
                        "VDD_CPU_mW": vdd_cpu,
                    }
            except (OSError, IOError):
                continue

        return None

    @staticmethod
    def _read_sysfs(path: str) -> Optional[int]:
        """Read a single integer from a sysfs file."""
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except (OSError, IOError, ValueError):
            return None