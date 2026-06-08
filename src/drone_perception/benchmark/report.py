"""Benchmark report generation: JSON + Markdown.

Generates human-readable comparison tables and machine-readable JSON
from a list of BenchmarkResult objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .results import BenchmarkResult


def generate_json(results: List[BenchmarkResult], path: Path) -> None:
    """Save all results as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "benchmark_suite": "drone_perception",
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_markdown(results: List[BenchmarkResult], path: Path) -> None:
    """Generate a markdown comparison report from benchmark results."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Benchmark Report")
    lines.append("")

    if not results:
        lines.append("No results to report.")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return

    # System info from first result
    r0 = results[0]
    lines.append("## System")
    lines.append(f"- Platform: {r0.platform}")
    lines.append(f"- GPU: {r0.gpu}")
    lines.append(f"- Python: {r0.python_version}")
    lines.append(f"- Date: {r0.timestamp}")
    lines.append("")

    # ── Accuracy Table ─────────────────────────────────────────────
    if any(r.accuracy for r in results):
        lines.append("## Detection Accuracy & Latency")
        lines.append("")

        # Header
        headers = ["Scenario", "Backend", "Precision", "imgsz", "mAP@50", "mAP@50:95",
                    "FPS", "p50 (ms)", "p95 (ms)"]
        has_power = any(r.power for r in results)
        if has_power:
            headers.extend(["Avg Pwr (W)", "Peak Pwr (W)", "Pwr Mode"])
        has_system_mem = any(r.memory and r.memory.avg_system_used_mb for r in results)
        if has_system_mem:
            headers.extend(["Avg RAM (MB)", "Peak RAM (MB)"])
        elif any(r.memory for r in results):
            headers.extend(["Avg RSS (MB)", "Peak RSS (MB)"])

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in results:
            vals = [
                r.scenario,
                r.backend_name,
                r.precision,
                str(r.imgsz),
            ]
            if r.accuracy:
                vals.extend([
                    f"{r.accuracy.map50:.4f}",
                    f"{r.accuracy.map50_95:.4f}",
                ])
            else:
                vals.extend(["—", "—"])

            if r.latency:
                vals.extend([
                    f"{r.latency.fps:.1f}",
                    f"{r.latency.p50_latency_ms:.1f}",
                    f"{r.latency.p95_latency_ms:.1f}",
                ])
            else:
                vals.extend(["—", "—", "—"])

            if has_power and r.power:
                vals.extend([
                    f"{r.power.avg_total_w:.2f}",
                    f"{r.power.peak_total_w:.2f}",
                    r.power.power_mode,
                ])
            elif has_power:
                vals.extend(["—", "—", "—"])

            if has_system_mem and r.memory and r.memory.avg_system_used_mb:
                vals.extend([
                    f"{r.memory.avg_system_used_mb:.0f}",
                    f"{r.memory.peak_system_used_mb:.0f}",
                ])
            elif has_system_mem:
                vals.extend(["—", "—"])
            elif not has_system_mem and r.memory:
                vals.extend([
                    f"{r.memory.avg_rss_mb:.0f}",
                    f"{r.memory.peak_rss_mb:.0f}",
                ])

            lines.append("| " + " | ".join(vals) + " |")

        lines.append("")

    # ── Per-Class mAP Table ────────────────────────────────────────
    accuracy_results = [r for r in results if r.accuracy and r.accuracy.per_class_ap50]
    if accuracy_results:
        lines.append("## Per-Class AP@50 Comparison")
        lines.append("")

        # Collect all class names across results
        all_classes = []
        seen = set()
        for r in accuracy_results:
            for cls_name in r.accuracy.per_class_ap50:
                if cls_name not in seen:
                    all_classes.append(cls_name)
                    seen.add(cls_name)

        headers = ["Class"] + [r.scenario for r in accuracy_results]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for cls_name in all_classes:
            vals = [cls_name]
            for r in accuracy_results:
                ap = r.accuracy.per_class_ap50.get(cls_name, 0)
                vals.append(f"{ap:.4f}")
            lines.append("| " + " | ".join(vals) + " |")

        lines.append("")

    # ── Memory Table ────────────────────────────────────────────────
    mem_results = [r for r in results if r.memory]
    if mem_results:
        lines.append("## Memory Usage")
        lines.append("")
        headers = ["Scenario", "Avg RSS (MB)", "Peak RSS (MB)"]
        if any(r.memory.avg_system_used_mb for r in mem_results):
            headers.extend(["Avg Sys Used (MB)", "Peak Sys Used (MB)", "Sys Total (MB)"])

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in mem_results:
            vals = [
                r.scenario,
                f"{r.memory.avg_rss_mb:.0f}",
                f"{r.memory.peak_rss_mb:.0f}",
            ]
            if r.memory.avg_system_used_mb:
                vals.extend([
                    f"{r.memory.avg_system_used_mb:.0f}",
                    f"{r.memory.peak_system_used_mb:.0f}",
                    f"{r.memory.system_total_mb:.0f}",
                ])
            lines.append("| " + " | ".join(vals) + " |")

        lines.append("")

    # ── Power Table (Jetson only) ──────────────────────────────────
    pwr_results = [r for r in results if r.power]
    if pwr_results:
        lines.append("## Power Consumption (Jetson)")
        lines.append("")
        # Check if any results use combined CPU+GPU rail
        has_separate = all(r.power.has_separate_gpu_cpu for r in pwr_results)
        if has_separate:
            lines.append("| Scenario | Avg Total (W) | Peak Total (W) | "
                          "Avg GPU (W) | Peak GPU (W) | Avg CPU (W) | Peak CPU (W) | "
                          "Mode | Samples |")
            lines.append("| " + " | ".join(["---"] * 9) + " |")
        else:
            # Orin Nano Super: combined CPU+GPU rail, no separate GPU/CPU
            lines.append("| Scenario | Avg Total (W) | Peak Total (W) | "
                          "Avg CPU+GPU (W) | Peak CPU+GPU (W) | "
                          "Mode | Samples |")
            lines.append("| " + " | ".join(["---"] * 7) + " |")

        for r in pwr_results:
            vals = [
                r.scenario,
                f"{r.power.avg_total_w:.2f}",
                f"{r.power.peak_total_w:.2f}",
            ]
            if r.power.has_separate_gpu_cpu:
                vals.extend([
                    f"{r.power.avg_gpu_w:.2f}",
                    f"{r.power.peak_gpu_w:.2f}",
                    f"{r.power.avg_cpu_w:.2f}",
                    f"{r.power.peak_cpu_w:.2f}",
                ])
            else:
                # Combined rail: avg_gpu_w stores VDD_CPU_GPU_CV
                vals.extend([
                    f"{r.power.avg_gpu_w:.2f}",
                    f"{r.power.peak_gpu_w:.2f}",
                ])
            vals.extend([
                r.power.power_mode,
                str(r.power.sample_count),
            ])
            lines.append("| " + " | ".join(vals) + " |")

        lines.append("")

    # ── SAHI Comparison ─────────────────────────────────────────────
    sahi_results = [r for r in results if r.sahi_config]
    baseline_results = [r for r in results if not r.sahi_config]
    if sahi_results and baseline_results:
        lines.append("## SAHI Impact")
        lines.append("")
        lines.append("| Config | mAP@50 | mAP@50:95 | FPS | p50 (ms) | Diff mAP@50 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")

        baseline = baseline_results[0]
        base_map50 = baseline.accuracy.map50 if baseline.accuracy else 0
        for r in sahi_results:
            ss = r.sahi_config.get("slice_size", "?")
            so = r.sahi_config.get("slice_overlap", "?")
            m50 = r.accuracy.map50 if r.accuracy else 0
            m5095 = r.accuracy.map50_95 if r.accuracy else 0
            fps = r.latency.fps if r.latency else 0
            p50 = r.latency.p50_latency_ms if r.latency else 0
            diff = (m50 - base_map50) * 100
            delta = f"+{diff:.1f}pp" if diff > 0 else f"{diff:.1f}pp"
            lines.append(f"| {ss}x{so} | {m50:.4f} | {m5095:.4f} | {fps:.1f} | {p50:.1f} | {delta} |")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))