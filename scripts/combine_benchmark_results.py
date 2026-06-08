"""Combine individual benchmark JSON results into unified reports.

Loads laptop and Jetson result files separately, reconstructs BenchmarkResult
objects, and generates clean markdown reports for each platform.

Laptop files: {scenario}_results.json (e.g. pytorch_960_results.json)
Jetson files: jetson_{scenario}_results.json (e.g. jetson_trt960_maxn_results.json)
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drone_perception.benchmark.results import BenchmarkResult
from src.drone_perception.benchmark.report import generate_json, generate_markdown


# Files to skip (aggregated outputs, not individual results)
SKIP_NAMES = {
    "benchmark_suite_results.json",
    "laptop_benchmark_report.json",
    "jetson_benchmark_report.json",
}


def load_results(files: list[Path]) -> list[BenchmarkResult]:
    """Load BenchmarkResult objects from a list of JSON files."""
    results = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        try:
            r = BenchmarkResult.from_dict(d)
            results.append(r)
            pwr = f", {r.power.avg_total_w:.2f}W ({r.power.power_mode})" if r.power else ""
            print(f"  Loaded: {f.name} (mAP@50={r.accuracy.map50:.4f}, FPS={r.latency.fps:.1f}{pwr})")
        except Exception as e:
            print(f"  [!] Error loading {f.name}: {e}")
    return results


def sort_results(results: list[BenchmarkResult], order: list[str]) -> list[BenchmarkResult]:
    """Sort results by scenario order."""
    def sort_key(r):
        try:
            return order.index(r.scenario)
        except ValueError:
            return 99
    return sorted(results, key=sort_key)


def main():
    base_dir = Path("outputs/benchmark_suite")

    # ─── Laptop Report ──────────────────────────────────────
    print("=== Laptop Benchmarks ===")
    laptop_order = [
        "pytorch_960", "onnx_960",
        "sahi_640_o0.25", "sahi_640_o0.2", "sahi_480_o0.25", "sahi_320_o0.25",
    ]
    # Laptop files: no "jetson_" prefix (check base dir + subdirs)
    laptop_files = []
    for f in sorted(base_dir.glob("*_results.json")):
        if not f.name.startswith("jetson_") and f.name not in SKIP_NAMES:
            laptop_files.append(f)
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            for f in sorted(subdir.glob("*_results.json")):
                if not f.name.startswith("jetson_") and f.name not in SKIP_NAMES:
                    laptop_files.append(f)
    laptop_results = load_results(laptop_files)
    laptop_results = sort_results(laptop_results, laptop_order)

    if laptop_results:
        output_path = base_dir / "laptop_benchmark_report"
        generate_json(laptop_results, output_path.with_suffix(".json"))
        generate_markdown(laptop_results, output_path.with_suffix(".md"))
        print(f"\nLaptop: {len(laptop_results)} results → {output_path.with_suffix('.md')}")
    else:
        print("  No laptop results found.")

    # ─── Jetson Report ──────────────────────────────────────
    print("\n=== Jetson Benchmarks ===")
    jetson_order = [
        "trt960_maxn", "trt960_15w",
        "trt960_int8_maxn", "trt960_int8_15w",
        "trt640_maxn",
        "sahi_640_o0.25", "sahi_480_o0.25",
    ]
    # Jetson files: "jetson_" prefix
    jetson_files = [
        f for f in sorted(base_dir.glob("jetson_*_results.json"))
        if f.name not in SKIP_NAMES
    ]
    jetson_results = load_results(jetson_files)
    jetson_results = sort_results(jetson_results, jetson_order)

    if jetson_results:
        output_path = base_dir / "jetson_benchmark_report"
        generate_json(jetson_results, output_path.with_suffix(".json"))
        generate_markdown(jetson_results, output_path.with_suffix(".md"))
        print(f"\nJetson: {len(jetson_results)} results → {output_path.with_suffix('.md')}")
    else:
        print("  No Jetson results found.")


if __name__ == "__main__":
    main()