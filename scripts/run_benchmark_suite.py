"""Unified benchmark suite CLI: latency + mAP + memory + power.

Runs the full benchmark suite over the VisDrone val set, collecting
all four metric categories in a single pass per backend.

Usage:
    # Full suite on laptop (PyTorch + ONNX)
    python scripts/run_benchmark_suite.py \\
        --weights-pt runs/detect/outputs/train/yolov8n_960/weights/best.pt \\
        --weights-onnx runs/detect/outputs/train/yolov8n_960/weights/best.onnx \\
        --data-dir data/visdrone_yolo --backends pytorch,onnx --imgsz 960

    # Full suite on Jetson (TRT only)
    python scripts/run_benchmark_suite.py \\
        --weights-engine runs/detect/outputs/train/yolov8n_960/weights/best_fp16.engine \\
        --data-dir data/visdrone_yolo --backends tensorrt --imgsz 960 --jetson

    # SAHI sweep
    python scripts/run_benchmark_suite.py \\
        --weights-onnx best.onnx --data-dir data/visdrone_yolo --sahi-sweep

    # Named scenario
    python scripts/run_benchmark_suite.py \\
        --scenario onnx_fp32_960 --weights-onnx best.onnx --data-dir data/visdrone_yolo
"""

import click
from pathlib import Path

from src.drone_perception.benchmark.suite import BenchmarkSuite
from src.drone_perception.benchmark.scenarios import (
    SCENARIO_MAP, LAPTOP_SCENARIOS, JETSON_SCENARIOS, BenchmarkScenario,
)
from src.drone_perception.benchmark.report import generate_json, generate_markdown
from src.drone_perception.benchmark.results import BenchmarkResult
from src.drone_perception.inference.onnx_backend import ONNXBackend
from src.drone_perception.inference.sahi_backend import SahiBackend


@click.command()
@click.option("--weights-pt", type=click.Path(exists=True), help="Path to PyTorch .pt weights")
@click.option("--weights-onnx", type=click.Path(exists=True), help="Path to ONNX .onnx model")
@click.option("--weights-engine", type=click.Path(exists=True), help="Path to TensorRT .engine model")
@click.option("--data-dir", required=True, type=click.Path(exists=True), help="YOLO dataset root directory")
@click.option("--backends", default="onnx", help="Comma-separated backends: pytorch,onnx,tensorrt")
@click.option("--imgsz", default=960, type=int, help="Model input size")
@click.option("--conf-map", default=0.001, type=float, help="Confidence threshold for mAP evaluation")
@click.option("--iou-map", default=0.65, type=float, help="IoU threshold for mAP evaluation")
@click.option("--max-images", default=0, type=int, help="Max val images (0=all)")
@click.option("--scenario", default=None, help="Named scenario from scenarios.py")
@click.option("--sahi-sweep", is_flag=True, help="Run SAHI config sweep")
@click.option("--sahi-slice-size", default=640, type=int, help="SAHI slice size")
@click.option("--sahi-overlap", default=0.25, type=float, help="SAHI overlap ratio")
@click.option("--jetson", is_flag=True, help="Enable Jetson power measurement")
@click.option("--power-mode", default=None, type=click.Choice(["maxn_super", "25w", "15w", "7w"]),
              help="Set Jetson power mode before benchmarking")
@click.option("--output-dir", default="outputs/benchmark_suite", help="Output directory")
@click.option("--no-power", is_flag=True, help="Disable power measurement even on Jetson")
def main(
    weights_pt, weights_onnx, weights_engine, data_dir,
    backends, imgsz, conf_map, iou_map, max_images,
    scenario, sahi_sweep, sahi_slice_size, sahi_overlap,
    jetson, power_mode, output_dir, no_power,
):
    """Run unified benchmark suite: latency + mAP + memory + power."""
    from src.drone_perception.benchmark.power import set_jetson_power_mode, is_jetson

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Jetson setup
    on_jetson = jetson or is_jetson()
    measure_power = on_jetson and not no_power

    if power_mode and on_jetson:
        click.echo(f"[→] Setting Jetson power mode: {power_mode}")
        success = set_jetson_power_mode(power_mode)
        if success:
            click.echo("[✓] Power mode set successfully")
        else:
            click.echo("[!] Failed to set power mode (need sudo?)")

    # Parse backends
    backend_list = [b.strip() for b in backends.split(",")]

    # Determine scenarios
    scenarios = []
    if scenario:
        if scenario in SCENARIO_MAP:
            scenarios = [SCENARIO_MAP[scenario]]
        else:
            click.echo(f"Unknown scenario: {scenario}")
            click.echo(f"Available: {', '.join(SCENARIO_MAP.keys())}")
            return
    elif sahi_sweep:
        # Create SAHI sweep scenarios
        sahi_configs = [
            {"slice_size": 640, "slice_overlap": 0.25},
            {"slice_size": 480, "slice_overlap": 0.25},
            {"slice_size": 640, "slice_overlap": 0.20},
            {"slice_size": 320, "slice_overlap": 0.25},
        ]
        for cfg in sahi_configs:
            name = f"sahi_{cfg['slice_size']}_o{cfg['slice_overlap']}"
            scenarios.append(BenchmarkScenario(
                name=name,
                backend="onnx" if "tensorrt" not in backend_list else "tensorrt",
                precision="FP16" if "tensorrt" in backend_list else "FP32",
                imgsz=imgsz,
                conf_map=conf_map,
                iou_map=iou_map,
                conf_deploy=0.25,
                iou_deploy=0.45,
                sahi_config=cfg,
                platform="jetson" if on_jetson else "laptop",
            ))
    else:
        # Generate scenarios from CLI args
        for backend_name in backend_list:
            sahi_cfg = None
            name = f"{backend_name}_{imgsz}"
            precision = "FP32"
            if backend_name == "tensorrt":
                precision = "FP16"
            scenarios.append(BenchmarkScenario(
                name=name,
                backend=backend_name,
                precision=precision,
                imgsz=imgsz,
                conf_map=conf_map,
                iou_map=iou_map,
                conf_deploy=0.25,
                iou_deploy=0.45,
                sahi_config=sahi_cfg,
                platform="jetson" if on_jetson else "laptop",
            ))

    # Create suite
    suite = BenchmarkSuite(data_dir=data_dir, imgsz=imgsz, on_jetson=on_jetson)
    results: list[BenchmarkResult] = []

    click.echo(f"\n{'='*70}")
    click.echo(f"  Benchmark Suite")
    click.echo(f"  Scenarios: {len(scenarios)}")
    click.echo(f"  Data: {data_dir}")
    click.echo(f"  Jetson: {on_jetson}")
    click.echo(f"{'='*70}\n")

    for i, sc in enumerate(scenarios):
        click.echo(f"\n[{i+1}/{len(scenarios)}] Scenario: {sc.name}")
        click.echo(f"  Backend: {sc.backend}, Precision: {sc.precision}, imgsz: {sc.imgsz}")
        if sc.sahi_config:
            click.echo(f"  SAHI: slice_size={sc.sahi_config['slice_size']}, "
                       f"overlap={sc.sahi_config['slice_overlap']}")

        # Create backend
        backend = None
        if sc.backend == "pytorch" and weights_pt:
            from src.drone_perception.inference.pytorch_backend import PyTorchBackend
            backend = PyTorchBackend(
                weights=str(weights_pt),
                imgsz=sc.imgsz, conf=sc.conf_map, iou=sc.iou_map,
            )
        elif sc.backend == "onnx" and weights_onnx:
            backend = ONNXBackend(
                onnx_path=str(weights_onnx),
                imgsz=sc.imgsz, conf=sc.conf_map, iou=sc.iou_map,
            )
        elif sc.backend == "tensorrt" and weights_engine:
            try:
                from src.drone_perception.inference.tensorrt_backend import TensorRTBackend
                backend = TensorRTBackend(
                    engine_path=str(weights_engine),
                    imgsz=sc.imgsz, conf=sc.conf_map, iou=sc.iou_map,
                )
            except ImportError:
                click.echo("[!] TensorRT not available — skipping")
                continue
        else:
            click.echo(f"[!] No weights provided for {sc.backend} — skipping")
            continue

        # Wrap with SAHI if needed
        if sc.sahi_config:
            backend = SahiBackend(
                backend,
                slice_size=sc.sahi_config["slice_size"],
                slice_overlap=sc.sahi_config["slice_overlap"],
                merge_iou=sc.iou_map,
            )

        # Run benchmark
        result = suite.run(
            backend=backend,
            scenario=sc.name,
            max_images=max_images,
            conf_threshold=sc.conf_map,
            iou_threshold=sc.iou_map,
            measure_power=measure_power,
            output_dir=output_dir,
        )
        results.append(result)

        # Clean up backend
        del backend

    if not results:
        click.echo("[!] No benchmarks were run")
        return

    # Generate reports
    generate_json(results, output_dir / "benchmark_suite_results.json")
    generate_markdown(results, output_dir / "benchmark_suite_report.md")

    click.echo(f"\n{'='*70}")
    click.echo(f"  Benchmark Suite Complete: {len(results)} scenarios")
    click.echo(f"  Results: {output_dir / 'benchmark_suite_results.json'}")
    click.echo(f"  Report:  {output_dir / 'benchmark_suite_report.md'}")
    click.echo(f"{'='*70}\n")


if __name__ == "__main__":
    main()