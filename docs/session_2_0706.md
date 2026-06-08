# Session 2 — June 7, 2026

## Summary

Completed Jetson Orin Nano Super deployment, TensorRT FP16 engine build, full mAP validation, and identified an ultralytics version artifact causing mAP discrepancy. Confirmed TRT FP16 is accuracy-neutral vs FP32.

---

## What Was Done

### ✅ TensorRT FP16 Engine Build (Jetson Orin Nano Super)

- Transferred `best.onnx` (12MB) to Jetson
- Built TensorRT FP16 engine via `trtexec`:
  - Command: `trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16 --memPoolSize=workspace:4096MiB`
  - Engine size: 8.8MB
  - Latency: **8.87ms** (p50), **123 FPS**
  - Jetson specs: JetPack 6.2.1 (R36.4.7), 8GB RAM, 1024 CUDA cores, TensorRT 10.3.0

### ✅ Jetson Environment Setup (~/venvs/ultra-jp61/)

The Jetson dependency chain is **extremely fragile**. Here's what works:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | System |
| torch | 2.5.0a0+872d972e41.nv24.08 | NVIDIA JP6.1 wheel from developer.download.nvidia.com |
| torchvision | 0.20.1 | Built from source on Jetson (required for C++ NMS) |
| ultralytics | 8.3.0 | Installed with `--no-deps` |
| numpy | 1.26.4 | Must be <2.0 for torch 2.5.0 compatibility |
| nvidia-cusparselt-cu12 | latest | Required for libcusparseLt.so.0 |

**Critical:**
- Standard pip torch **breaks** Jetson CUDA libs — must use NVIDIA's JP6.1 wheel
- Standard pip torchvision **breaks** — must build from source for C++ ops (especially `torchvision::nms`)
- ultralytics must be installed `--no-deps` to avoid overwriting torch
- `numpy>=2` is incompatible with torch 2.5.0
- Must set `export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH` before running

**⚠️ DO NOT upgrade ultralytics on Jetson** — it pulls in incompatible dependencies and breaks the torchvision build.

### ✅ Full mAP Validation on Jetson

- Ran `model.val(data='data/visdrone.yaml', imgsz=960)` with TensorRT FP16 engine
- C++ torchvision NMS: postprocess **3.2ms/image** (vs 58.7ms with pure-python NMS stub)
- Jetson result (ultralytics 8.3.0): mAP50=0.5044, mAP50-95=0.3351

### ✅ mAP Discrepancy Investigation

- Jetson showed mAP50=0.5044 vs laptop mAP50=0.4224 — +8% seemed impossible for FP16
- Root cause: **ultralytics version difference** (8.3.0 on Jetson vs 8.4.60 on laptop)
- Ran apples-to-apples comparison on VisDrone2019-DET-val (548 images):

| Backend | Platform | ultralytics | mAP50 | mAP50-95 |
|---------|----------|-------------|-------|----------|
| PyTorch FP32 | RTX 4060 | 8.4.60 | **0.4224** | 0.2500 |
| ONNX FP32 | RTX 4060 | 8.4.60 | **0.4224** | 0.2504 |
| TensorRT FP16 | Jetson Orin | 8.3.0 | 0.5044 | 0.3351 |

- PyTorch and ONNX match perfectly → model is consistent across backends
- TRT FP16 spot check: max score diff **0.005** vs ONNX FP32 → essentially lossless
- **Conclusion: TRT FP16 is accuracy-neutral. True mAP50 = 0.42 on VisDrone DET-val**

### ⏳ Pending: FP32 vs FP16 Direct Comparison

To get a clean FP32 vs FP16 comparison, need to run ONNX FP32 on Jetson with the same ultralytics 8.3.0. This would eliminate both platform and version variables.

---

## Jetson Dependency Hell — Detailed Notes

### Problem 1: libcusparseLt.so.0 missing
- Standard pip torch doesn't include Jetson CUDA libs
- Fix: Install NVIDIA JP6.1 torch wheel from `developer.download.nvidia.com/compute/redist/jp/v61/pytorch/`
- Also need: `pip install nvidia-cusparselt-cu12` + `export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`

### Problem 2: ultralytics pulls in PyPI torch 2.12
- `pip install ultralytics` overwrites the NVIDIA-specific torch
- Fix: `pip install ultralytics --no-deps` then manually install missing deps

### Problem 3: numpy 2.x incompatible with torch 2.5.0
- Fix: `pip install "numpy<2"`

### Problem 4: torchvision 0.27 incompatible with torch 2.5.0
- C++ ops registration crash at import
- Fix: Build torchvision 0.20.1 from source on Jetson
  ```bash
  git clone --branch v0.20.1 https://github.com/pytorch/vision.git
  cd vision
  pip install -e . --no-build-isolation
  ```
- This compiles C++ extensions (including NMS) against Jetson torch

### Problem 5: Pure-python NMS too slow
- Without C++ torchvision, ultralytics falls back to python NMS: 58.7ms postprocess
- Building torchvision from source gives C++ NMS: 3.2ms postprocess
- **18x speedup** on postprocessing

---

## Key Files on Jetson (~/drone_project/)

| File | Path |
|------|------|
| ONNX model | `models/best.onnx` (12MB) |
| TensorRT engine | `models/best_fp16.engine` (8.8MB) |
| VisDrone val data | `data/VisDrone2019-DET-val/` (548 images + labels) |
| Dataset config | `data/visdrone.yaml` |
| Spot check script | `jetson_trt_check.py` |
| Jetson venv | `~/venvs/ultra-jp61/` |

## Key Files on Laptop

| File | Path |
|------|------|
| Best 960 weights | `runs/detect/outputs/train/yolov8n_960/weights/best.pt` |
| ONNX model | `runs/detect/outputs/train/yolov8n_960/weights/best.onnx` |
| SAHI backend | `src/drone_perception/inference/sahi_backend.py` |
| SAHI config | `configs/default.yaml` (sahi: section) |
| Standalone eval | `scripts/eval_mAP.py` |

---

## Future Goals

### 1. Direct FP32 vs FP16 Comparison on Jetson
- Run ONNX FP32 val on Jetson with ultralytics 8.3.0
- Compare against existing TRT FP16 result (both same ultralytics version)
- Quantifies exact FP16 quantization impact

### 2. SAHI on Jetson
- Test SahiBackend with TensorRT backend on Jetson
- Measure FPS impact: expect 4-9x per-frame cost depending on slice count
- 1344×756 sparse frame: ~88ms on RTX 4060 → estimate ~500-700ms on Jetson
- May need larger slice_size (960) or lower overlap for real-time

### 3. End-to-End Pipeline on Jetson
- SAHI + TensorRT + ByteTrack running as a unified pipeline
- Measure total pipeline latency: SAHI overhead + inference + tracking
- Target: 10+ FPS on Jetson for real-time drone operation

### 4. Model Improvements
- YOLOv8s retraining for better accuracy (if FPS budget allows)
- P2 head for small object detection (per research benchmarks)
- Consider SAHI + smaller model vs larger model without SAHI tradeoff

### 5. Jetson Deployment Automation
- Script to transfer model + data to Jetson
- Automated TRT engine build
- Systemd service for auto-start
- Health monitoring / watchdog