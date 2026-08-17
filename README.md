# Jetson Aerial Perception Pipeline

![C++ TensorRT inference demo](edge_detection_cpp.gif)

Real-time aerial object detection and tracking pipeline deployed on **NVIDIA Jetson Orin Nano Super** (8GB, JetPack 6.2.1, L4T 35.6, TensorRT 10.3). YOLOv8n trained on VisDrone, exported through ONNX to TensorRT FP16, with SAHI tiling for small object recovery, ByteTrack for multi-object tracking, and a **C++ TensorRT inference binary** for production deployment.

## Results at a Glance

| Config | mAP@50 | FPS | Latency (p50) | Power | Platform |
|--------|--------|-----|---------------|-------|----------|
| **C++ TensorRT FP16 960** | 0.406 | **37.7** | 31.1 ms | 8.7W | Jetson Orin Nano Super 15W |
| Python TensorRT FP16 960 | 0.406 | 20.1 | ~50 ms | 8.7W | Jetson Orin Nano Super 15W |
| TensorRT INT8 960 | 0.277 | 60.9 | 16.8 ms | 8.9W | Jetson Orin Nano Super MAXN |
| SAHI 640/0.25 (FP16) | **0.449** | 4.2 | 284.4 ms | 5.5W | Jetson Orin Nano Super MAXN |
| PyTorch FP32 960 | 0.410 | 109.8 | 8.6 ms | -- | RTX 4060 Laptop |

**Key findings:**

- **C++ inference is 88% faster than Python** (37.7 vs 20.1 FPS at 640x360) with identical detections
- **15W power mode is 28% faster than MAXN** for Python inference (28.1 vs 22.0 FPS) because fixed clocks avoid thermal throttling. C++ at 15W reaches 37.7 FPS.
- **SAHI recovers +10.6% mAP** over baseline (0.406 -> 0.449) for small objects
- **INT8 gives 2.8x speed but costs 31.8% mAP** -- not viable for YOLOv8n on small-object datasets

### C++ vs Python on Jetson Orin Nano Super (15W mode, imgsz=960)

| Metric | Python TRT | C++ TRT | Improvement |
|--------|------------|---------|-------------|
| Avg FPS (640x360 video) | 20.1 | 37.7 | **+88%** |
| Avg FPS (1280x720 video) | ~15 | 32.7 | **+118%** |
| Avg latency | ~50 ms | 26.5 ms | **-47%** |
| P50 latency | -- | 31.1 ms | -- |
| Startup time | ~3s | ~0.1s | **30x faster** |
| RAM usage | ~2.5 GB | ~150 MB | **17x less** |
| Binary size | N/A | ~5 MB | -- |

### End-to-End Latency Breakdown (C++, 1280x720 input, 15W mode)

| Stage | Time | Where |
|-------|------|-------|
| CUDA letterbox preprocess | ~2 ms | GPU (resize + normalize + HWC-to-CHW, zero-copy to TRT) |
| TensorRT FP16 inference | ~28 ms | GPU (the fixed cost, same as Python) |
| CPU NMS + decode | ~1 ms | CPU (per-class greedy NMS on 18900 anchors) |
| OpenCV annotation + write | varies | CPU (depends on detection count and codec) |
| **Total per frame** | **~31 ms** | |

Python adds ~8 ms for NumPy preprocess (CPU-to-GPU copy) and ~3 ms for Python NMS overhead.

## Pipeline Architecture

```
                        Input Stream
                     (video / camera / RTSP)
                              |
                              v
                    +-------------------+
                    |  CUDA Preprocess  |  GPU: letterbox resize +
                    |   (C++ kernel)    |  normalize, zero-copy to TRT
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |  TensorRT FP16    |  GPU: YOLOv8n inference
                    |   (1x14x18900)    |  10-class VisDrone output
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |   NMS + Decode    |  CPU: per-class NMS,
                    |   (C++ / Python)  |  xywh to xyxy, letterbox undo
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |  Optional: SAHI   |  Slice into tiles at 640px
                    |  (wrapper only)   |  for +10.6% mAP (4.2 FPS)
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |  Optional: Track  |  ByteTrack multi-object
                    |   (Python only)   |  tracking with ID persistence
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |  Annotate + Output|  Draw boxes, class labels,
                    |                   |  FPS overlay, save video
                    +-------------------+

Training pipeline (separate):
  VisDrone --> YOLOv8n train (960) --> ONNX export --> TensorRT FP16 engine
                                                  --> .engine file (8.8 MB)
```

## Real-World Inference

The model generalizes to unseen aerial/surveillance footage beyond VisDrone:

| Source | Backend | Resolution | FPS | Detections/frame |
|--------|---------|-----------|-----|------------------|
| VisDrone val | C++ TRT FP16 | 640x360 | 37.7 | 8-41 |
| Coimbatore flyover (YouTube) | C++ TRT FP16 | 1280x720 | 32.7 | 8-45 |
| YouTube aerial traffic | Python TRT FP16 | 640x360 | 23 | 8-39 |
| YouTube aerial traffic + SAHI | ONNX (laptop) | -- | 50 | 17-23 |

### Failure Cases

The model struggles in these scenarios (important for deployment decisions):

- **Dense crowds**: Overlapping boxes, NMS merges nearby pedestrians
- **Tiny pedestrians at high altitude**: Below 10px height, mAP drops sharply (pedestrian AP=0.488, bicycle AP=0.184)
- **Motion blur**: Fast drone movement smears objects, reducing confidence
- **Night/low light**: Not trained on nighttime data; detections drop significantly
- **Heavy occlusion**: Parked cars, awnings, and trees partially hide objects
- **High altitude (>120m)**: Objects become sub-5px; only SAHI recovers some (bicycle AP 0.11 -> 0.25 with SAHI)
- **VisDrone-specific classes**: "awning-tricycle" (AP=0.144) and "tricycle" are rare and poorly defined

## INT8 Quantization Analysis

| | FP16 | INT8 PTQ | Delta |
|---|---|---|---|
| mAP@50 | 0.406 | 0.277 | **-31.8%** |
| FPS (MAXN) | 22.0 | 60.9 | +177% |
| FPS (15W) | 28.1 | 29.3 | +4% |
| Engine size | 8.8 MB | 4.9 MB | -44% |

INT8 destroys small-object detection (van -57%, bus -48%, pedestrian -38%). YOLOv8n has only 3M parameters -- too little capacity to absorb quantization error. **FP16 is the right precision for this model + dataset.**

## Project Structure

```
drone_project/
+-- configs/                # YAML configs (default, model, dataset)
+-- src/drone_perception/   # Core Python modules
|   +-- config.py           # 3-layer config merge (default -> model -> dataset -> CLI)
|   +-- data/               # VisDrone download + YOLO annotation conversion
|   +-- train/              # YOLO fine-tuning with OOM handling + P2 head
|   +-- validate/           # mAP evaluation + per-class + size analysis
|   +-- export/             # ONNX + TensorRT (FP16/INT8) export + verification
|   +-- inference/          # Backend-agnostic: PyTorch / ONNX / TensorRT / SAHI
|   +-- track/              # ByteTrack (native + standalone modes)
|   +-- benchmark/          # Latency + mAP + memory + power measurement suite
|   +-- analyze/            # Failure analysis (size, occlusion, blur, density)
+-- src/cpp_inference/       # Production C++ TensorRT binary
|   +-- include/             # Headers (types, trt_engine, cuda_preprocess, video_io)
|   +-- src/                 # Implementation (TRT 10.x API, CUDA letterbox, NMS)
|   +-- CMakeLists.txt       # Build system (Jetson: sm_87, nvinfer, cudart, OpenCV)
|   +-- README.md            # Build instructions + measured benchmarks
|   +-- scripts/             # sync_to_jetson.sh
+-- scripts/                # CLI entry points
+-- outputs/benchmark_suite/  # Benchmark reports + JSON results
+-- docs/                    # Session notes + C++ inference plan
```

## Hardware

| | Training | Inference |
|---|---|---|
| **Device** | RTX 4060 Laptop (8GB VRAM) | Jetson Orin Nano Super (8GB unified) |
| **SoC** | AD106, CUDA 12.4 | Tegra Orin (A78AE + Ampere GPU, sm_87) |
| **JetPack** | N/A | 6.2.1 (L4T 35.6, TensorRT 10.3, CUDA 12.6) |
| **Framework** | PyTorch 2.5.1, CUDA 12.4 | TensorRT 10.3 C++ API |
| **Precision** | FP32 | FP16 (primary), INT8 (evaluated, not viable) |

## Setup

### Laptop (Training + ONNX Benchmarking)

```bash
conda create --name drone_perception --clone gpu_base -y
conda activate drone_perception
pip install -r requirements.txt
```

### Jetson (TensorRT Deployment)

```bash
# On Jetson Orin Nano Super (JetPack 6.2.1)
# Uses pre-built venv at ~/venvs/ultra-jp61
export PYTHONPATH=$HOME/drone_project
export LD_LIBRARY_PATH=$HOME/venvs/ultra-jp61/lib/python3.10/site-packages/nvidia/cusparselt/lib
```

## Usage

### C++ Inference (Production)

```bash
# Build on Jetson
cd ~/drone_project/src/cpp_inference
mkdir build && cd build
CUDACXX=/usr/local/cuda/bin/nvcc cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run on video file
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source ~/drone_project/data/video.mp4 --imgsz 960

# Run with output video
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source video.mp4 --imgsz 960 --conf 0.25 \
                  --output output.mp4

# Live camera
./drone_inference --engine models/best_fp16.engine --source 0 --imgsz 960 --show

# RTSP stream
./drone_inference --engine models/best_fp16.engine \
                  --source rtsp://192.168.1.100:554/stream --imgsz 960
```

### Python Inference (Development)

```bash
# Single image
python scripts/run_inference.py --weights best.onnx --backend onnx --source test.jpg --imgsz 960

# Video file
python scripts/run_inference.py --weights best.onnx --backend onnx --source video.mp4 --imgsz 960

# Live preview (ESC/Q to quit)
python scripts/run_inference.py --weights best.onnx --backend onnx --source video.mp4 --imgsz 960 --show

# Webcam live stream
python scripts/run_inference.py --weights best.onnx --backend onnx --source 0 --imgsz 960 --show

# RTSP IP camera
python scripts/run_inference.py --weights best.onnx --backend onnx --source rtsp://... --imgsz 960 --show

# SAHI for small object recovery (+10.6% mAP)
python scripts/run_inference.py --weights best.onnx --backend onnx --source video.mp4 --imgsz 960 --sahi

# Save live stream to file
python scripts/run_inference.py --weights best.onnx --backend onnx --source 0 --imgsz 960 --show --output out.mp4
```

### Tracking (ByteTrack)

```bash
python scripts/run_tracking.py --weights best.engine --backend tensorrt --source video.mp4
python scripts/run_tracking.py --weights best.engine --backend tensorrt --source video.mp4 --sahi
```

### Full Pipeline (training to deployment)

```bash
# 1. Prepare VisDrone dataset
python scripts/prepare_data.py

# 2. Train YOLOv8n at 960
python scripts/train_model.py --model yolov8n --epochs 100 --imgsz 960

# 3. Validate
python scripts/validate_model.py --weights runs/detect/train/weights/best.pt

# 4. Export to ONNX + TensorRT
python scripts/export_model.py --weights best.pt --format onnx    # Laptop
python scripts/export_model.py --weights best.pt --format engine  # On Jetson

# 5. Full benchmark suite
python scripts/run_benchmark_suite.py --backends tensorrt --weights-engine best_fp16.engine \
    --data-dir data/VisDrone2019-DET-val --jetson --power-mode 15w

# 6. Analyze failure cases
python scripts/analyze_failures.py
```

### Jetson C++ Deployment

```bash
# Transfer video + model to Jetson
scp video.mp4 vee@192.168.55.1:~/drone_project/data/

# Sync C++ source and build
./src/cpp_inference/scripts/sync_to_jetson.sh build

# Run on Jetson (15W mode for best FPS)
ssh vee@192.168.55.1
cd ~/drone_project/src/cpp_inference/build
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
    ./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                      --source ~/drone_project/data/video.mp4 --imgsz 960

# Copy annotated output back to laptop
scp vee@192.168.55.1:~/drone_project/outputs/cpp_inference_test.mp4 .
```

## Jetson Benchmark Details

### Power Modes

| Mode | ID | GPU Clock | Best For |
|------|----|-----------|----------|
| 15W | 0 | 612 MHz (fixed) | **Best FPS** -- stable clocks outperform MAXN |
| 25W | 1 | 918 MHz | Default mode |
| MAXN_SUPER | 2 | Uncapped | More power, but thermal throttling -> slower |
| 7W | 3 | 408 MHz | Battery saving (requires reboot) |

### Per-Class AP@50 (FP16 vs INT8 vs SAHI)

| Class | FP16 | INT8 | SAHI 640/0.25 |
|-------|------|------|---------------|
| car | 0.798 | 0.713 | 0.792 |
| pedestrian | 0.488 | 0.305 | **0.594** |
| motor | 0.502 | 0.405 | 0.556 |
| bus | 0.503 | 0.263 | 0.515 |
| people | 0.380 | 0.270 | **0.456** |
| van | 0.388 | 0.165 | **0.437** |
| truck | 0.376 | 0.231 | 0.353 |
| bicycle | 0.184 | 0.110 | **0.254** |
| awning-tricycle | 0.144 | 0.108 | 0.197 |

SAHI's biggest gains are on the classes that INT8 hurts most -- small and medium objects get a second life through tiling.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Model | YOLOv8n | 3M params, fits 8GB unified memory, 37.7 FPS (C++) on Jetson |
| Tracker | ByteTrack | No ReID overhead, better for aerial top-down views |
| Precision | FP16 | INT8 costs 31.8% mAP, FP16 is the sweet spot |
| Resolution | 960 | +46% mAP over 640 for small objects |
| SAHI config | 640/0.25 | +10.6% mAP, best accuracy/speed tradeoff |
| Power mode | 15W | Stable clocks outperform MAXN (no thermal throttling) |
| Production inference | C++ TensorRT | 88% faster than Python, 17x less RAM, no runtime dependency |

**Note on FPS numbers**: "28 FPS on Jetson" in the design table above refers to the Python TensorRT backend measured by the benchmark suite (which includes mAP validation overhead). The C++ binary reaches 37.7 FPS on the same hardware because it eliminates Python overhead and does GPU-only preprocessing.

## C++ Inference Technical Notes

- **TensorRT 10.x API**: Uses `setTensorAddress()` + `enqueueV3()` (not deprecated binding API)
- **Engine output**: `1x14x18900` (4 bbox + 10 VisDrone classes, 18900 anchors at imgsz=960)
- **CUDA letterbox preprocessing**: Matches Python exactly (ratio, padding, gray fill, BGR-to-RGB, HWC-to-CHW)
- **Per-class NMS**: Same algorithm as Python backend, detections match within +/-1 per frame
- **Build requirements**: CMake 3.18+, CUDA 12.6+, TensorRT 10.3, OpenCV 4.x, g++ 11+
- **Video codec**: Jetson OpenCV falls back to mp4v container; output plays in VLC

## Known Limitations

- **Small objects**: Below ~10px height, detection drops sharply. SAHI recovers some but at 4.2 FPS.
- **Night scenes**: Model trained only on daytime VisDrone data; no night/IR augmentation.
- **Dense crowds**: Overlapping detections get merged by NMS; tracking is needed to separate individuals.
- **Motion blur**: Fast drone movement reduces detection confidence.
- **INT8 quantization**: Not viable for YOLOv8n on VisDrone (31.8% mAP drop). A larger model (YOLOv8s) may tolerate INT8 better but hasn't been tested.
- **Video output codec**: Jetson OpenCV doesn't support XVID in MP4 container; falls back to mp4v (playable in VLC but not all players).

## Dataset

This project uses the **VisDrone** dataset for training and evaluation:

> Zhu, P., Wen, L., Du, D., Fu, X., Hu, Q., & Li, H. (2021). Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7380-7399.

- **Dataset**: [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) -- 10 classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Training split**: 6,471 images; **Validation split**: 548 images
- **Key challenge**: Small objects (average ~20px height), dense scenes, aerial perspective

## Benchmark Reports

Full reports with latency, mAP, memory, and power breakdowns:
- [`outputs/benchmark_suite/jetson_benchmark_report.md`](outputs/benchmark_suite/jetson_benchmark_report.md) -- 7 Jetson scenarios
- [`outputs/benchmark_suite/laptop_benchmark_report.md`](outputs/benchmark_suite/laptop_benchmark_report.md) -- 6 laptop scenarios
- [`src/cpp_inference/README.md`](src/cpp_inference/README.md) -- C++ inference build, usage, and measured benchmarks
