# C++ TensorRT Inference for Jetson Orin Nano Super

Production C++ inference binary for real-time aerial object detection. Replaces the Python TensorRT pipeline with a compiled binary using native TensorRT C++ API, CUDA preprocessing, and custom NMS.

## Quick Start

### Build on Jetson

```bash
cd ~/drone_project/src/cpp_inference
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Run

```bash
# Video file inference
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source ~/drone_project/data/test_video4.mp4 \
                  --imgsz 960 --conf 0.25

# With output video
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source ~/drone_project/data/test_video4.mp4 \
                  --imgsz 960 --conf 0.25 \
                  --output ~/drone_project/outputs/cpp_inference.mp4

# Live camera
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source 0 --imgsz 960 --show

# RTSP stream
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                  --source rtsp://192.168.1.100:554/stream --imgsz 960
```

### Sync from Laptop

```bash
# Sync source files only
./scripts/sync_to_jetson.sh

# Sync and build
./scripts/sync_to_jetson.sh build

# Clean and sync
./scripts/sync_to_jetson.sh clean
```

## Architecture

```
main.cpp
   │
   ├── VideoIO (OpenCV) — video/camera input, annotation, output
   │
   ├── CUDAPreprocess — letterbox resize + normalize + HWC→CHW on GPU
   │     Matches Python letterbox exactly: ratio, padding, gray fill
   │
   ├── TRTEngine — TensorRT C++ API
   │     Loads .engine, async enqueue, GPU buffer management
   │
   └── NMS — YOLOv8 output decode + per-class NMS
         Decode (4+nc)×anchors format, scale back to original coords
```

## Measured Performance (Jetson Orin Nano Super, 15W mode)

Tested on `test_video4.mp4` (640×360, 1602 frames), `best_fp16.engine`, `imgsz=960`, `conf=0.25`:

| Metric | Python TRT | C++ TRT | Improvement |
|--------|------------|---------|-------------|
| **Average FPS** | 20.1 | 37.7 | **+88%** |
| **Avg latency** | ~50ms | 26.5ms | **-47%** |
| **P50 latency** | — | 31.1ms | — |
| **P95 latency** | — | 31.3ms | — |
| **Startup time** | ~3s | ~0.1s | **30× faster** |
| **RAM** | ~2.5 GB | ~150 MB | **17× less** |
| **Binary size** | N/A | ~5 MB | — |

Detection counts match Python within ±1 (minor floating-point differences in letterbox).

## Preprocessing Match

The C++ CUDA kernel replicates the Python letterbox preprocessing exactly:

1. Compute `ratio = min(imgsz / orig_h, imgsz / orig_w)`
2. Resize: `(new_w, new_h) = (orig_w * ratio, orig_h * ratio)`
3. Pad to square: `pad_w = (imgsz - new_w) / 2`, `pad_h = (imgsz - new_h) / 2`
4. Fill padding with gray `(114, 114, 114)`
5. Bilinear interpolation for resize
6. uint8 → float32, normalize (/255.0)
7. BGR → RGB channel swap
8. HWC → CHW transpose

Boxes are scaled back: `orig_x = (letterbox_x - pad_w) / ratio`

## Output Decoding

YOLOv8n output format: `(1, 4+nc, anchors)` where:
- Channels 0-3: bbox (cx, cy, w, h) in letterbox space
- Channels 4-4+nc: per-class confidence scores
- anchors: varies by resolution (8400 at 640, more at 960)

**Verify anchor count on your engine:**
```bash
trtexec --loadEngine=models/best_fp16.engine --verbose 2>&1 | grep -i binding
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `libnvinfer.so` not found | Add `link_directories(/usr/lib/aarch64-linux-gnu)` to CMakeLists.txt |
| CUDA arch compile error | Ensure `-arch=sm_87` in CMAKE_CUDA_FLAGS (Orin) |
| Output shape mismatch | Run `trtexec --verbose` and check actual dimensions |
| Segfault on engine load | Engine must be built on same Jetson (TensorRT version must match) |
| Boxes misaligned | Check letterbox ratio/padding matches Python; verify BGR→RGB swap |
| GStreamer OpenCV conflict | Use `cv::CAP_FFMPEG` for video files |