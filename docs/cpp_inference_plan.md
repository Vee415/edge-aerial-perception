# C++ TensorRT Inference Binary — Implementation Plan

Production C++ inference binary for Jetson Orin Nano Super. Replaces Python TensorRT backend with compiled C++ binary using native TensorRT C++ API, CUDA preprocessing, and custom NMS. Same detections, lower latency, production-ready.

## Goal

```
Current (Python):
  python run_inference.py → TensorRT Python bindings → inference (22 FPS)

Target (C++):
  ./drone_inference → TensorRT C++ API → CUDA preprocess → custom NMS (26+ FPS)
```

A single compiled binary that loads `best_fp16.engine`, reads video, runs inference, draws boxes, saves output. No Python runtime needed.

## Project Structure

```
src/cpp_inference/
├── CMakeLists.txt              # Build system
├── README.md                   # Build & usage instructions
├── include/
│   ├── trt_engine.h            # TensorRT engine wrapper class
│   ├── cuda_preprocess.h       # CUDA resize + normalize kernel
│   ├── nms.h                   # C++ NMS implementation
│   ├── video_io.h              # OpenCV video read/write/display
│   └── types.h                 # Detection, Result structs
├── src/
│   ├── main.cpp                # CLI entry point (argparse)
│   ├── trt_engine.cpp          # TensorRT C++ API: load, allocate, infer
│   ├── cuda_preprocess.cu      # CUDA kernel: resize + normalize + HWC→CHW
│   ├── nms.cpp                 # NMS + confidence filtering
│   └── video_io.cpp            # Video capture + annotation + writer
└── scripts/
    └── sync_to_jetson.sh        # Rsync build to Jetson
```

## Architecture

```
main.cpp
   │
   ├── VideoIO (OpenCV)
   │     ├── read frame from video/camera
   │     └── write annotated frame to output
   │
   ├── CUDAPreprocess (CUDA kernel)
   │     ├── resize to 960×960 (bilinear interpolation on GPU)
   │     ├── normalize (÷255.0)
   │     └── HWC → CHW transpose (on GPU, zero-copy to TensorRT input)
   │
   ├── TRTEngine (TensorRT C++ API)
   │     ├── load .engine file
   │     ├── allocate input/output GPU buffers
   │     ├── enqueue inference (async, CUDA stream)
   │     └── copy output back to CPU
   │
   ├── NMS (C++)
   │     ├── decode raw output (84 × 8400 for YOLOv8n)
   │     ├── filter by confidence threshold
   │     └── apply NMS per class
   │
   └── Annotate (OpenCV)
         ├── draw bounding boxes + class labels
         └── overlay FPS counter
```

## Implementation Steps

### Step 1: types.h — Data Structures

```cpp
// include/types.h
#pragma once
#include <vector>
#include <string>

struct Detection {
    float x1, y1, x2, y2;    // bounding box (pixel coords, original image space)
    float score;              // confidence
    int class_id;             // 0-9 for VisDrone
};

struct InferenceResult {
    std::vector<Detection> detections;
    float inference_ms;       // TensorRT enqueue time
    float total_ms;           // full pipeline: preprocess + infer + postprocess
};
```

VisDrone class names (for annotation):
```cpp
const char* CLASS_NAMES[] = {
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
};
```

### Step 2: trt_engine.h / trt_engine.cpp — TensorRT C++ API

This is the core. Loads the serialized engine, allocates GPU memory, runs inference.

**Key TensorRT C++ API calls:**

```cpp
// Load engine
nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
std::ifstream file(engine_path, std::ios::binary);
// read file into buffer
nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer, size);

// Create execution context
nvinfer1::IExecutionContext* context = engine->createExecutionContext();

// Get I/O tensor names
int input_index = engine->getBindingIndex("images");
int output_index = engine->getBindingIndex("output0");  // check with trtexec

// Allocate GPU buffers
void* buffers[2];
cudaMalloc(&buffers[input_index],  input_size);   // 1×3×960×960 × float
cudaMalloc(&buffers[output_index], output_size);  // 1×84×8400 × float (YOLOv8n)

// Run inference (async)
context->enqueueV2(buffers, stream, nullptr);

// Copy output back
cudaMemcpyAsync(host_output, buffers[output_index], output_size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**Important details:**

- **Engine file format**: Serialized by `trtexec --saveEngine`. Just read the binary file.
- **Binding names**: Run `trtexec --loadEngine=models/best_fp16.engine --verbose` on Jetson to see exact input/output tensor names. YOLOv8n typically uses `images` (input) and `output0` (output).
- **Output shape**: YOLOv8n at 960 → output is `1 × 84 × 8400` where 84 = 4 (bbox) + 10 (classes) per anchor, 8400 = number of anchors. **Verify this on Jetson first with `trtexec`**.
- **Memory**: Input = `3 × 960 × 960 × sizeof(float)` = ~10.5MB. Output = `84 × 8400 × sizeof(float)` = ~2.8MB. Both on GPU.
- **FP16 input**: TensorRT FP16 engine accepts FP32 input and internally converts. Send FP32 input, get FP32 output. The FP16 is inside the engine.

**Buffer management pattern:**

```cpp
class TRTEngine {
public:
    TRTEngine(const std::string& engine_path, int imgsz = 960);
    ~TRTEngine();
    
    void infer(const float* input_data,  // preprocessed CHW float data on GPU
               float* output_data);       // raw output on CPU
    
    int imgsz() const { return imgsz_; }
    
private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    
    void* input_buffer_ = nullptr;   // GPU
    void* output_buffer_ = nullptr;  // GPU
    float* host_output_ = nullptr;   // CPU
    
    int imgsz_;
    int input_index_;
    int output_index_;
    size_t input_size_;
    size_t output_size_;
    
    // Logger (required by TensorRT)
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } logger_;
};
```

**Destructor must free everything:**

```cpp
TRTEngine::~TRTEngine() {
    cudaStreamDestroy(stream_);
    cudaFree(input_buffer_);
    cudaFree(output_buffer_);
    delete[] host_output_;
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}
```

### Step 3: cuda_preprocess.h / cuda_preprocess.cu — GPU Preprocessing

This is where the C++ version beats Python. NumPy preprocessing runs on CPU and copies to GPU. CUDA preprocessing stays on GPU the entire time.

**What it does:**
1. Takes raw OpenCV frame (uint8 HWC, 3 channels)
2. Resizes from (H, W) to (960, 960) — bilinear interpolation on GPU
3. Converts uint8 → float32 and normalizes (/ 255.0)
4. Transposes HWC → CHW (channels-first for TensorRT)
5. Result is in GPU memory, ready for TensorRT input (zero-copy)

**CUDA kernel:**

```cuda
// cuda_preprocess.cu
__global__ void preprocess_kernel(
    const uint8_t* src,      // input image (H×W×3, uint8)
    float* dst,              // output tensor (3×960×960, float32)
    int src_w, int src_h,    // original dimensions
    int dst_size             // target size (960)
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_size || dy >= dst_size) return;
    
    // Map destination pixel to source (bilinear)
    float sx = (float)(dx * src_w) / (float)dst_size;
    float sy = (float)(dy * src_h) / (float)dst_size;
    
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    
    float fx = sx - x0, fy = sy - y0;
    
    // Bilinear interpolation for each channel
    for (int c = 0; c < 3; c++) {
        float val = (1-fx)*(1-fy) * src[(y0*src_w + x0)*3 + c]
                  + fx*(1-fy)     * src[(y0*src_w + x1)*3 + c]
                  + (1-fx)*fy     * src[(y1*src_w + x0)*3 + c]
                  + fx*fy         * src[(y1*src_w + x1)*3 + c];
        // Normalize and write CHW
        dst[c * dst_size * dst_size + dy * dst_size + dx] = val / 255.0f;
    }
}
```

**Host-side launcher:**

```cpp
void cuda_preprocess(const cv::Mat& frame, float* gpu_output, int dst_size, cudaStream_t stream);
// 1. Upload frame to GPU (cudaMemcpyAsync, H×W×3 uint8)
// 2. Launch kernel with dim3(960/16, 960/16) grid
// 3. Returns immediately (async on stream)
```

**Note on letterboxing**: YOLOv8 uses letterbox resize (maintain aspect ratio + pad). The CUDA kernel should match your Python preprocessing exactly. Check `onnx_backend.py` for the exact letterbox logic — you need to replicate it. If your Python uses simple resize (stretch), do the same in CUDA. Consistency matters more than correctness here.

**Alternative (simpler, less performant):** Use OpenCV GPU resize (`cv::cuda::resize`) + a simple normalize kernel. This avoids writing bilinear interpolation yourself. Good for first version.

### Step 4: nms.h / nms.cpp — NMS + Output Decoding

YOLOv8n output shape: `1 × 84 × 8400`

```
84 rows = 4 (bbox: cx, cy, w, h) + 10 (class scores for VisDrone)
8400 columns = number of predictions (anchors across all scales)
```

**Decode raw output:**

```cpp
std::vector<Detection> decode_output(
    const float* output,       // [84 × 8400]
    int num_classes,           // 10
    int num_anchors,           // 8400
    float conf_threshold,      // 0.25
    float iou_threshold,       // 0.45
    int imgsz,                 // 960
    int orig_w, int orig_h     // original image size (for scale)
) {
    std::vector<Detection> detections;
    
    for (int i = 0; i < num_anchors; i++) {
        // Find max class score
        float max_score = 0;
        int max_class = -1;
        for (int c = 0; c < num_classes; c++) {
            float score = output[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }
        
        if (max_score < conf_threshold) continue;
        
        // Decode bbox (cx, cy, w, h) → (x1, y1, x2, y2) in pixel coords
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w  = output[2 * num_anchors + i];
        float h  = output[3 * num_anchors + i];
        
        Detection det;
        det.x1 = (cx - w/2) * orig_w / imgsz;  // scale back to original image
        det.y1 = (cy - h/2) * orig_h / imgsz;
        det.x2 = (cx + w/2) * orig_w / imgsz;
        det.y2 = (cy + h/2) * orig_h / imgsz;
        det.score = max_score;
        det.class_id = max_class;
        detections.push_back(det);
    }
    
    // Apply NMS
    return nms(detections, iou_threshold);
}
```

**NMS implementation:**

```cpp
std::vector<Detection> nms(std::vector<Detection>& detections, float iou_threshold) {
    // Sort by score descending
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;
    
    for (int i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        
        for (int j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;  // per-class NMS
            
            float iou = compute_iou(detections[i], detections[j]);
            if (iou > iou_threshold) suppressed[j] = true;
        }
    }
    return result;
}
```

**⚠️ Verify output shape on Jetson first.** Run:
```bash
trtexec --loadEngine=models/best_fp16.engine --verbose 2>&1 | grep -i binding
```
This will show exact tensor names and dimensions. The `84 × 8400` is for YOLOv8n at 640. At 960 the number of anchors changes. **Check before coding.**

### Step 5: video_io.h / video_io.cpp — Video Input/Output

```cpp
class VideoIO {
public:
    bool open(const std::string& source);   // video file or camera index
    bool read(cv::Mat& frame);
    void write(const cv::Mat& frame);
    void close();
    
    int width() const;
    int height() const;
    double fps() const;
    
private:
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
    std::string output_path_;
};
```

**Annotation function:**

```cpp
void annotate_frame(cv::Mat& frame, const std::vector<Detection>& detections,
                    float fps, const char** class_names) {
    for (const auto& det : detections) {
        cv::rectangle(frame,
                      cv::Point(det.x1, det.y1),
                      cv::Point(det.x2, det.y2),
                      cv::Scalar(0, 255, 0), 2);
        
        std::string label = std::string(class_names[det.class_id]) + 
                           " " + std::to_string((int)(det.score * 100)) + "%";
        cv::putText(frame, label, cv::Point(det.x1, det.y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    // FPS overlay
    cv::putText(frame, "FPS: " + std::to_string((int)fps),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
}
```

### Step 6: main.cpp — CLI Entry Point

```cpp
int main(int argc, char** argv) {
    // Parse args: --engine, --source, --imgsz, --conf, --iou, --output
    // ...
    
    TRTEngine engine(engine_path, imgsz);
    CUDAPreprocess preprocessor(imgsz);
    VideoIO video;
    video.open(source);
    
    cv::Mat frame;
    int frame_count = 0;
    std::vector<float> latencies;
    
    while (video.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // 1. Preprocess on GPU
        preprocessor.run(frame, engine.input_buffer(), engine.stream());
        
        // 2. TensorRT inference
        engine.infer();
        
        // 3. Postprocess (decode + NMS)
        auto detections = decode_output(engine.output(), num_classes, num_anchors,
                                         conf, iou, imgsz, frame.cols, frame.rows);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
        
        // 4. Annotate + write
        annotate_frame(frame, detections, 1000.0f / ms, CLASS_NAMES);
        video.write(frame);
        
        frame_count++;
        if (frame_count % 100 == 0) {
            float avg_fps = 1000.0f * 100 / (std::accumulate(latencies.end()-100, latencies.end(), 0.0f));
            std::cout << "Frame " << frame_count << ": " << detections.size()
                      << " det, " << avg_fps << " FPS" << std::endl;
        }
    }
    
    float avg_fps = 1000.0f * frame_count / std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    std::cout << "[✓] " << frame_count << " frames at " << avg_fps << " FPS" << std::endl;
    
    return 0;
}
```

### Step 7: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(drone_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT paths (Jetson defaults)
set(TRT_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT lib dir")
set(TRT_INCLUDE_DIR "/usr/include/nvinfer" CACHE PATH "TensorRT include dir")

# Sources
set(SOURCES
    src/main.cpp
    src/trt_engine.cpp
    src/cuda_preprocess.cu
    src/nms.cpp
    src/video_io.cpp
)

add_executable(drone_inference ${SOURCES})

target_include_directories(drone_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(drone_inference
    nvinfer
    cudart
    ${OpenCV_LIBS}
    stdc++fs
)

# Install target
install(TARGETS drone_inference DESTINATION bin)
```

**⚠️ Jetson-specific**: TensorRT headers are at `/usr/include/nvinfer/` and libs at `/usr/lib/aarch64-linux-gnu/`. These come with JetPack. No separate install needed.

## Build & Run on Jetson

```bash
# On Jetson
cd ~/drone_project/src/cpp_inference
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run on video file
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                 --source ~/drone_project/data/test_video4.mp4 \
                 --imgsz 960 --conf 0.25 --output ~/drone_project/outputs/inference/test_video4_cpp.mp4

# Run on camera
./drone_inference --engine ~/drone_project/models/best_fp16.engine \
                 --source 0 --imgsz 960 --conf 0.25
```

## Pre-Implementation Checklist (DO THIS FIRST)

Before writing any C++, verify these on the Jetson:

```bash
# 1. Check TensorRT engine I/O bindings
ssh vee@192.168.55.1
trtexec --loadEngine=models/best_fp16.engine --verbose 2>&1 | grep -i "binding\|input\|output"

# 2. Check TensorRT C++ headers exist
ls /usr/include/nvinfer/
# Should see: NvInfer.h, NvInferPlugin.h, etc.

# 3. Check TensorRT C++ libs exist
ls /usr/lib/aarch64-linux-gnu/libnvinfer*
# Should see: libnvinfer.so, libnvinfer_plugin.so

# 4. Check CUDA toolkit
nvcc --version
# Should show CUDA 12.x

# 5. Check cmake version
cmake --version
# Need >= 3.18 for CUDA language support

# 6. Check OpenCV
pkg-config --modversion opencv4
# Need >= 4.x

# 7. Run a quick C++ TensorRT smoke test
cat > /tmp/trt_smoke.cpp << 'EOF'
#include <NvInfer.h>
int main() {
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(nullptr);
    return runtime ? 0 : 1;
}
EOF
g++ /tmp/trt_smoke.cpp -lnvinfer -o /tmp/trt_smoke && /tmp/trt_smoke && echo "OK" || echo "FAIL"
```

## Benchmark Plan (Python vs C++)

After C++ binary is working, run same video on both and compare:

```bash
# Python baseline
python scripts/run_inference.py --weights models/best_fp16.engine --backend tensorrt \
    --source data/test_video4.mp4 --imgsz 960

# C++ version
./drone_inference --engine models/best_fp16.engine \
    --source data/test_video4.mp4 --imgsz 960 --conf 0.25
```

| Metric | Python | C++ | Expected Delta |
|--------|--------|-----|---------------|
| Startup time | ~3s | ~0.1s | 30× faster |
| Preprocess | ~8ms (CPU) | ~2ms (GPU) | 4× faster |
| TensorRT infer | ~35ms | ~35ms | Same |
| Postprocess (NMS) | ~3ms | ~1ms | 3× faster |
| Total per frame | ~46ms | ~38ms | ~18% faster |
| FPS | ~22 | ~26 | +18% |
| Peak RAM | ~2.5GB | ~150MB | 17× less |
| Binary size | N/A | ~5MB | — |

## Key References

| Resource | URL |
|----------|-----|
| TensorRT C++ API docs | https://docs.nvidia.com/deeplearning/tensorrt/api/ |
| TensorRT C++ samples | `/usr/src/tensorrt/samples/` on Jetson |
| YOLOv8 TensorRT C++ example | https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8 |
| CUDA preprocessing kernel | https://github.com/CN-DeepVision/preprocess-cuda |
| Jetson CUDA setup | JetPack 6.x includes CUDA toolkit |

## Potential Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `libnvinfer.so` not found | LD_LIBRARY_PATH not set | Add to CMake: `link_directories(/usr/lib/aarch64-linux-gnu)` |
| CUDA kernel compile fails | Wrong CUDA arch flag | Add: `-DCMAKE_CUDA_FLAGS="-arch=sm_87"` (Orin = sm_87) |
| Output shape mismatch | YOLOv8n at 960 ≠ 8400 anchors | Run `trtexec --verbose` to get actual shape |
| NMS gives different results than Python | Different decode logic | Compare raw output values from both, fix offsets |
| Segfault on engine load | Engine built on different TensorRT version | Rebuild engine on same Jetson |
| Preprocess mismatch | Letterbox vs stretch | Match Python preprocessing exactly |
| GStreamer OpenCV conflict | Jetson OpenCV built with GStreamer | Use `cv::VideoCapture` with FFMPEG backend: `cv::CAP_FFMPEG` |

## Success Criteria

1. ✅ Binary compiles and runs on Jetson
2. ✅ Same detection output as Python TensorRT backend (same boxes, same classes, within ±1 pixel)
3. ✅ Faster than Python (target: 26+ FPS vs 22 FPS)
4. ✅ Lower memory usage than Python (target: <500MB vs 2.5GB)
5. ✅ Works with both video files and camera input
6. ✅ Benchmark comparison documented in README