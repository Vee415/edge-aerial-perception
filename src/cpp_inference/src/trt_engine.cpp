#include "trt_engine.h"
#include "types.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages; show warnings and above
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------
TRTEngine::TRTEngine(const std::string& engine_path, int imgsz)
    : imgsz_(imgsz) {
    // 1. Create logger & runtime
    logger_ = new TRTLogger();
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) {
        fprintf(stderr, "[ERROR] Failed to create TensorRT runtime\n");
        std::exit(1);
    }

    // 2. Load serialized engine from disk
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[ERROR] Cannot open engine file: %s\n", engine_path.c_str());
        std::exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    file.close();

    // 3. Deserialize engine
    engine_ = runtime_->deserializeCudaEngine(buf.data(), size);
    if (!engine_) {
        fprintf(stderr, "[ERROR] Failed to deserialize engine\n");
        std::exit(1);
    }

    // 4. Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        fprintf(stderr, "[ERROR] Failed to create execution context\n");
        std::exit(1);
    }

    // 5. Create CUDA stream
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_));

    // 6. Discover I/O tensor names and shapes using TRT 10 API
    int nb_tensors = engine_->getNbIOTensors();
    fprintf(stderr, "[TRT] Engine has %d I/O tensors\n", nb_tensors);

    for (int i = 0; i < nb_tensors; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

        // Compute volume
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= dims.d[d];

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            input_size_ = vol * sizeof(float);
            fprintf(stderr, "[TRT] Input \"%s\": dims=[", name);
            for (int d = 0; d < dims.nbDims; d++) fprintf(stderr, "%ld%s", (long)dims.d[d], d < dims.nbDims-1 ? "," : "");
            fprintf(stderr, "] vol=%zu size=%zu bytes\n", vol, input_size_);
        } else {
            output_name_ = name;
            output_size_ = vol * sizeof(float);
            // YOLOv8 output: (1, 4+nc, anchors) where nc = dims.d[1]-4
            output_channels_ = dims.d[1];   // e.g. 14 for VisDrone (4+10)
            output_anchors_ = dims.d[2];    // e.g. 18900 at imgsz=960
            fprintf(stderr, "[TRT] Output \"%s\": dims=[", name);
            for (int d = 0; d < dims.nbDims; d++) fprintf(stderr, "%ld%s", (long)dims.d[d], d < dims.nbDims-1 ? "," : "");
            fprintf(stderr, "] channels=%d anchors=%d vol=%zu size=%zu bytes\n",
                    output_channels_, output_anchors_, vol, output_size_);
        }
    }

    if (input_name_.empty() || output_name_.empty()) {
        fprintf(stderr, "[ERROR] Could not find input/output tensors in engine\n");
        std::exit(1);
    }
    fprintf(stderr, "[TRT] Using input=\"%s\" output=\"%s\"\n", input_name_.c_str(), output_name_.c_str());

    // 7. Allocate GPU buffers
    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&device_input_), input_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc input failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&device_output_), output_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc output failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }

    // 8. Allocate host output buffer
    host_output_ = new float[output_size_ / sizeof(float)];

    fprintf(stderr, "[TRT] Engine loaded: %s (imgsz=%d, output=%d×%d)\n",
            engine_path.c_str(), imgsz_, output_channels_, output_anchors_);
}

TRTEngine::~TRTEngine() {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    cudaFree(device_input_);
    cudaFree(device_output_);
    delete[] host_output_;

    // TRT 10.x: use delete instead of destroy()
    delete context_;
    delete engine_;
    delete runtime_;
    delete static_cast<TRTLogger*>(logger_);
}

// ---------------------------------------------------------------------------
// Warmup
// ---------------------------------------------------------------------------
void TRTEngine::warmup(int n) {
    fprintf(stderr, "[TRT] Warming up (%d iterations)...\n", n);
    LetterboxInfo lb_dummy{};
    lb_dummy.ratio = 1.0f;
    lb_dummy.pad_w = 0;
    lb_dummy.pad_h = 0;
    lb_dummy.imgsz = imgsz_;
    lb_dummy.orig_w = imgsz_;
    lb_dummy.orig_h = imgsz_;

    InferenceResult dummy_result;
    for (int i = 0; i < n; i++) {
        infer(device_input_, dummy_result, lb_dummy);
    }
    fprintf(stderr, "[TRT] Warmup complete\n");
}

// ---------------------------------------------------------------------------
// Inference — TRT 10.x API (setTensorAddress + enqueueV3)
// ---------------------------------------------------------------------------
void TRTEngine::infer(const float* gpu_input, InferenceResult& result,
                      const LetterboxInfo& lb_info,
                      float conf, float iou, int max_det) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Copy input to engine buffer (if not already there via cuda_preprocess)
    if (gpu_input != device_input_) {
        cudaMemcpyAsync(device_input_, gpu_input, input_size_,
                        cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_));
    }

    // Set tensor addresses for TRT 10.x
    context_->setTensorAddress(input_name_.c_str(), device_input_);
    context_->setTensorAddress(output_name_.c_str(), device_output_);

    // Run TensorRT inference (async, TRT 10.x style)
    context_->enqueueV3(reinterpret_cast<cudaStream_t>(stream_));

    // Copy output back to host
    cudaMemcpyAsync(host_output_, device_output_, output_size_,
                    cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));

    auto t1 = std::chrono::high_resolution_clock::now();
    result.infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Decode output + NMS
    auto t2 = std::chrono::high_resolution_clock::now();
    result.detections = decode_output(host_output_, lb_info, conf, iou, max_det);
    auto t3 = std::chrono::high_resolution_clock::now();
    result.postprocess_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();
}

// ---------------------------------------------------------------------------
// Decode YOLOv8 output + NMS
// ---------------------------------------------------------------------------
std::vector<Detection> TRTEngine::decode_output(const float* output,
                                                 const LetterboxInfo& lb_info,
                                                 float conf_thresh, float iou_thresh,
                                                 int max_det) {
    // YOLOv8 output shape: (1, 4+nc, anchors) → accessed as output[c * anchors + a]
    // Channel 0-3: cx, cy, w, h (in letterbox space, imgsz×imgsz)
    // Channel 4..4+nc: per-class scores
    const int nc = output_channels_ - 4;   // e.g. 14-4 = 10 for VisDrone
    const int anchors = output_anchors_;
    const int expected_nc = NUM_CLASSES;    // VisDrone = 10

    if (nc != expected_nc) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "[WARN] Engine has %d classes, expected %d. Using engine classes.\n", nc, expected_nc);
            warned = true;
        }
    }
    const int num_classes = nc;

    std::vector<Detection> candidates;
    candidates.reserve(512);

    for (int a = 0; a < anchors; a++) {
        // Find max class score
        float max_score = 0;
        int max_class = -1;
        for (int c = 0; c < num_classes; c++) {
            float s = output[(4 + c) * anchors + a];
            if (s > max_score) {
                max_score = s;
                max_class = c;
            }
        }

        if (max_score < conf_thresh) continue;

        // Decode bbox: cx, cy, w, h → x1, y1, x2, y2 in letterbox space
        float cx = output[0 * anchors + a];
        float cy = output[1 * anchors + a];
        float w  = output[2 * anchors + a];
        float h  = output[3 * anchors + a];

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        Detection det;
        det.score = max_score;
        det.class_id = max_class;

        // Scale from letterbox space to original image space:
        // undo padding then scale by inverse of letterbox ratio
        det.x1 = (x1 - lb_info.pad_w) / lb_info.ratio;
        det.y1 = (y1 - lb_info.pad_h) / lb_info.ratio;
        det.x2 = (x2 - lb_info.pad_w) / lb_info.ratio;
        det.y2 = (y2 - lb_info.pad_h) / lb_info.ratio;

        // Clip to image bounds
        det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(lb_info.orig_w)));
        det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(lb_info.orig_h)));
        det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(lb_info.orig_w)));
        det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(lb_info.orig_h)));

        candidates.push_back(det);
    }

    // Apply per-class NMS
    auto result = nms(candidates, iou_thresh);

    // Cap at max_det
    if (static_cast<int>(result.size()) > max_det) {
        result.resize(max_det);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Per-class NMS
// ---------------------------------------------------------------------------
float TRTEngine::compute_iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter + 1e-6f;

    return inter / union_area;
}

std::vector<Detection> TRTEngine::nms(std::vector<Detection>& dets, float iou_thresh) {
    // Sort by score descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    // Group by class
    int max_class = 0;
    for (const auto& d : dets) {
        max_class = std::max(max_class, d.class_id);
    }
    std::vector<std::vector<int>> class_groups(max_class + 1);

    for (int i = 0; i < static_cast<int>(dets.size()); i++) {
        class_groups[dets[i].class_id].push_back(i);
    }

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    // Per-class greedy NMS
    for (int cls = 0; cls <= max_class; cls++) {
        const auto& group = class_groups[cls];
        if (group.empty()) continue;

        for (int gi = 0; gi < static_cast<int>(group.size()); gi++) {
            int i = group[gi];
            if (suppressed[i]) continue;
            result.push_back(dets[i]);

            for (int gj = gi + 1; gj < static_cast<int>(group.size()); gj++) {
                int j = group[gj];
                if (suppressed[j]) continue;

                if (compute_iou(dets[i], dets[j]) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}