#pragma once

#include "types.h"

#include <string>
#include <vector>

// Forward-declare TensorRT types to avoid header dependency in this header
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
class ILogger;
}  // namespace nvinfer1

class TRTEngine {
public:
    /// Load a serialized TensorRT engine from disk.
    /// @param engine_path  Path to .engine file
    /// @param imgsz        Input resolution (e.g. 960)
    TRTEngine(const std::string& engine_path, int imgsz = 960);

    /// Free all GPU memory and TensorRT objects.
    ~TRTEngine();

    // Non-copyable
    TRTEngine(const TRTEngine&) = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;

    /// Run inference on preprocessed GPU input.
    /// @param gpu_input  Preprocessed float32 CHW data on GPU (3×imgsz×imgsz)
    /// @param result     Output: decoded detections in original image coords
    /// @param lb_info    Letterbox info for scaling boxes back
    /// @param conf       Confidence threshold
    /// @param iou        NMS IoU threshold
    /// @param max_det    Max detections per image
    void infer(const float* gpu_input, InferenceResult& result,
               const LetterboxInfo& lb_info,
               float conf = 0.25f, float iou = 0.45f, int max_det = 300);

    /// Get the CUDA stream used for async operations.
    void* cuda_stream() const { return stream_; }

    /// Get GPU input buffer (for direct CUDA preprocess write).
    float* input_buffer() const { return device_input_; }

    /// Get the input size in bytes.
    int imgsz() const { return imgsz_; }

    /// Warm up the engine with dummy inputs.
    void warmup(int n = 10);

private:
    // TensorRT objects
    nvinfer1::ILogger* logger_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // CUDA stream
    void* stream_ = nullptr;  // cudaStream_t

    // GPU buffers
    float* device_input_ = nullptr;    // 3×imgsz×imgsz float32
    float* device_output_ = nullptr;   // raw engine output
    float* host_output_ = nullptr;     // CPU copy of output

    // Tensor names (discovered at runtime from engine)
    std::string input_name_;
    std::string output_name_;

    // Engine dimensions
    int imgsz_;
    size_t input_size_ = 0;   // bytes
    size_t output_size_ = 0;  // bytes
    int output_channels_ = 0; // 4 + NUM_CLASSES (e.g. 14 for VisDrone)
    int output_anchors_ = 0;  // depends on imgsz (e.g. 18900 at 960)

    /// Internal: decode raw engine output + NMS.
    std::vector<Detection> decode_output(const float* output,
                                          const LetterboxInfo& lb_info,
                                          float conf, float iou, int max_det);

    /// Internal: per-class NMS.
    std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh);

    /// Internal: compute IoU of two boxes.
    static float compute_iou(const Detection& a, const Detection& b);
};