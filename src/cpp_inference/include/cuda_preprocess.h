#pragma once

#include <opencv2/core.hpp>

struct LetterboxInfo;

/// Letterbox resize + normalize + HWC-to-CHW, all on GPU.
///
/// 1. Uploads the BGR uint8 frame to GPU
/// 2. Applies letterbox resize (maintain aspect ratio, pad with gray 114)
/// 3. Converts uint8 → float32 and normalizes (/255.0)
/// 4. Transposes HWC → CHW
/// 5. Result is in gpu_output, ready for TensorRT input
///
/// @param frame        Input BGR image (H×W×3, uint8)
/// @param gpu_output   Pre-allocated GPU buffer (3×imgsz×imgsz, float32)
/// @param imgsz        Target size (960)
/// @param stream       CUDA stream (must match TRTEngine's stream for sync)
/// @param[out] lb_info Letterbox parameters for scaling boxes back
void cuda_preprocess(const cv::Mat& frame, float* gpu_output, int imgsz,
                     void* stream, LetterboxInfo& lb_info);

/// Upload a uint8 BGR frame to GPU and get a GPU pointer.
/// Used internally by cuda_preprocess. Returned pointer must be freed with cudaFree().
/// @param frame  Input frame (H×W×3, uint8, contiguous)
/// @param size   Output: size in bytes of the allocated GPU buffer
/// @return       GPU pointer to the uploaded data
void* upload_frame_to_gpu(const cv::Mat& frame, size_t& size);