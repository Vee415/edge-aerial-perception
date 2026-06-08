#include "cuda_preprocess.h"
#include "types.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Device-compatible min/max for CUDA kernels
__device__ __forceinline__ int clamp_min(int v, int lo) { return v < lo ? lo : v; }
__device__ __forceinline__ int clamp_max(int v, int hi) { return v > hi ? hi : v; }
__device__ __forceinline__ int clamp_val(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------------------
// CUDA kernel: letterbox resize + normalize + HWC→CHW
// ---------------------------------------------------------------------------
// This kernel replicates the Python letterbox preprocessing exactly:
//   1. Compute ratio = min(imgsz/orig_h, imgsz/orig_w)
//   2. Resize: map letterbox (dst_x, dst_y) back to source (src_x, src_y)
//      using bilinear interpolation
//   3. Areas outside the resized region get padding value (114, 114, 114)
//   4. Convert uint8 → float32 and normalize (/255.0)
//   5. Write as CHW format: dst[c * imgsz * imgsz + y * imgsz + x]
//
// Letterbox coordinate mapping:
//   src_x = (dst_x - pad_w) / ratio
//   src_y = (dst_y - pad_h) / ratio
//   where pad_w = (imgsz - new_w) / 2, pad_h = (imgsz - new_h) / 2
//   and new_w = orig_w * ratio, new_h = orig_h * ratio

__global__ void letterbox_preprocess_kernel(
    const unsigned char* src,  // source image (orig_h × orig_w × 3, BGR, uint8)
    float* dst,                // destination tensor (3 × imgsz × imgsz, float32, RGB-normalized)
    int orig_w, int orig_h,   // source dimensions
    int imgsz,                 // target size
    float ratio,               // min(imgsz/orig_h, imgsz/orig_w)
    int pad_w, int pad_h,      // left and top padding
    int new_w, int new_h       // resized dimensions (before padding)
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= imgsz || dy >= imgsz) return;

    // Check if this pixel is in the padded region
    bool in_pad_left   = dx < pad_w;
    bool in_pad_right  = dx >= pad_w + new_w;
    bool in_pad_top    = dy < pad_h;
    bool in_pad_bottom = dy >= pad_h + new_h;

    float r_val, g_val, b_val;

    if (in_pad_left || in_pad_right || in_pad_top || in_pad_bottom) {
        // Padding region: gray (114, 114, 114)
        // Note: OpenCV uses BGR, but we output RGB
        r_val = 114.0f / 255.0f;
        g_val = 114.0f / 255.0f;
        b_val = 114.0f / 255.0f;
    } else {
        // Map from letterbox coords back to source image
        float src_x = (float)(dx - pad_w) / ratio;
        float src_y = (float)(dy - pad_h) / ratio;

        // Bilinear interpolation
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = clamp_max(x0 + 1, orig_w - 1);
        int y1 = clamp_max(y0 + 1, orig_h - 1);

        // Clamp to image bounds
        x0 = clamp_val(x0, 0, orig_w - 1);
        y0 = clamp_val(y0, 0, orig_h - 1);

        float fx = src_x - (int)src_x;
        float fy = src_y - (int)src_y;

        // Source is BGR: src[y * orig_w * 3 + x * 3 + c]
        // Channel 0 = B, 1 = G, 2 = R
        int base00 = (y0 * orig_w + x0) * 3;
        int base10 = (y0 * orig_w + x1) * 3;
        int base01 = (y1 * orig_w + x0) * 3;
        int base11 = (y1 * orig_w + x1) * 3;

        float b_px = (1-fx)*(1-fy) * src[base00 + 0] + fx*(1-fy) * src[base10 + 0]
                   + (1-fx)*fy     * src[base01 + 0] + fx*fy     * src[base11 + 0];
        float g_px = (1-fx)*(1-fy) * src[base00 + 1] + fx*(1-fy) * src[base10 + 1]
                   + (1-fx)*fy     * src[base01 + 1] + fx*fy     * src[base11 + 1];
        float r_px = (1-fx)*(1-fy) * src[base00 + 2] + fx*(1-fy) * src[base10 + 2]
                   + (1-fx)*fy     * src[base01 + 2] + fx*fy     * src[base11 + 2];

        // Normalize and convert BGR → RGB
        b_val = b_px / 255.0f;
        g_val = g_px / 255.0f;
        r_val = r_px / 255.0f;
    }

    // Write CHW format: [C, H, W] where C=0 is R, C=1 is G, C=2 is B
    // This matches the Python pipeline: img[:,:,::-1].transpose(2,0,1)
    // which converts BGR→RGB and HWC→CHW
    int idx = dy * imgsz + dx;
    dst[0 * imgsz * imgsz + idx] = r_val;  // Channel 0: R
    dst[1 * imgsz * imgsz + idx] = g_val;  // Channel 1: G
    dst[2 * imgsz * imgsz + idx] = b_val;  // Channel 2: B
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
void cuda_preprocess(const cv::Mat& frame, float* gpu_output, int imgsz,
                     void* stream_ptr, LetterboxInfo& lb_info) {
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    // Compute letterbox parameters (must match Python exactly)
    float ratio = std::min(static_cast<float>(imgsz) / orig_h,
                           static_cast<float>(imgsz) / orig_w);
    int new_h = static_cast<int>(orig_h * ratio);
    int new_w = static_cast<int>(orig_w * ratio);
    int dh = imgsz - new_h;
    int dw = imgsz - new_w;
    int pad_h = dh / 2;   // top padding
    int pad_w = dw / 2;   // left padding

    // Fill out letterbox info for coordinate mapping later
    lb_info.ratio = ratio;
    lb_info.pad_w = pad_w;
    lb_info.pad_h = pad_h;
    lb_info.imgsz = imgsz;
    lb_info.orig_w = orig_w;
    lb_info.orig_h = orig_h;

    // Ensure frame is contiguous
    cv::Mat frame_contiguous;
    if (!frame.isContinuous()) {
        frame_contiguous = frame.clone();
    } else {
        frame_contiguous = frame;
    }

    // Upload source image to GPU
    size_t src_size = orig_h * orig_w * 3 * sizeof(unsigned char);
    unsigned char* gpu_src = nullptr;
    cudaError_t err = cudaMalloc(&gpu_src, src_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc for source frame failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpyAsync(gpu_src, frame_contiguous.data, src_size,
                          cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMemcpyAsync source frame failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_src);
        return;
    }

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((imgsz + block.x - 1) / block.x,
              (imgsz + block.y - 1) / block.y);

    letterbox_preprocess_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        gpu_src, gpu_output, orig_w, orig_h, imgsz, ratio, pad_w, pad_h, new_w, new_h);

    // Free the temporary source buffer (async — safe because kernel uses it,
    // and we synchronize via the stream later during inference)
    err = cudaFreeAsync(gpu_src, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (err != cudaSuccess) {
        fprintf(stderr, "[WARN] cudaFreeAsync failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_src);  // fallback
    }
}

void* upload_frame_to_gpu(const cv::Mat& frame, size_t& size) {
    size = frame.rows * frame.cols * 3 * sizeof(unsigned char);
    void* gpu_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc for frame upload failed: %s\n", cudaGetErrorString(err));
        size = 0;
        return nullptr;
    }
    err = cudaMemcpy(gpu_ptr, frame.data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMemcpy for frame upload failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_ptr);
        size = 0;
        return nullptr;
    }
    return gpu_ptr;
}