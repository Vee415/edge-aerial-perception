#pragma once

#include <cstdint>
#include <string>
#include <vector>

// VisDrone 10-class dataset
constexpr int NUM_CLASSES = 10;
constexpr const char* CLASS_NAMES[NUM_CLASSES] = {
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
};

// VisDrone class colors (BGR) for annotation
constexpr const uint8_t CLASS_COLORS[NUM_CLASSES][3] = {
    {0, 255, 0},     // pedestrian  - green
    {0, 255, 255},   // people      - yellow
    {255, 0, 0},     // bicycle     - blue
    {0, 0, 255},     // car         - red
    {255, 0, 255},   // van         - magenta
    {0, 165, 255},   // truck       - orange
    {255, 255, 0},   // tricycle    - cyan
    {128, 0, 128},   // awning-tri  - purple
    {0, 128, 255},   // bus         - dark orange
    {128, 128, 0},   // motor       - teal
};

struct Detection {
    float x1, y1, x2, y2;  // bounding box in original image pixel coords
    float score;            // confidence score
    int class_id;           // 0..9 for VisDrone
};

struct InferenceResult {
    std::vector<Detection> detections;
    float preprocess_ms = 0;  // CUDA preprocess time
    float infer_ms = 0;       // TensorRT enqueue time
    float postprocess_ms = 0; // NMS + decode time
    float total_ms = 0;       // end-to-end per frame
};

struct LetterboxInfo {
    float ratio;   // scale factor
    int pad_w;     // left padding in pixels
    int pad_h;     // top padding in pixels
    int imgsz;     // target size (960)
    int orig_w;    // original image width
    int orig_h;    // original image height
};