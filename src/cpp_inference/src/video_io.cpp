#include "video_io.h"
#include "types.h"

#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>

// ---------------------------------------------------------------------------
// VideoIO
// ---------------------------------------------------------------------------
VideoIO::VideoIO() = default;

VideoIO::~VideoIO() {
    close();
}

bool VideoIO::open(const std::string& source) {
    // Try to parse as camera index (integer)
    bool is_camera = false;
    int cam_idx = -1;
    try {
        size_t pos = 0;
        cam_idx = std::stoi(source, &pos);
        if (pos == source.size()) {
            is_camera = true;
        }
    } catch (...) {
        // Not an integer — treat as file/RTSP
    }

    if (is_camera) {
        cap_.open(cam_idx);
    } else {
        // For RTSP streams, try GStreamer backend first (common on Jetson)
        if (source.find("rtsp://") == 0) {
            cap_.open(source, cv::CAP_GSTREAMER);
            if (!cap_.isOpened()) {
                fprintf(stderr, "[WARN] GStreamer failed for RTSP, trying default backend\n");
                cap_.open(source);
            }
        } else {
            cap_.open(source);
        }
    }

    if (!cap_.isOpened()) {
        fprintf(stderr, "[ERROR] Cannot open video source: %s\n", source.c_str());
        return false;
    }

    fprintf(stderr, "[VideoIO] Opened: %s (%dx%d @ %.1f FPS)\n",
            source.c_str(), width(), height(), fps());
    return true;
}

bool VideoIO::read(cv::Mat& frame) {
    return cap_.read(frame);
}

bool VideoIO::open_writer(const std::string& output_path, double fps, int fourcc) {
    output_path_ = output_path;

    if (fourcc == 0) {
        // Try codecs in order of compatibility: XVID > mp4v > MJPG
        // XVID is widely compatible with .mp4 containers on all platforms
        // Fall back to mp4v if XVID isn't available
        fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    }

    int w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "[ERROR] Invalid frame size for writer: %dx%d\n", w, h);
        return false;
    }

    writer_.open(output_path, fourcc, fps, cv::Size(w, h));
    if (!writer_.isOpened()) {
        fprintf(stderr, "[ERROR] Cannot open video writer: %s\n", output_path.c_str());
        return false;
    }

    writer_opened_ = true;
    fprintf(stderr, "[VideoIO] Writer opened: %s (%dx%d @ %.1f FPS)\n",
            output_path.c_str(), w, h, fps);
    return true;
}

void VideoIO::write(const cv::Mat& frame) {
    if (writer_opened_) {
        writer_ << frame;
    }
}

void VideoIO::close() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    if (writer_opened_) {
        writer_.release();
        writer_opened_ = false;
    }
}

int VideoIO::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoIO::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double VideoIO::fps() const {
    double f = cap_.get(cv::CAP_PROP_FPS);
    return f > 0 ? f : 30.0;  // default to 30 if unknown
}

int VideoIO::frame_count() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
}

// ---------------------------------------------------------------------------
// Annotation
// ---------------------------------------------------------------------------
void annotate_frame(cv::Mat& frame,
                    const std::vector<Detection>& detections,
                    float fps,
                    int frame_idx) {
    // Draw each detection
    for (const auto& det : detections) {
        int cls = det.class_id;
        if (cls < 0 || cls >= NUM_CLASSES) continue;

        // Get class color (BGR)
        cv::Scalar color(CLASS_COLORS[cls][0],
                         CLASS_COLORS[cls][1],
                         CLASS_COLORS[cls][2]);

        // Draw bounding box
        cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
        cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
        cv::rectangle(frame, pt1, pt2, color, 2);

        // Draw label
        char label[64];
        snprintf(label, sizeof(label), "%s %.0f%%",
                 CLASS_NAMES[cls], det.score * 100.0f);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Label background
        int label_y = std::max(static_cast<int>(det.y1) - 5, text_size.height + 2);
        cv::Point bg_tl(static_cast<int>(det.x1), label_y - text_size.height - 2);
        cv::Point bg_br(static_cast<int>(det.x1) + text_size.width + 4, label_y + 2);
        cv::rectangle(frame, bg_tl, bg_br, color, cv::FILLED);

        // Label text (black for readability on colored background)
        cv::putText(frame, label,
                    cv::Point(static_cast<int>(det.x1) + 2, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // FPS overlay (top-left, white text with black shadow)
    char fps_text[64];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
    cv::putText(frame, fps_text, cv::Point(12, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(frame, fps_text, cv::Point(10, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // Frame counter (if provided)
    if (frame_idx >= 0) {
        char det_text[64];
        snprintf(det_text, sizeof(det_text), "Detections: %d  Frame: %d",
                 static_cast<int>(detections.size()), frame_idx);
        cv::putText(frame, det_text, cv::Point(12, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, det_text, cv::Point(10, 56),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}