#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

struct Detection;

class VideoIO {
public:
    VideoIO();
    ~VideoIO();

    /// Open a video file, camera index, or RTSP stream.
    /// @param source  File path, camera index (e.g. "0"), or RTSP URL
    /// @return true if opened successfully
    bool open(const std::string& source);

    /// Read the next frame. Returns false on EOF or error.
    bool read(cv::Mat& frame);

    /// Set up video writer for output.
    /// @param output_path  Path for the output video file
    /// @param fps          Output FPS (typically matches input FPS)
    /// @param fourcc       Codec fourcc (default MJPG)
    bool open_writer(const std::string& output_path, double fps, int fourcc = 0);

    /// Write an annotated frame to the output video.
    void write(const cv::Mat& frame);

    /// Close all resources.
    void close();

    int width() const;
    int height() const;
    double fps() const;
    int frame_count() const;

private:
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
    std::string output_path_;
    bool writer_opened_ = false;
};

/// Draw detection results on a frame.
void annotate_frame(cv::Mat& frame,
                    const std::vector<Detection>& detections,
                    float fps,
                    int frame_idx = -1);