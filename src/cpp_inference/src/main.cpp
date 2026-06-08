#include "trt_engine.h"
#include "cuda_preprocess.h"
#include "video_io.h"
#include "types.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Simple CLI argument parser
// ---------------------------------------------------------------------------
struct Args {
    std::string engine_path;
    std::string source;      // video file path, camera index, or RTSP URL
    int imgsz = 960;
    float conf = 0.25f;
    float iou = 0.45f;
    int max_det = 300;
    std::string output;      // output video path (empty = no output)
    bool show = false;       // display live (press 'q' or ESC to quit)
    int warmup = 10;
};

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Required:\n"
        "  --engine PATH     TensorRT engine file (.engine)\n"
        "  --source PATH     Video file, camera index (e.g. 0), or RTSP URL\n"
        "\n"
        "Optional:\n"
        "  --imgsz N         Input resolution (default: 960)\n"
        "  --conf F          Confidence threshold (default: 0.25)\n"
        "  --iou F           NMS IoU threshold (default: 0.45)\n"
        "  --max-det N       Max detections per image (default: 300)\n"
        "  --output PATH     Output video path (default: none)\n"
        "  --show            Display live (press 'q' or ESC to quit)\n"
        "  --warmup N        Number of warmup iterations (default: 10)\n"
        "\n"
        "Examples:\n"
        "  %s --engine models/best_fp16.engine --source data/test_video4.mp4\n"
        "  %s --engine models/best_fp16.engine --source 0 --show\n"
        "  %s --engine models/best_fp16.engine --source rtsp://... --output out.mp4\n",
        prog, prog, prog, prog);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            args.engine_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            args.source = argv[++i];
        } else if (arg == "--imgsz" && i + 1 < argc) {
            args.imgsz = std::atoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            args.conf = std::atof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            args.iou = std::atof(argv[++i]);
        } else if (arg == "--max-det" && i + 1 < argc) {
            args.max_det = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--show") {
            args.show = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "[ERROR] Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (args.engine_path.empty() || args.source.empty()) {
        fprintf(stderr, "[ERROR] --engine and --source are required\n\n");
        print_usage(argv[0]);
        std::exit(1);
    }

    return args;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    fprintf(stderr, "=== Drone Inference (C++ TensorRT) ===\n");
    fprintf(stderr, "Engine:  %s\n", args.engine_path.c_str());
    fprintf(stderr, "Source:  %s\n", args.source.c_str());
    fprintf(stderr, "imgsz:   %d\n", args.imgsz);
    fprintf(stderr, "conf:    %.2f\n", args.conf);
    fprintf(stderr, "iou:     %.2f\n", args.iou);
    fprintf(stderr, "max_det: %d\n", args.max_det);
    fprintf(stderr, "\n");

    // ---- Initialize TensorRT engine ----
    TRTEngine engine(args.engine_path, args.imgsz);
    engine.warmup(args.warmup);

    // ---- Open video source ----
    VideoIO video;
    if (!video.open(args.source)) {
        fprintf(stderr, "[ERROR] Failed to open video source\n");
        return 1;
    }

    // ---- Set up output writer ----
    if (!args.output.empty()) {
        if (!video.open_writer(args.output, video.fps())) {
            fprintf(stderr, "[ERROR] Failed to open output writer\n");
            return 1;
        }
    }

    // ---- Frame loop ----
    cv::Mat frame;
    int frame_count = 0;
    std::vector<float> latencies;
    latencies.reserve(10000);

    fprintf(stderr, "[INFO] Starting inference loop...\n");

    while (video.read(frame)) {
        if (frame.empty()) break;

        auto t_total_start = std::chrono::high_resolution_clock::now();

        // 1. CUDA letterbox preprocess
        LetterboxInfo lb_info;
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        cuda_preprocess(frame, engine.input_buffer(), args.imgsz,
                        engine.cuda_stream(), lb_info);
        auto t_pre_end = std::chrono::high_resolution_clock::now();
        float preprocess_ms = std::chrono::duration<float, std::milli>(t_pre_end - t_pre_start).count();

        // 2. TensorRT inference + decode + NMS
        InferenceResult result;
        engine.infer(engine.input_buffer(), result, lb_info,
                     args.conf, args.iou, args.max_det);

        result.preprocess_ms = preprocess_ms;
        result.total_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t_total_start).count();
        latencies.push_back(result.total_ms);

        // 3. Compute FPS
        float fps = result.total_ms > 0 ? 1000.0f / result.total_ms : 0;

        // 4. Annotate frame
        annotate_frame(frame, result.detections, fps, frame_count);

        // 5. Write / display
        if (!args.output.empty()) {
            video.write(frame);
        }

        if (args.show) {
            cv::imshow("Drone Inference", frame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {  // ESC or 'q'
                fprintf(stderr, "[INFO] Quit key pressed\n");
                break;
            }
        }

        // 6. Progress log
        frame_count++;
        if (frame_count % 100 == 0) {
            float avg_100 = 0;
            int n = std::min(100, static_cast<int>(latencies.size()));
            for (int i = latencies.size() - n; i < static_cast<int>(latencies.size()); i++) {
                avg_100 += latencies[i];
            }
            avg_100 /= n;
            fprintf(stderr, "[INFO] Frame %d: %d detections, %.1f FPS (avg %.1f ms over last %d frames)\n",
                    frame_count, static_cast<int>(result.detections.size()),
                    fps, avg_100, n);
        }
    }

    // ---- Summary statistics ----
    if (!latencies.empty()) {
        float total_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
        float avg_ms = total_ms / latencies.size();

        // Sort for percentiles
        std::vector<float> sorted = latencies;
        std::sort(sorted.begin(), sorted.end());
        float p50 = sorted[sorted.size() / 2];
        float p95 = sorted[static_cast<size_t>(sorted.size() * 0.95)];

        fprintf(stderr, "\n=== Results ===\n");
        fprintf(stderr, "Frames processed: %d\n", frame_count);
        fprintf(stderr, "Average FPS:      %.1f\n", 1000.0f * frame_count / total_ms);
        fprintf(stderr, "Average latency:   %.1f ms\n", avg_ms);
        fprintf(stderr, "P50 latency:       %.1f ms\n", p50);
        fprintf(stderr, "P95 latency:       %.1f ms\n", p95);
    }

    video.close();
    fprintf(stderr, "[INFO] Done\n");
    return 0;
}