# Benchmark Report

## System
- Platform: Windows-11-10.0.26200-SP0
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Python: 3.12.13
- Date: 2026-06-07T20:46:23.800170

## Detection Accuracy & Latency

| Scenario | Backend | Precision | imgsz | mAP@50 | mAP@50:95 | FPS | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sahi_640_o0.25 | SAHI(ONNX Runtime) | FP32 | 960 | 0.4508 | 0.2819 | 7.9 | 130.6 | 217.4 | 1161 | 1189 |
| sahi_480_o0.25 | SAHI(ONNX Runtime) | FP32 | 960 | 0.4044 | 0.2513 | 5.7 | 165.8 | 257.9 | 1179 | 1208 |
| sahi_640_o0.2 | SAHI(ONNX Runtime) | FP32 | 960 | 0.4518 | 0.2823 | 8.0 | 130.5 | 209.6 | 1179 | 1206 |
| sahi_320_o0.25 | SAHI(ONNX Runtime) | FP32 | 960 | 0.2276 | 0.1336 | 2.7 | 381.0 | 602.7 | 1205 | 1254 |

## Per-Class AP@50 Comparison

| Class | sahi_640_o0.25 | sahi_480_o0.25 | sahi_640_o0.2 | sahi_320_o0.25 |
| --- | --- | --- | --- | --- |
| pedestrian | 0.5952 | 0.5547 | 0.5956 | 0.3558 |
| people | 0.4559 | 0.4213 | 0.4549 | 0.2442 |
| bicycle | 0.2582 | 0.2051 | 0.2604 | 0.1025 |
| car | 0.7920 | 0.7641 | 0.7934 | 0.6060 |
| van | 0.4371 | 0.4091 | 0.4374 | 0.2477 |
| truck | 0.3560 | 0.2859 | 0.3543 | 0.1069 |
| tricycle | 0.3447 | 0.2711 | 0.3426 | 0.1308 |
| awning-tricycle | 0.1983 | 0.1712 | 0.1975 | 0.0792 |
| bus | 0.5154 | 0.4455 | 0.5233 | 0.1110 |
| motor | 0.5552 | 0.5160 | 0.5589 | 0.2923 |

## Memory Usage

| Scenario | Avg RSS (MB) | Peak RSS (MB) |
| --- | --- | --- |
| sahi_640_o0.25 | 1161 | 1189 |
| sahi_480_o0.25 | 1179 | 1208 |
| sahi_640_o0.2 | 1179 | 1206 |
| sahi_320_o0.25 | 1205 | 1254 |
