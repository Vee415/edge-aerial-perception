# Benchmark Report

## System
- Platform: Linux-5.15.148-tegra-aarch64-with-glibc2.35
- GPU: Orin
- Python: 3.10.12
- Date: 2026-06-07T21:17:46.000000

## Detection Accuracy & Latency

| Scenario | Backend | Precision | imgsz | mAP@50 | mAP@50:95 | FPS | p50 (ms) | p95 (ms) | Avg Pwr (W) | Peak Pwr (W) | Pwr Mode | Avg RAM (MB) | Peak RAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trt960_maxn | TensorRT-FP16 | FP16 | 960 | 0.4060 | 0.2607 | 22.0 | 48.2 | 53.5 | 5.18 | 5.88 | MAXN_SUPER | 1501 | 1508 |
| trt960_15w | TensorRT-FP16 | FP16 | 960 | 0.4060 | 0.2607 | 28.1 | 35.9 | 44.6 | 8.70 | 10.57 | 15W | 2567 | 2585 |
| trt960_int8_maxn | TensorRT-INT8 | INT8 | 960 | 0.2770 | 0.1736 | 60.9 | 16.8 | 17.4 | 8.86 | 9.78 | MAXN_SUPER | 2586 | 2597 |
| trt960_int8_15w | TensorRT-INT8 | INT8 | 960 | 0.2770 | 0.1736 | 29.3 | 35.1 | 36.6 | 6.27 | 6.51 | 15W | 2555 | 2563 |
| trt640_maxn | TensorRT-FP16 | FP16 | 640 | 0.0001 | 0.0000 | 23.4 | 44.8 | 46.9 | 5.01 | 5.48 | MAXN_SUPER | 1493 | 1498 |
| sahi_640_o0.25 | SAHI(TensorRT-FP16) | FP16 | 960 | 0.4492 | 0.2808 | 4.2 | 284.4 | 346.1 | 5.46 | 10.33 | MAXN_SUPER | 1536 | 1560 |
| sahi_480_o0.25 | SAHI(TensorRT-FP16) | FP16 | 960 | 0.4023 | 0.2503 | 6.5 | 157.5 | 162.8 | 9.96 | 10.81 | MAXN_SUPER | 1607 | 1618 |

## Per-Class AP@50 Comparison

| Class | trt960_maxn | trt960_15w | trt960_int8_maxn | trt960_int8_15w | trt640_maxn | sahi_640_o0.25 | sahi_480_o0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pedestrian | 0.4876 | 0.4876 | 0.3045 | 0.3045 | 0.0012 | 0.5944 | 0.5547 |
| people | 0.3803 | 0.3803 | 0.2703 | 0.2703 | 0.0000 | 0.4564 | 0.4213 |
| bicycle | 0.1840 | 0.1840 | 0.1104 | 0.1104 | 0.0000 | 0.2539 | 0.2015 |
| car | 0.7980 | 0.7980 | 0.7128 | 0.7128 | 0.0000 | 0.7920 | 0.7639 |
| van | 0.3883 | 0.3883 | 0.1647 | 0.1647 | 0.0000 | 0.4371 | 0.3959 |
| truck | 0.3762 | 0.3762 | 0.2311 | 0.2311 | 0.0000 | 0.3531 | 0.2835 |
| tricycle | 0.2962 | 0.2962 | 0.2009 | 0.2009 | 0.0000 | 0.3377 | 0.2712 |
| awning-tricycle | 0.1444 | 0.1444 | 0.1077 | 0.1077 | 0.0000 | 0.1969 | 0.1698 |
| bus | 0.5026 | 0.5026 | 0.2625 | 0.2625 | 0.0000 | 0.5146 | 0.4458 |
| motor | 0.5020 | 0.5020 | 0.4048 | 0.4048 | 0.0000 | 0.5557 | 0.5156 |

## Memory Usage

| Scenario | Avg RSS (MB) | Peak RSS (MB) | Avg Sys Used (MB) | Peak Sys Used (MB) | Sys Total (MB) |
| --- | --- | --- | --- | --- | --- |
| trt960_maxn | 1230 | 1237 | 1501 | 1508 | 7620 |
| trt960_15w | 1236 | 1243 | 2567 | 2585 | 7620 |
| trt960_int8_maxn | 1475 | 1488 | 2586 | 2597 | 7620 |
| trt960_int8_15w | 1451 | 1461 | 2555 | 2563 | 7620 |
| trt640_maxn | 1215 | 1221 | 1493 | 1498 | 7620 |
| sahi_640_o0.25 | 1264 | 1276 | 1536 | 1560 | 7620 |
| sahi_480_o0.25 | 1334 | 1341 | 1607 | 1618 | 7620 |

## Power Consumption (Jetson)

| Scenario | Avg Total (W) | Peak Total (W) | Avg CPU+GPU (W) | Peak CPU+GPU (W) | Mode | Samples |
| --- | --- | --- | --- | --- | --- | --- |
| trt960_maxn | 5.18 | 5.88 | 1.48 | 1.78 | MAXN_SUPER | 0 |
| trt960_15w | 8.70 | 10.57 | 3.06 | 4.54 | 15W | 0 |
| trt960_int8_maxn | 8.86 | 9.78 | 3.44 | 4.04 | MAXN_SUPER | 0 |
| trt960_int8_15w | 6.27 | 6.51 | 1.27 | 1.42 | 15W | 0 |
| trt640_maxn | 5.01 | 5.48 | 1.43 | 1.67 | MAXN_SUPER | 0 |
| sahi_640_o0.25 | 5.46 | 10.33 | 1.66 | 4.39 | MAXN_SUPER | 0 |
| sahi_480_o0.25 | 9.96 | 10.81 | 4.20 | 4.70 | MAXN_SUPER | 0 |
