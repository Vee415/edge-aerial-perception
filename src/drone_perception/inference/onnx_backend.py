"""ONNX Runtime inference backend."""

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .backend import FrameResult, InferenceBackend


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend.

    Uses CUDA execution provider when available, falls back to CPU.
    Handles letterbox preprocessing and NMS postprocessing internally.
    """

    def __init__(self, onnx_path: str | Path, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45):
        self.onnx_path = Path(onnx_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        # Set up ONNX Runtime session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Class names for NMS
        self.class_names = [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor",
        ]

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Letterbox resize with padding. Returns (resized_img, (ratio, pad_w, pad_h, dw, dh))."""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(h * r), int(w * r)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        dh = self.imgsz - new_h
        dw = self.imgsz - new_w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        pad_info = (r, left, top)
        return padded, pad_info

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox and normalize frame for ONNX input."""
        img, self._pad_info = self._letterbox(frame)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis, ...]  # Add batch dim
        return img

    def _postprocess(self, output: np.ndarray, pad_info: tuple) -> FrameResult:
        """Apply confidence filtering, NMS, and scale boxes back to original image coordinates.

        Handles Ultralytics ONNX output format: (1, 4+nc, num_anchors).
        First 4 channels are xywh (center x, center y, width, height).
        Remaining nc channels are per-class scores (no separate objectness).
        """
        ratio, pad_w, pad_h = pad_info

        # Ultralytics ONNX output: (1, 4+nc, anchors) or (4+nc, anchors)
        if output.ndim == 3:
            output = output[0]  # (4+nc, anchors)

        # Transpose to (anchors, 4+nc) for easier per-detection access
        predictions = output.T  # (anchors, 4+nc)

        # Split into boxes and class scores
        boxes_xywh = predictions[:, :4]       # (anchors, 4) — cx, cy, w, h
        class_scores = predictions[:, 4:]      # (anchors, nc)

        # Get max class score as confidence
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(int)

        # Filter by confidence
        mask = scores >= self.conf
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return FrameResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
                class_ids=np.empty(0, dtype=int),
                inference_ms=0,
            )

        # Convert xywh → xyxy
        boxes_xyxy = np.empty_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Per-class NMS
        keep = self._nms(boxes_xyxy, scores, class_ids, self.iou)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale back to original image coordinates (undo letterbox)
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio  # x1
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio  # y1
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio  # x2
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio  # y2

        return FrameResult(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            inference_ms=0,  # Set by predict()
        )

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
             iou_threshold: float) -> list[int]:
        """Per-class Non-Maximum Suppression.

        Args:
            boxes_xyxy: Bounding boxes in xyxy format, shape (N, 4).
            scores: Confidence scores, shape (N,).
            class_ids: Class IDs, shape (N,).
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of indices to keep.
        """
        keep_all = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            # Standard greedy NMS
            order = cls_scores.argsort()[::-1]
            keep = []
            x1, y1, x2, y2 = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                remaining = np.where(iou <= iou_threshold)[0]
                order = order[remaining + 1]

            for k in keep:
                keep_all.append(cls_indices[k])

        return keep_all

    def predict(self, frame: np.ndarray) -> FrameResult:
        """Run ONNX inference on a single BGR frame."""
        input_tensor = self._preprocess(frame)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: input_tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        result = self._postprocess(output[0], self._pad_info)
        result.inference_ms = inference_ms
        return result

    def warmup(self, n: int = 10) -> None:
        """Run warmup inferences."""
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.predict(dummy)

    @property
    def name(self) -> str:
        return "ONNX Runtime"

    @property
    def precision(self) -> str:
        return "FP32"