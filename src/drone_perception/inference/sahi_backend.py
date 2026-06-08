"""SAHI (Slicing Aided Hyper Inference) backend wrapper.

Slices the input image into overlapping tiles, runs detection on each tile
using any inner InferenceBackend, then shifts detections back to full-frame
coordinates and merges with per-class NMS.

This dramatically improves small object detection — a 23x40px object that
becomes ~16x28px at imgsz=960 stays 23x40px inside each 640px tile, giving
the feature maps enough resolution to detect it.

Usage:
    backend = ONNXBackend("best.onnx", imgsz=960)
    sahi = SahiBackend(backend, slice_size=640, slice_overlap=0.25)
    result = sahi.predict(frame)  # transparent — same FrameResult as any backend
"""

import time

import cv2
import numpy as np

from .backend import FrameResult, InferenceBackend


class SahiBackend(InferenceBackend):
    """SAHI wrapper that slices images for better small-object detection.

    Implements the Decorator pattern over InferenceBackend — wraps any
    backend (PyTorch, ONNX, TensorRT) and the rest of the pipeline
    doesn't need to change.

    Args:
        backend: Inner inference backend to run on each slice.
        slice_size: Tile size in pixels (both width and height).
        slice_overlap: Overlap ratio between adjacent tiles (0–0.5).
        merge_iou: IoU threshold for merging duplicate detections across
            overlapping tiles.
    """

    def __init__(self, backend: InferenceBackend,
                 slice_size: int = 640,
                 slice_overlap: float = 0.25,
                 merge_iou: float = 0.45):
        self._backend = backend
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.merge_iou = merge_iou

    def predict(self, frame: np.ndarray) -> FrameResult:
        """Run SAHI inference: slice, detect per tile, shift, merge."""
        t0 = time.perf_counter()

        h, w = frame.shape[:2]
        all_boxes = []
        all_scores = []
        all_class_ids = []

        slices = self._generate_slices(h, w)

        for x_off, y_off in slices:
            # Crop tile
            x1 = x_off
            y1 = y_off
            x2 = min(x_off + self.slice_size, w)
            y2 = min(y_off + self.slice_size, h)
            tile = frame[y1:y2, x1:x2]

            # Pad if tile is smaller than slice_size (edge tiles)
            th, tw = tile.shape[:2]
            if th < self.slice_size or tw < self.slice_size:
                padded = np.full((self.slice_size, self.slice_size, 3),
                                 114, dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            # Detect on tile
            result = self._backend.predict(tile)

            if result.num_detections == 0:
                continue

            # Shift boxes from tile coords to full-frame coords
            boxes = result.boxes.copy()
            boxes[:, 0] += x_off  # x1
            boxes[:, 1] += y_off  # y1
            boxes[:, 2] += x_off  # x2
            boxes[:, 3] += y_off  # y2

            # Clip to image bounds (removes padding-area detections)
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            # Filter out degenerate boxes from padding area
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid = (box_w > 1) & (box_h > 1)

            all_boxes.append(boxes[valid])
            all_scores.append(result.scores[valid])
            all_class_ids.append(result.class_ids[valid])

        total_ms = (time.perf_counter() - t0) * 1000

        # No detections across all tiles
        if not all_boxes:
            return FrameResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
                class_ids=np.empty(0, dtype=int),
                inference_ms=total_ms,
            )

        # Concatenate all slice detections
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        # Per-class NMS to merge overlapping detections
        keep = self._nms(boxes, scores, class_ids, self.merge_iou)

        return FrameResult(
            boxes=boxes[keep],
            scores=scores[keep],
            class_ids=class_ids[keep],
            inference_ms=total_ms,
        )

    def _generate_slices(self, h: int, w: int) -> list[tuple[int, int]]:
        """Generate (x_offset, y_offset) for each slice tile.

        Step between adjacent tiles = slice_size * (1 - overlap).
        The last tile in each dimension extends to the image edge.
        If the image fits in one tile, returns a single full-image slice.
        """
        step = int(self.slice_size * (1 - self.slice_overlap))

        xs = list(range(0, w - self.slice_size + 1, step))
        ys = list(range(0, h - self.slice_size + 1, step))

        # Ensure we cover the right/bottom edge
        if not xs or xs[-1] + self.slice_size < w:
            xs.append(max(0, w - self.slice_size))
        if not ys or ys[-1] + self.slice_size < h:
            ys.append(max(0, h - self.slice_size))

        return [(x, y) for y in ys for x in xs]

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
             class_ids: np.ndarray, iou_threshold: float) -> list[int]:
        """Per-class Non-Maximum Suppression for merging slice detections.

        Identical logic to ONNXBackend._nms — kept as a static method
        so SahiBackend has no dependency on ONNXBackend internals.
        """
        keep_all = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            order = cls_scores.argsort()[::-1]
            keep = []
            x1, y1, x2, y2 = (cls_boxes[:, 0], cls_boxes[:, 1],
                              cls_boxes[:, 2], cls_boxes[:, 3])
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

    def warmup(self, n: int = 10) -> None:
        """Warmup the inner backend with slice-sized dummy frames."""
        dummy = np.random.randint(
            0, 255, (self.slice_size, self.slice_size, 3), dtype=np.uint8
        )
        for _ in range(n):
            self._backend.predict(dummy)

    @property
    def name(self) -> str:
        return f"SAHI({self._backend.name})"

    @property
    def precision(self) -> str:
        return self._backend.precision