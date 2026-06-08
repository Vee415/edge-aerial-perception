"""Abstract inference backend and common data types.

All backends produce the same FrameResult, enabling backend-agnostic
benchmarking and tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameResult:
    """Unified output from any inference backend.

    Attributes:
        boxes: Bounding boxes in xyxy format, shape (N, 4).
        scores: Confidence scores, shape (N,).
        class_ids: Class IDs, shape (N,).
        inference_ms: Single frame inference latency in milliseconds.
    """
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    inference_ms: float

    @property
    def num_detections(self) -> int:
        return len(self.scores)

    def filter_by_conf(self, conf_thresh: float) -> "FrameResult":
        """Return a new FrameResult with only detections above threshold."""
        mask = self.scores >= conf_thresh
        return FrameResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            inference_ms=self.inference_ms,
        )


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> FrameResult:
        """Run inference on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            FrameResult with detections.
        """
        ...

    @abstractmethod
    def warmup(self, n: int = 10) -> None:
        """Run n warmup inferences to stabilize timing."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g., 'PyTorch', 'ONNX', 'TensorRT-FP16')."""
        ...

    @property
    @abstractmethod
    def precision(self) -> str:
        """Precision string (e.g., 'FP32', 'FP16', 'INT8')."""
        ...