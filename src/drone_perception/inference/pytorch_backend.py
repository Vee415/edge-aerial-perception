"""PyTorch (Ultralytics) inference backend."""

import time

import numpy as np
import torch
from ultralytics import YOLO

from .backend import FrameResult, InferenceBackend


class PyTorchBackend(InferenceBackend):
    """PyTorch inference using Ultralytics YOLO.

    This is the baseline backend — no optimization, direct model inference.
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.45, device: str | int = 0):
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

    def predict(self, frame: np.ndarray) -> FrameResult:
        """Run PyTorch inference on a single BGR frame."""
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            return FrameResult(
                boxes=r.boxes.xyxy.cpu().numpy(),
                scores=r.boxes.conf.cpu().numpy(),
                class_ids=r.boxes.cls.cpu().numpy().astype(int),
                inference_ms=inference_ms,
            )
        else:
            return FrameResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
                class_ids=np.empty(0, dtype=int),
                inference_ms=inference_ms,
            )

    def warmup(self, n: int = 10) -> None:
        """Run warmup inferences."""
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(source=dummy, imgsz=self.imgsz, verbose=False)

    @property
    def name(self) -> str:
        return "PyTorch"

    @property
    def precision(self) -> str:
        return "FP32"