"""TensorRT inference backend — for Jetson Orin Nano Super deployment.

Requires TensorRT to be installed (comes with JetPack on Jetson devices).
Compatible with ultralytics 8.3.0 (Jetson pinned version) and newer.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from .backend import FrameResult, InferenceBackend


class TensorRTBackend(InferenceBackend):
    """TensorRT inference backend using Ultralytics engine loading.

    This backend loads a .engine file exported by Ultralytics/TensorRT.
    Must be used on the same device where the engine was built.

    Handles ultralytics 8.3.0 compatibility issue where engine files
    lack metadata, causing 'AutoBackend' object has no attribute 'task'.
    """

    def __init__(self, engine_path: str | Path, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, device: int = 0):
        from ultralytics import YOLO

        self.engine_path = Path(engine_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

        # Determine precision from filename
        name_lower = self.engine_path.stem.lower()
        if "_int8" in name_lower or "int8" in name_lower:
            self._precision = "INT8"
        elif "_fp16" in name_lower or "half" in name_lower or "fp16" in name_lower:
            self._precision = "FP16"
        else:
            self._precision = "FP16"  # Default assumption for TRT engines

        # Load engine — task="detect" required for ultralytics 8.3.0
        # which can't auto-detect task from engine metadata
        try:
            self.model = YOLO(str(self.engine_path), task="detect")
        except TypeError:
            # Older ultralytics may not accept task kwarg
            self.model = YOLO(str(self.engine_path))

        self._fix_ultralytics_task_bug()

    def _fix_ultralytics_task_bug(self) -> None:
        """Fix ultralytics 8.3.0 bug: engine files lack metadata.

        On first predict, ultralytics calls model.warmup() which accesses
        self.task on the AutoBackend, but engine files don't set this attribute.
        We fix it by:
        1. Running a dummy predict to trigger predictor creation
        2. Catching the AttributeError
        3. Setting task="detect" on the AutoBackend model
        4. Setting done_warmup=True to skip re-warmup
        5. Retrying the predict
        """
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(source=dummy, imgsz=self.imgsz, verbose=False)
        except AttributeError as e:
            if "task" in str(e) and hasattr(self.model, "predictor") and self.model.predictor is not None:
                # Fix: set task attribute on the AutoBackend model
                self.model.predictor.model.task = "detect"
                self.model.predictor.done_warmup = True
                # Retry
                self.model.predict(source=dummy, imgsz=self.imgsz, verbose=False)
            else:
                raise

    def predict(self, frame: np.ndarray) -> FrameResult:
        """Run TensorRT inference on a single BGR frame."""
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
        """Run warmup inferences.

        The initial warmup is done in __init__ via _fix_ultralytics_task_bug.
        Additional warmup frames for timing stability.
        """
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(source=dummy, imgsz=self.imgsz, verbose=False)

    @property
    def name(self) -> str:
        return f"TensorRT-{self._precision}"

    @property
    def precision(self) -> str:
        return self._precision