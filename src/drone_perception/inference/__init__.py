"""Backend-agnostic inference module."""

from .backend import InferenceBackend, FrameResult
from .pytorch_backend import PyTorchBackend
from .onnx_backend import ONNXBackend
from .sahi_backend import SahiBackend

# TensorRT backend only available on Jetson with TensorRT installed
try:
    from .tensorrt_backend import TensorRTBackend
except ImportError:
    TensorRTBackend = None

__all__ = [
    "InferenceBackend", "FrameResult",
    "PyTorchBackend", "ONNXBackend", "SahiBackend", "TensorRTBackend",
]