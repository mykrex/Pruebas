from .base import OCREngine
from .paddle import PaddleOCRAdapter
from .dolphin import DolphinOCRAdapter

__all__ = ["OCREngine", "PaddleOCRAdapter", "DolphinOCRAdapter"]
