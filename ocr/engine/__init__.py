from .base import OCREngine
from .paddle import PaddleOCRAdapter
from .dolphin import DolphinOCRAdapter
from .dots import DotsOCRAdapter

__all__ = ["OCREngine", "PaddleOCRAdapter", "DolphinOCRAdapter", "DotsOCRAdapter"]
