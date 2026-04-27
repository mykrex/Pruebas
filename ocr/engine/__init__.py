from .base import OCREngine
from .paddle import PaddleOCRAdapter
from .dots import DotsOCRAdapter

__all__ = ["OCREngine", "PaddleOCRAdapter", "DotsOCRAdapter"]
