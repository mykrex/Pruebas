from abc import ABC, abstractmethod

import numpy as np

from ..schema import TextLine


class OCREngine(ABC):
    """
    Contract every OCR backend must satisfy.
    Subclass and implement extract() to swap engines without touching
    language classification, KIE, or visualisation code.
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> list[TextLine]:
        """Run OCR on an RGB numpy image. Return one TextLine per detected region."""
        ...
