from dataclasses import dataclass, field

import numpy as np


@dataclass
class TextLine:
    text: str
    confidence: float
    bbox: np.ndarray = field(repr=False, compare=False)
    lang: str = "unknown"
    original_text: str | None = None
    was_mixed: bool = False
