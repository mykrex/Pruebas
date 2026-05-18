import os
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

import numpy as np

from ..config import Config
from ..schema import TextLine
from .base import OCREngine


class PaddleOCRAdapter(OCREngine):
    """PaddleOCR v3.x implementation of OCREngine."""

    def __init__(self, config: Config, **paddle_kwargs) -> None:
        from paddleocr import PaddleOCR
        self._threshold = config.confidence_threshold
        self._engine = PaddleOCR(
            lang="hi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **paddle_kwargs,
        )

    def extract(self, image: np.ndarray) -> list[TextLine]:
        lines: list[TextLine] = []
        for result in self._engine.predict(image):
            res    = result.get("res", result)
            texts  = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            bboxes = res.get("dt_polys", res.get("rec_polys", res.get("det_polys", [])))
            for text, score, bbox in zip(texts, scores, bboxes):
                if score >= self._threshold and text.strip():
                    lines.append(TextLine(
                        text=text.strip(),
                        confidence=round(float(score), 4),
                        bbox=np.array(bbox, dtype=np.int32),
                    ))
        return lines
