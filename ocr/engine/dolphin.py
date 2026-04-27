import sys
import os
import json
import tempfile
import numpy as np
from PIL import Image as PILImage, ImageOps

from ..config import Config
from ..schema import TextLine
from .base import OCREngine


class DolphinOCRAdapter(OCREngine):

    def __init__(self, config: Config,
                 model_path: str = "./ocr/hf_model",
                 dolphin_repo: str = "./ocr/Dolphin") -> None:
        sys.path.insert(0, dolphin_repo)
        from ..Dolphin.demo_page import DOLPHIN
    
        print(f"  Loading Dolphin from: {model_path}")
        self._model = DOLPHIN(model_path)
        self._threshold = config.confidence_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Dolphin fue entrenado con documentos a color.
        Si la imagen viene en escala de grises (preprocesada por el compañero),
        mejora el contraste antes de pasársela.
        """
        pil = PILImage.fromarray(image)
        # Detectar si es efectivamente grises (canales R,G,B casi idénticos)
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        if np.std(r.astype(int) - g.astype(int)) < 3:
            pil = ImageOps.autocontrast(pil, cutoff=2)
        return np.array(pil)

    def extract(self, image: np.ndarray) -> list[TextLine]:
        from demo_page import process_single_image

        # Dolphin trabaja con PIL
        imagen_pil = PILImage.fromarray(image)

        # Crear estructura de directorios que Dolphin necesita
        save_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(save_dir, "output_json"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "markdown", "figures"), exist_ok=True)

        process_single_image(
            image=imagen_pil,
            model=self._model,
            save_dir=save_dir,
            image_name="doc",
        )

        # Leer el JSON que Dolphin guardó
        json_path = os.path.join(save_dir, "output_json", "doc.json")
        if not os.path.exists(json_path):
            print(f"  Warning: Dolphin produced no output JSON in {save_dir}")
            return []

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_dolphin_output(data)

    def _parse_dolphin_output(self, data: list) -> list[TextLine]:
        """
        Convierte el JSON de Dolphin a TextLines.
        Filtra elementos tipo 'fig' (figuras sin texto útil).
        Cada elemento tiene: label, text, bbox [x1,y1,x2,y2], reading_order.
        """
        lines = []
        for elem in data:
            label = elem.get("label", "")
            text  = elem.get("text", "").strip()

            # Ignorar figuras y textos vacíos o referencias a imágenes
            if label == "fig" or not text or text.startswith("!["):
                continue

            # Decodificar unicode escapes si vienen como \\uXXXX
            try:
                text = text.encode().decode("unicode_escape")
            except Exception:
                pass

            bbox_raw = elem.get("bbox", [0, 0, 0, 0])  # [x1, y1, x2, y2]
            if len(bbox_raw) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
                bbox = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.int32
                )
            else:
                bbox = np.zeros((4, 2), dtype=np.int32)

            # Dolphin no da score por línea — usamos 1.0 como proxy
            # y dejamos que el threshold de confianza lo maneje el pipeline
            if text:
                lines.append(TextLine(
                    text=text,
                    confidence=1.0,
                    bbox=bbox,
                ))

        return lines