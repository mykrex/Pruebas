"""
dots.ocr es un VLM de 1.7B parámetros de Xiaohongshu (rednote-hilab).
Se usa directamente con transformers + qwen_vl_utils.

Instalación:
    pip install transformers qwen_vl_utils Pillow

Descarga del modelo (~3.5GB):
    huggingface-cli download rednote-hilab/dots.ocr --local-dir ./ocr/DotsOCR
        El nombre del directorio NO puede tener puntos.
        Usa 'DotsOCR' en lugar de 'dots.ocr'.

Notas de hardware:
    - flash_attention_2 requiere CUDA — en MPS usamos 'sdpa'
    - bfloat16 funciona en MPS con PyTorch >= 2.0
    - ~3.5GB de memoria para el modelo (vs ~8GB de Dolphin)
"""

import json
import re
import sys
import os

import numpy as np
import torch
from PIL import Image as PILImage

from ..config import Config
from ..schema import TextLine
from .base import OCREngine


# Prompt oficial de dots.ocr para parseo completo de documentos.
# Se le pide al modelo que devuelva un JSON con bbox, categoría y texto de cada elemento.
_PARSE_PROMPT = """\
Please output the layout information from the document image, \
including each layout element's bbox, its category, and the corresponding \
text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are \
['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', \
'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Content: Extract the complete text within each bbox, \
preserving the original language and script.
4. Reading Order: Elements must be listed in natural reading order.
5. Final Output: The entire output must be a single JSON object.\
"""

# Categorías que contienen texto útil (excluir Picture/Formula/etc.)
_TEXT_CATEGORIES = {
    "Text", "Title", "Section-header", "List-item",
    "Caption", "Footnote", "Table",
}


class DotsOCRAdapter(OCREngine):
    """
    Dots.ocr es un VLM que
    procesa la imagen completa y devuelve un JSON con layout estructurado.
    Esto lo hace más preciso en documentos con layout complejo, pero más
    lento en CPU/MPS que los motores clásicos.
    """

    def __init__(
        self,
        config: Config,
        model_path: str = "./ocr/DotsOCR",
    ) -> None:
        self._threshold = config.confidence_threshold
        self._model_path = os.path.abspath(model_path)
        self._device = self._detect_device()

        print(f"  Loading dots.ocr from : {self._model_path}")
        print(f"  Device                : {self._device}")

        self._model, self._processor = self._load_model()

    # ── Device detection ────────

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── Model loading ───────────

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        # flash_attention_2 solo funciona en CUDA.
        # En MPS y CPU usamos 'sdpa' (scaled dot-product attention), que es
        # el fallback estándar de PyTorch.
        attn_impl = "flash_attention_2" if self._device == "cuda" else "sdpa"

        # bfloat16 está soportado en MPS desde PyTorch 2.0 y en CUDA.
        # En CPU usamos float32 para evitar problemas de precisión.
        dtype = torch.bfloat16 if self._device != "cpu" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._device != "cuda":
            model = model.to(self._device)

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )

        return model, processor

    # ── Preprocessing ───────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        dots.ocr fue entrenado con documentos a color.
        Si la imagen viene en escala de grises, mejora el
        contraste para facilitar la detección de texto en regiones oscuras.
        """
        from PIL import ImageOps
        pil = PILImage.fromarray(image)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.std(r.astype(int) - g.astype(int)) < 3:
            # Imagen en escala de grises → aplicar autocontraste
            pil = ImageOps.autocontrast(pil, cutoff=2)
            return np.array(pil)
        return image

    # ── Inference ───────────────

    def extract(self, image: np.ndarray) -> list[TextLine]:
        from qwen_vl_utils import process_vision_info

        # dots.ocr espera la imagen como PIL
        pil_image = PILImage.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": _PARSE_PROMPT},
                ],
            }
        ]

        # Preparar inputs siguiendo el patrón oficial de dots.ocr
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=8000,   # suficiente para documentos de 1 página
            )

        # Recortar los tokens del prompt del output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"\n── RAW OUTPUT (primeros 300 chars) ──\n{raw_output[:300]}\n────────────────────")

        return self._parse_output(raw_output, image.shape)

    # ── Output parsing ───────────────

    def _parse_output(self, raw: str, image_shape: tuple) -> list[TextLine]:
        """
        Parsea el JSON que devuelve dots.ocr y lo convierte a TextLines.

        dots.ocr devuelve un JSON con esta estructura:
        {
          "elements": [
            {
              "bbox": [x1, y1, x2, y2],
              "category": "Text",
              "text": "contenido del elemento"
            },
            ...
          ]
        }
        """
        # Limpiar bloques de código Markdown si los hay
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Intentar extraer JSON con regex si el modelo devolvió texto extra
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("  Warning: dots.ocr output no parseable como JSON")
                    return self._fallback_plain_text(cleaned)
            else:
                print("  Warning: No se encontró JSON en el output de dots.ocr")
                return self._fallback_plain_text(cleaned)

        # Normalizar estructura — dots.ocr puede devolver lista o dict con "elements"
        if isinstance(data, list):
            elements = data
        elif isinstance(data, dict):
            elements = data.get("elements", data.get("layout", []))
        else:
            return []

        lines: list[TextLine] = []
        h, w = image_shape[:2]

        for elem in elements:
            category = elem.get("category", elem.get("type", "Text"))
            text     = elem.get("text", elem.get("content", "")).strip()
            bbox_raw = elem.get("bbox", [])

            # Filtrar categorías sin texto útil
            if category not in _TEXT_CATEGORIES:
                continue
            if not text:
                continue

            # Convertir bbox [x1,y1,x2,y2] → polígono de 4 puntos
            if len(bbox_raw) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox_raw]
                # Clamp por si el modelo devuelve coords fuera de imagen
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                bbox = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.int32,
                )
            else:
                bbox = np.zeros((4, 2), dtype=np.int32)

            # dots.ocr no da score de confianza por elemento
            lines.append(TextLine(
                text=text,
                confidence=1.0,
                bbox=bbox,
            ))

        return lines

    def _fallback_plain_text(self, raw: str) -> list[TextLine]:
        """
        Si el JSON falla completamente, extraer texto línea por línea
        como fallback de último recurso.
        """
        lines = []
        for linea in raw.split("\n"):
            linea = linea.strip()
            if linea and len(linea) > 2:
                lines.append(TextLine(
                    text=linea,
                    confidence=0.5,
                    bbox=np.zeros((4, 2), dtype=np.int32),
                ))
        return lines