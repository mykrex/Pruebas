"""
Pipeline OCR para documentos oficiales de India — PaddleOCR v3.x
=================================================================
Flujo:
  1. Extracción de todo el texto con PaddleOCR (lang="hi")
     - lang="hi" reconoce Devanagari real, permitiendo filtrar por Unicode correctamente
     - lang="en" producía texto latino garbled para scripts indios, inutilizando el filtro
  2. Separación por idioma via rangos Unicode + umbral de confianza
  3. KIE sobre las líneas en inglés → resultado estructurado JSON

Instalación:
  pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install paddleocr
"""

import re
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage
from paddleocr import PaddleOCR


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.7

# Rangos Unicode de scripts indios comunes
INDIAN_SCRIPT_RANGES = [
    (0x0900, 0x097F),  # Devanagari  (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi    (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

# Patrones KIE para documentos oficiales de India
KIE_PATTERNS = {
    "name": [
        r"(?:name|full\s*name|applicant\s*name)[:\s]+([A-Z][a-zA-Z\s\.]+)",
        r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})$",
    ],
    "date_of_birth": [
        r"(?:date\s*of\s*birth|dob|birth\s*date)[:\s]+([\d]{1,2}[\/\-\.][\d]{1,2}[\/\-\.][\d]{2,4})",
        r"(?:born\s*on)[:\s]+([\d]{1,2}[\/\-\.][\d]{1,2}[\/\-\.][\d]{2,4})",
    ],
    "document_number": [
        r"(?:id\s*no|document\s*no|card\s*no|aadhaar|pan|passport\s*no|dl\s*no)[:\s#]*([A-Z0-9]{5,20})",
        r"(?:no\.|number)[:\s]+([A-Z0-9]{5,20})",
    ],
    "gender": [
        r"(?:gender|sex)[:\s]+(male|female|other|m|f)",
    ],
    "nationality": [
        r"(?:nationality|citizen)[:\s]+([A-Za-z]+)",
    ],
    "address": [
        r"(?:address|addr|residence)[:\s]+(.{10,100})",
    ],
    "expiry_date": [
        r"(?:expiry|expiration|valid\s*until|valid\s*thru)[:\s]+([\d]{1,2}[\/\-\.][\d]{1,2}[\/\-\.][\d]{2,4})",
    ],
    "issue_date": [
        r"(?:issue\s*date|date\s*of\s*issue|issued\s*on)[:\s]+([\d]{1,2}[\/\-\.][\d]{1,2}[\/\-\.][\d]{2,4})",
    ],
}


# ─────────────────────────────────────────────
# ETAPA 1: EXTRACCIÓN DE TEXTO (API v3.x)
# ─────────────────────────────────────────────
MAX_SIDE = 1400

def cargar_imagen_rgb(ruta: str) -> np.ndarray:
    """
    Carga cualquier imagen y la convierte siempre a RGB (3 canales).
    Resuelve crashes con imágenes RGBA, grayscale o paleta indexada.
    """
    img = PILImage.open(ruta)
    img = img.convert("RGB")      # cubre RGBA, L, P y cualquier otro modo
    w, h = img.size
    lado_mayor = max(w, h)
    if lado_mayor > MAX_SIDE:
        factor = MAX_SIDE / lado_mayor
        nuevo_w = int(w * factor)
        nuevo_h = int(h * factor)
        img = img.resize((nuevo_w, nuevo_h), PILImage.LANCZOS)
        print(f"  Imagen redimensionada: {w}x{h} → {nuevo_w}x{nuevo_h}")

    return np.array(img)


def extraer_texto(ruta_imagen: str, ocr_engine) -> list:
    """Extrae líneas de texto. Retorna también el array RGB para visualización."""
    imagen_array = cargar_imagen_rgb(ruta_imagen)
    resultados   = ocr_engine.predict(imagen_array)

    lineas = []
    for resultado in resultados:
        res    = resultado.get("res", resultado)
        textos = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        bboxes = res.get("dt_polys", res.get("rec_polys", res.get("det_polys", [])))

        for texto, score, bbox in zip(textos, scores, bboxes):
            if score >= CONFIDENCE_THRESHOLD and texto.strip():
                lineas.append({
                    "texto":     texto.strip(),
                    "confianza": round(float(score), 4),
                    "bbox":      np.array(bbox, dtype=np.int32),
                })

    return lineas, imagen_array


# ─────────────────────────────────────────────
# ETAPA 2: SEPARACIÓN POR IDIOMA
# ─────────────────────────────────────────────

def es_caracter_indio(char):
    code_point = ord(char)
    return any(inicio <= code_point <= fin for inicio, fin in INDIAN_SCRIPT_RANGES)


def clasificar_linea(texto):
    """
    Clasifica una línea como 'english' o 'indian'.
    Si más del 20% de los caracteres alfabéticos son de script indio → indian.
    """
    chars_alfabeticos = [c for c in texto if c.isalpha()]
    if not chars_alfabeticos:
        return "unknown"

    chars_indios = sum(1 for c in chars_alfabeticos if es_caracter_indio(c))
    return "indian" if (chars_indios / len(chars_alfabeticos)) > 0.20 else "english"


def limpiar_latin(texto: str) -> str | None:
    """
    Extrae solo los caracteres latinos de un texto mixto (ej: 'भारत REPUBLIC OF INDIA').
    Retorna None si no queda texto latino significativo después de limpiar.

    Ejemplos:
        'भारत गणराज्य REPUBLIC OF INDIA' → 'REPUBLIC OF INDIA'
        'उपनाम / Surname and Given Name'  → '/ Surname and Given Name'
        'भारत गणराज्य'                   → None
    """
    # Conservar solo caracteres por debajo de U+0900 (Latino, dígitos, puntuación)
    limpio = ''.join(c for c in texto if ord(c) < 0x0900)
    limpio = limpio.strip()

    # Descartar si solo quedan símbolos/puntuación sin texto real
    if not limpio or all(c in '.,;:/-_()[]{}"\' ' for c in limpio):
        return None

    return limpio


def separar_por_idioma(lineas):
    """
    Separa líneas en tres grupos:
      - english : líneas puramente latinas + líneas mixtas (con la parte latina extraída)
      - indian  : líneas puramente en script indio
      - unknown : solo números/símbolos sin caracteres alfabéticos
    """
    grupos = {"english": [], "indian": [], "unknown": []}

    for linea in lineas:
        idioma = clasificar_linea(linea["texto"])

        if idioma == "indian":
            texto_limpio = limpiar_latin(linea["texto"])
            if texto_limpio:
                # Línea mixta: guardar la parte latina para el KIE
                # Se marca como english pero se conserva el original para referencia
                linea_limpia = {
                    **linea,
                    "texto":           texto_limpio,
                    "texto_original":  linea["texto"],
                    "idioma_detectado": "english",
                    "fue_mixta":        True,
                }
                grupos["english"].append(linea_limpia)
                # La bbox se muestra naranja en viz (era predominantemente india)
                linea["idioma_detectado"] = "indian"
                grupos["indian"].append(linea)
            else:
                # Puramente india — descartar del KIE
                linea["idioma_detectado"] = "indian"
                grupos["indian"].append(linea)
        else:
            linea["idioma_detectado"] = idioma
            grupos[idioma].append(linea)

    print("\n── LÍNEAS MIXTAS RECUPERADAS ──")
    for l in grupos["english"]:
        if l.get("fue_mixta"):
            print(f"  original: {l['texto_original']}")
            print(f"  limpio:   {l['texto']}")
    return grupos


# ─────────────────────────────────────────────
# ETAPA 3: KIE
# ─────────────────────────────────────────────

def aplicar_kie(lineas_english):
    entidades = {}
    textos = [l["texto"] for l in lineas_english]
    texto_completo = " ".join(textos)

    # Pasada 1: línea por línea
    for linea in textos:
        for campo, patrones in KIE_PATTERNS.items():
            if campo in entidades:
                continue
            for patron in patrones:
                match = re.search(patron, linea, re.IGNORECASE)
                if match:
                    entidades[campo] = match.group(1).strip()
                    break

    # Pasada 2: texto completo (captura campo+valor en líneas separadas)
    for campo, patrones in KIE_PATTERNS.items():
        if campo in entidades:
            continue
        for patron in patrones:
            match = re.search(patron, texto_completo, re.IGNORECASE)
            if match:
                entidades[campo] = match.group(1).strip()
                break

    return entidades


# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────

def visualizar(imagen_rgb: np.ndarray, grupos: dict, ruta_salida: str):
    """
    Dibuja bounding boxes sobre la imagen original:
      Verde   = línea en inglés
      Naranja = script indio
      Gris    = desconocido (solo números/símbolos)
    Guarda el resultado como imagen.
    """
    # PaddleOCR entrega RGB, cv2 trabaja en BGR
    vis = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

    VERDE   = ( 34, 197,  94)
    NARANJA = (  0, 165, 255)
    GRIS    = (160, 160, 160)

    colores = {"english": VERDE, "indian": NARANJA, "unknown": GRIS}

    for idioma, lineas in grupos.items():
        color = colores[idioma]
        for linea in lineas:
            bbox = linea["bbox"]
            cv2.polylines(vis, [bbox], isClosed=True, color=color, thickness=2)
            # etiqueta pequeña arriba del bbox
            origen = (int(bbox[0][0]), int(bbox[0][1]) - 5)
            cv2.putText(vis, linea["texto"][:30], origen,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(ruta_salida, vis)
    print(f"\n  Imagen guardada en: {ruta_salida}")
    print("  Verde=inglés | Naranja=script indio | Gris=números/símbolos")


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────

def procesar_imagen(ruta_imagen: str, ocr_engine: PaddleOCR,
                    guardar_viz: bool = True) -> dict:
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta_imagen}")

    print(f"\n{'='*50}")
    print(f"Procesando: {ruta.name}")
    print(f"{'='*50}")

    print("→ Etapa 1: Extrayendo texto...")
    lineas, imagen_rgb = extraer_texto(str(ruta), ocr_engine)
    print(f"  {len(lineas)} líneas con confianza >= {CONFIDENCE_THRESHOLD}")

    print("→ Etapa 2: Separando por idioma...")
    grupos = separar_por_idioma(lineas)
    print(f"  Inglés: {len(grupos['english'])} | Indio: {len(grupos['indian'])} | Desconocido: {len(grupos['unknown'])}")

    print("→ Etapa 3: Extrayendo entidades (KIE)...")
    entidades = aplicar_kie(grupos["english"])
    print(f"  Encontradas: {list(entidades.keys())}")

    if guardar_viz:
        print("→ Guardando visualización...")
        ruta_salida = str(ruta.parent / f"{ruta.stem}_resultado.jpg")
        visualizar(imagen_rgb, grupos, ruta_salida)

    return {
        "ruta": str(ruta),
        "lineas_raw": lineas,
        "resumen_idiomas": {
            "english": len(grupos["english"]),
            "indian":  len(grupos["indian"]),
            "unknown": len(grupos["unknown"]),
        },
        "entidades": entidades,
    }


if __name__ == "__main__":

    print("Inicializando PaddleOCR v3.x...")
    ocr = PaddleOCR(
        lang="hi",                           # Hindi: reconoce Devanagari real + latino
        use_doc_orientation_classify=False,  # evita crash en macOS Apple Silicon
        use_doc_unwarping=False,             # desactiva UVDoc (el modelo más pesado)
        use_textline_orientation=False,      # innecesario para documentos ya limpios
    )

    resultado = procesar_imagen("public/visa_p1.png", ocr)

    print("\n── RESULTADO FINAL ──")
    print(json.dumps(resultado["entidades"], indent=2, ensure_ascii=False))

    print("\n── TODAS LAS LÍNEAS DETECTADAS ──")
    for linea in resultado["lineas_raw"]:
        print(f"[{linea['idioma_detectado']:8}] [{linea['confianza']:.2f}] {linea['texto']}")