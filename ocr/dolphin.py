"""
Pipeline OCR para documentos — Dolphin-v2 (ByteDance)
========================================================================
Dolphin es un modelo VLM que parsea documentos en dos etapas:
  Stage 1: Detecta layout y tipo de cada elemento (título, párrafo, tabla...)
  Stage 2: Extrae el contenido de cada elemento en orden de lectura

Dolphin preserva la estructura del documento,
lo que permite extraer pares campo→valor sin depender tanto de regex.

Flujo:
  1. Cargar imagen → convertir a RGB → redimensionar si necesario
  2. Dolphin parsea la página → JSON con elementos en orden de lectura
  3. Separación Unicode: filtrar texto latino vs scripts indios
  4. Extracción de pares campo→valor del texto latino estructurado
  5. (Opcional) Visualización con bboxes de los elementos detectados

"""

import re
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForImageTextToText


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

MODEL_PATH  = "./hf_model"   # ruta donde descargaste Dolphin-v2
MAX_SIDE    = 1400           # redimensionar si el lado mayor supera esto

# Rangos Unicode de scripts indios
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


# ─────────────────────────────────────────────
# INICIALIZACIÓN DEL MODELO
# ─────────────────────────────────────────────

def cargar_modelo(model_path: str = MODEL_PATH):
    """
    Carga Dolphin-v2 en el dispositivo disponible.
    Prioridad: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"  Usando dispositivo: {device}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return model, processor, device


# ─────────────────────────────────────────────
# ETAPA 1: CARGA Y PREPARACIÓN DE IMAGEN
# ─────────────────────────────────────────────

def cargar_imagen(ruta: str) -> PILImage.Image:
    """
    Carga la imagen, la convierte a RGB y la redimensiona si es necesario.
    Dolphin espera imágenes PIL en modo RGB.
    """
    img = PILImage.open(ruta).convert("RGB")

    w, h = img.size
    lado_mayor = max(w, h)
    if lado_mayor > MAX_SIDE:
        factor = MAX_SIDE / lado_mayor
        nuevo_w = int(w * factor)
        nuevo_h = int(h * factor)
        img = img.resize((nuevo_w, nuevo_h), PILImage.LANCZOS)
        print(f"  Imagen redimensionada: {w}x{h} → {nuevo_w}x{nuevo_h}")

    return img


# ─────────────────────────────────────────────
# ETAPA 2: PARSEO CON DOLPHIN
# ─────────────────────────────────────────────

def parsear_con_dolphin(imagen: PILImage.Image,
                        model, processor, device: str) -> list[dict]:
    """
    Corre Dolphin en modo page-level parsing.

    Dolphin devuelve una lista de elementos en orden de lectura, cada uno con:
      - type  : categoría del elemento (text, title, table, figure, etc.)
      - bbox  : coordenadas [x1, y1, x2, y2] en píxeles
      - content: texto extraído del elemento

    Para fotografías de documentos usa parsing holístico (Stage 2 unificado).
    Para documentos digitales usa parsing por elemento en paralelo.
    """
    # Preparar el input según el formato que espera Dolphin
    # El prompt estándar para page-level parsing
    prompt = "<PAGE_PARSE>"

    inputs = processor(
        text=prompt,
        images=imagen,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    # Decodificar la salida
    raw_output = processor.decode(outputs[0], skip_special_tokens=True)

    # Dolphin devuelve JSON estructurado — parsearlo
    elementos = _parsear_output_dolphin(raw_output)
    return elementos


def _parsear_output_dolphin(raw_output: str) -> list[dict]:
    """
    Dolphin produce su output en formato JSON o Markdown estructurado.
    Esta función lo normaliza a una lista de dicts con {type, content, bbox}.

    El formato exacto puede variar entre versiones — si el JSON falla,
    caemos back a extracción de texto plano línea por línea.
    """
    elementos = []

    # Intentar parsear como JSON primero
    try:
        # Dolphin a veces envuelve el JSON en bloques de código
        json_str = re.sub(r"```(?:json)?\s*|\s*```", "", raw_output).strip()
        data = json.loads(json_str)

        if isinstance(data, list):
            for item in data:
                elementos.append({
                    "type":    item.get("type", "text"),
                    "content": item.get("content", item.get("text", "")),
                    "bbox":    item.get("bbox", item.get("coordinates", [])),
                })
        elif isinstance(data, dict) and "elements" in data:
            for item in data["elements"]:
                elementos.append({
                    "type":    item.get("type", "text"),
                    "content": item.get("content", item.get("text", "")),
                    "bbox":    item.get("bbox", []),
                })
        return elementos

    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: texto plano, una línea = un elemento
    for linea in raw_output.split("\n"):
        linea = linea.strip()
        if linea:
            elementos.append({
                "type":    "text",
                "content": linea,
                "bbox":    [],
            })

    return elementos


# ─────────────────────────────────────────────
# ETAPA 3: SEPARACIÓN POR IDIOMA (Unicode)
# ─────────────────────────────────────────────

def es_caracter_indio(char: str) -> bool:
    cp = ord(char)
    return any(inicio <= cp <= fin for inicio, fin in INDIAN_SCRIPT_RANGES)


def limpiar_latin(texto: str) -> str | None:
    """Extrae solo la parte latina de un texto mixto."""
    limpio = "".join(c for c in texto if ord(c) < 0x0900)
    limpio = limpio.lstrip("/ ").strip()
    if not limpio or all(c in ".,;:/-_()[]{}\"' " for c in limpio):
        return None
    return limpio


def clasificar_texto(texto: str) -> str:
    """Clasifica texto como 'english', 'indian' o 'unknown'."""
    chars_alfa = [c for c in texto if c.isalpha()]
    if not chars_alfa:
        return "unknown"
    chars_indios = sum(1 for c in chars_alfa if es_caracter_indio(c))
    return "indian" if (chars_indios / len(chars_alfa)) > 0.20 else "english"


def separar_idiomas(elementos: list[dict]) -> dict:
    """
    Clasifica cada elemento por idioma.
    Para elementos mixtos extrae la parte latina y la agrega a 'english'.
    """
    grupos = {"english": [], "indian": [], "unknown": []}

    for elem in elementos:
        contenido = elem.get("content", "").strip()
        if not contenido:
            continue

        idioma = clasificar_texto(contenido)

        if idioma == "indian":
            limpio = limpiar_latin(contenido)
            if limpio:
                # Mixto: agregar versión limpia al grupo english
                grupos["english"].append({**elem, "content": limpio,
                                          "content_original": contenido,
                                          "fue_mixto": True})
            grupos["indian"].append({**elem, "idioma": "indian"})
        else:
            grupos[idioma].append({**elem, "idioma": idioma})

    return grupos


# ─────────────────────────────────────────────
# ETAPA 4: EXTRACCIÓN DE PARES CAMPO→VALOR
# ─────────────────────────────────────────────

# Separadores comunes entre campo y valor en documentos oficiales
_SEP = r"[\s]*[:/\-][\s]*"

# Palabras que NO son valores de nombre aunque aparezcan después de "Name:"
_PALABRAS_ENCABEZADO = {
    "INDIA", "GOVERNMENT", "REPUBLIC", "VISA", "NATIONAL", "CARD",
    "CERTIFICATE", "PASSPORT", "OF", "THE", "AND",
}

# Regex para detectar una fecha en cualquier formato
_PATRON_FECHA = (
    r"(\d{1,2}[\s]*[\/\-\.][\s]*\d{1,2}[\s]*[\/\-\.][\s]*\d{2,4}"
    r"|\d{1,2}\s+\w+\s*,?\s*\d{4})"    # ej: 19 June, 1970
)


def extraer_pares(elementos_english: list[dict]) -> dict:
    """
    Extrae pares campo→valor del texto estructurado en inglés.

    Estrategia 1: campo y valor en el mismo elemento separados por : o /
      "Full Name: SIDDHANT GUPTA"  →  name: "SIDDHANT GUPTA"

    Estrategia 2: elementos consecutivos donde el primero parece una etiqueta
      elemento N:   "DATE OF BIRTH"     (solo etiqueta, sin valor)
      elemento N+1: "20 / 09 / 1994"   (solo valor)
      →  date_of_birth: "20 / 09 / 1994"

    Retorna un dict con todos los pares encontrados, sin depender de
    un conjunto fijo de campos — cualquier par campo:valor es capturado.
    """
    pares = {}
    textos = [e["content"] for e in elementos_english if e.get("content", "").strip()]

    # ── Estrategia 1: campo:valor en la misma línea ──
    for texto in textos:
        # Buscar patron "etiqueta : valor" con separador : o /
        match = re.match(
            r"^([A-Za-z][A-Za-z\s\.\']{2,40}?)" + _SEP + r"(.{2,100})$",
            texto.strip()
        )
        if match:
            etiqueta = match.group(1).strip()
            valor    = match.group(2).strip()

            # Ignorar si la etiqueta es muy corta o el valor parece otra etiqueta
            if len(etiqueta) >= 2 and len(valor) >= 1:
                clave = _normalizar_clave(etiqueta)
                # Validar que el valor no sea solo palabras de encabezado
                if clave == "name" and _es_encabezado(valor):
                    continue
                if clave not in pares:
                    pares[clave] = valor

    # ── Estrategia 2: etiqueta en línea N, valor en línea N+1 ──
    for i in range(len(textos) - 1):
        etiqueta_candidata = textos[i].strip()
        valor_candidato    = textos[i + 1].strip()

        # La etiqueta no debe contener dígitos ni separadores de valor
        if re.search(r"\d", etiqueta_candidata):
            continue
        if re.search(r"[:/]", etiqueta_candidata):
            continue

        # El valor puede ser una fecha, número o texto
        clave = _normalizar_clave(etiqueta_candidata)
        if clave and clave not in pares:
            if clave == "name" and _es_encabezado(valor_candidato):
                continue
            pares[clave] = valor_candidato

    return pares


def _normalizar_clave(etiqueta: str) -> str:
    """
    Convierte una etiqueta de documento a una clave normalizada snake_case.
    Ejemplos:
        "Full Name"       → "full_name"
        "Date of Birth"   → "date_of_birth"
        "D.O.B."          → "dob"
        "Passport No"     → "passport_no"
    """
    clave = etiqueta.lower().strip()
    clave = re.sub(r"[^a-z0-9\s]", "", clave)   # quitar puntuación
    clave = re.sub(r"\s+", "_", clave)           # espacios → _
    clave = clave.strip("_")
    return clave if clave else ""


def _es_encabezado(valor: str) -> bool:
    """Retorna True si el valor parece un encabezado de documento, no un nombre."""
    palabras = {w.upper() for w in valor.split()}
    return palabras.issubset(_PALABRAS_ENCABEZADO)


# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────

def visualizar(imagen_pil: PILImage.Image, grupos: dict, ruta_salida: str):
    """
    Dibuja bboxes de los elementos detectados por Dolphin:
      Verde   = texto en inglés
      Naranja = script indio
      Gris    = desconocido
    """
    img_bgr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    VERDE   = ( 34, 197,  94)
    NARANJA = (  0, 165, 255)
    GRIS    = (160, 160, 160)
    colores = {"english": VERDE, "indian": NARANJA, "unknown": GRIS}

    for idioma, elementos in grupos.items():
        color = colores.get(idioma, GRIS)
        for elem in elementos:
            bbox = elem.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                texto_label = elem.get("content", "")[:30]
                cv2.putText(img_bgr, texto_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.imwrite(ruta_salida, img_bgr)
    print(f"  Visualización guardada en: {ruta_salida}")


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────

def procesar_imagen(ruta_imagen: str, model, processor, device: str,
                    guardar_viz: bool = True) -> dict:
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta_imagen}")

    print(f"\n{'='*50}")
    print(f"Procesando: {ruta.name}")
    print(f"{'='*50}")

    print("→ Etapa 1: Cargando imagen...")
    imagen = cargar_imagen(str(ruta))
    print(f"  Tamaño final: {imagen.size}")

    print("→ Etapa 2: Parseando con Dolphin...")
    elementos = parsear_con_dolphin(imagen, model, processor, device)
    print(f"  {len(elementos)} elementos detectados")

    print("→ Etapa 3: Separando por idioma...")
    grupos = separar_idiomas(elementos)
    print(f"  Inglés: {len(grupos['english'])} | "
          f"Indio: {len(grupos['indian'])} | "
          f"Desconocido: {len(grupos['unknown'])}")

    print("→ Etapa 4: Extrayendo pares campo→valor...")
    pares = extraer_pares(grupos["english"])
    print(f"  {len(pares)} pares encontrados: {list(pares.keys())}")

    if guardar_viz:
        print("→ Guardando visualización...")
        ruta_salida = str(ruta.parent / f"{ruta.stem}_dolphin.jpg")
        visualizar(imagen, grupos, ruta_salida)

    return {
        "ruta":       str(ruta),
        "elementos":  elementos,
        "resumen":    {k: len(v) for k, v in grupos.items()},
        "pares":      pares,
    }


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando Dolphin-v2...")
    model, processor, device = cargar_modelo(MODEL_PATH)

    resultado = procesar_imagen("public/originals/Indian_IDs_p1.png", model, processor, device)

    print("\n── PARES CAMPO→VALOR ENCONTRADOS ──")
    print(json.dumps(resultado["pares"], indent=2, ensure_ascii=False))

    print("\n── TODOS LOS ELEMENTOS DETECTADOS ──")
    for elem in resultado["elementos"]:
        idioma = elem.get("idioma", "?")
        tipo   = elem.get("type", "text")
        cont   = elem.get("content", "")
        print(f"  [{tipo:12}] [{idioma:8}] {cont[:60]}")