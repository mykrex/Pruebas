"""
Post-OCR filter for bilingual English/Hindi documents using EasyOCR.

Why EasyOCR instead of DocTR:
    DocTR only supports Latin vocabularies — it has no pretrained models
    for Devanagari script. When it encounters Hindi text it guesses Latin
    characters, producing garbled output that is hard to filter reliably.

    EasyOCR has pretrained models for both English ('en') and Hindi ('hi').
    It recognizes both scripts correctly in the same pass, so filtering
    is straightforward: we simply check whether the recognized text
    contains Latin or Devanagari characters using Unicode ranges.
    No pixel analysis, no K-Means, no shirorekha detection needed.

How it works — two filters:
    Filter 1 - Confidence : drops words EasyOCR was not confident about
    Filter 2 - Unicode    : drops words that are predominantly Devanagari,
                            keeping only English text for the analysis agent

Installation:
    pip install easyocr opencv-python pdf2image

    EasyOCR will automatically download the pretrained models for 'en'
    and 'hi' on the first run (~200MB total). An internet connection is
    required the first time only.

    For PDFs also install poppler:
        Windows : https://github.com/oschwartz10612/poppler-windows/releases
        Linux   : sudo apt install poppler-utils
        macOS   : brew install poppler

Usage:
    python detector.py --image visa.png
    python detector.py --image PANcard.png --debug
    python detector.py --image visa.pdf --output results/visa_result.jpg
"""

import argparse
import sys
import os
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Missing OpenCV.  pip install opencv-python")

try:
    import easyocr
except ImportError:
    sys.exit("Missing EasyOCR. pip install easyocr")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False


# ── parameters ────────────────────────────────────────────────────────────────
#
# CONFIDENCE_THRESHOLD:
#   EasyOCR returns a confidence score from 0.0 to 1.0 per word.
#   Words below this threshold are dropped regardless of script.
#   0.35 is intentionally permissive to avoid losing valid English
#   text from low-quality scans.
#
# LATIN_THRESHOLD:
#   Minimum fraction of characters that must fall within the Latin
#   Unicode range (below U+0900) for a word to be kept.
#   0.80 means at least 80% of characters must be Latin.

CONFIDENCE_THRESHOLD = 0.20
LATIN_THRESHOLD      = 0.30


# ── image loading ─────────────────────────────────────────────────────────────

def load_image(path: str) -> list[np.ndarray]:
    """
    Returns a list of BGR images, one per page if the input is a PDF.
    For regular images returns a single-element list.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if not PDF2IMAGE_OK:
            sys.exit("For PDFs: pip install pdf2image (and poppler)")
        pages = convert_from_path(path, dpi=300)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not load image: {path}")
    return [img]

def clean_latin_only(text: str) -> str | None:
    """
    Instead of discarding the whole word/block when it contains Devanagari,
    strips out individual Devanagari characters and returns only the Latin
    portion of the text.

    If the resulting clean text is empty or only punctuation, returns None
    so the block gets discarded entirely.

    Examples:
        "विभाग DEPARTMENT" → "DEPARTMENT"
        "INDIA"            → "INDIA"
        "भारत"             → None  (nothing Latin left)
        "New विभाग Delhi"  → "New  Delhi"
    """
    # keep only characters below U+0900 (Latin, digits, punctuation, spaces)
    cleaned = ''.join(c for c in text if ord(c) < 0x0900)

    # strip leftover whitespace and punctuation-only results
    cleaned = cleaned.strip()
    if not cleaned or all(c in '.,;:/-_()[]{}"\' ' for c in cleaned):
        return None

    return cleaned


# ── the two filters ───────────────────────────────────────────────────────────

def filter_1_confidence(confidence: float) -> str | None:
    """
    Filter 1 — EasyOCR confidence score.

    EasyOCR assigns a score from 0.0 to 1.0 to each recognized word.
    Low scores indicate the model was uncertain — this catches noise,
    blurry regions, and partially visible text regardless of script.

    Returns the discard reason as a string, or None if the word passes.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return f"low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})"
    return None


def filter_2_unicode(text: str) -> str | None:
    """
    Filter 2 — Unicode range check on the recognized text.

    Since EasyOCR correctly recognizes both English and Hindi, the OCR
    output actually contains the real characters of each script.
    We can now filter by Unicode range directly on meaningful text
    instead of trying to detect misrecognized Devanagari in Latin output.

    The Devanagari Unicode block occupies U+0900 to U+097F.
    Everything before U+0900 is Latin letters, Arabic digits (0-9),
    standard punctuation, and spaces — what we want to keep.

    We count how many characters fall before U+0900 and calculate the
    fraction over the total. If at least 80% are Latin, the word is kept.
    If the word is predominantly Devanagari, it is discarded.

    Examples:
        "Kumar"      → 5/5 Latin   → 100% → keep
        "विभाग"      → 0/5 Latin   →   0% → discard
        "INDIA"      → 5/5 Latin   → 100% → keep
        "भारत"       → 0/4 Latin   →   0% → discard
        "New विभाग"  → 3/9 Latin   →  33% → discard

    Returns the discard reason as a string, or None if the word passes.
    """
    if not text.strip():
        return "empty text"

    latin_chars = sum(1 for c in text if ord(c) < 0x0900)
    fraction    = latin_chars / len(text)

    if fraction < LATIN_THRESHOLD:
        return f"Devanagari text ({fraction:.0%} Latin < {LATIN_THRESHOLD:.0%})"
    return None


# ── EasyOCR pipeline ──────────────────────────────────────────────────────────

def run_easyocr(image_bgr: np.ndarray,
                reader: easyocr.Reader,
                debug: bool = False) -> list[dict]:
    """
    Runs EasyOCR on a single image and returns a structured list of words.

    EasyOCR returns results as a list of tuples:
        (bounding_box, text, confidence)

    Where bounding_box is a list of four corner points:
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    We convert this to a normalized geometry format consistent with
    what the rest of the pipeline expects:
        ((x1_norm, y1_norm), (x2_norm, y2_norm))

    EasyOCR recognizes both English and Hindi in the same pass because
    we initialized the reader with ['en', 'hi']. The model automatically
    determines which script each region belongs to and applies the
    appropriate recognizer.
    """
    h, w = image_bgr.shape[:2]
    raw  = reader.readtext(image_bgr)

    words = []
    for (bbox, text, confidence) in raw:
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # we use the top-left and bottom-right corners
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, x2   = min(x_coords), max(x_coords)
        y1, y2   = min(y_coords), max(y_coords)

        # normalize to 0.0-1.0 range
        geometry = (
            (round(x1 / w, 4), round(y1 / h, 4)),
            (round(x2 / w, 4), round(y2 / h, 4))
        )

        words.append({
            "text"      : text,
            "confidence": round(float(confidence), 3),
            "geometry"  : geometry
        })

    if debug:
        print(f"EasyOCR detected {len(words)} words total\n")

    return words


# ── filtering pipeline ────────────────────────────────────────────────────────

def filter_words(words, debug=False):
    kept      = []
    discarded = []

    for w in words:
        # filter 1: confidence — discard the whole block if too uncertain
        if w["confidence"] < CONFIDENCE_THRESHOLD:
            discarded.append({**w, "reason": f"low confidence ({w['confidence']:.2f})"})
            if debug:
                print(f"  DISCARDED — low confidence: '{w['text']}'")
            continue

        # filter 2: instead of discarding, clean out Devanagari characters
        cleaned = clean_latin_only(w["text"])

        if cleaned is None:
            # nothing Latin left after cleaning → discard the whole block
            discarded.append({**w, "reason": "entirely Devanagari"})
            if debug:
                print(f"  DISCARDED — entirely Devanagari: '{w['text']}'")
        elif cleaned != w["text"]:
            # block was mixed — keep only the Latin portion
            kept.append({**w, "text": cleaned, "original": w["text"]})
            if debug:
                print(f"  CLEANED   — '{w['text']}' → '{cleaned}'")
        else:
            # block was already fully Latin
            kept.append(w)
            if debug:
                print(f"  KEPT      — conf={w['confidence']:.2f}: '{w['text']}'")

    return kept, discarded


# ── main pipeline ─────────────────────────────────────────────────────────────

def process(image_path: str, debug: bool = False):
    """
    Full pipeline:
        1. EasyOCR reader is initialized with English and Hindi models.
           The models are downloaded automatically on first run (~200MB).
        2. The image is loaded with OpenCV.
        3. EasyOCR runs detection and recognition for both scripts.
        4. Filters 1 and 2 are applied to keep only English text.
    """
    print("Loading EasyOCR models (downloads ~200MB on first run)...")
    reader = easyocr.Reader(
        ['en', 'hi'],
        gpu=False   # set to True if you have a CUDA-compatible GPU
    )

    images = load_image(image_path)
    image_bgr = images[0]  # process first page

    print("Running EasyOCR...")
    words = run_easyocr(image_bgr, reader, debug)

    kept, discarded = filter_words(words, debug)
    return kept, discarded, image_bgr


# ── visualization ─────────────────────────────────────────────────────────────

def visualize(image_bgr: np.ndarray,
              kept: list,
              discarded: list,
              output_path: str = "result.jpg"):
    """
    Draws colored bounding boxes on the original image:
        Green  = English word kept (passed both filters)
        Red    = discarded by filter 1 (low confidence)
        Orange = discarded by filter 2 (Devanagari / Hindi text)
    """
    vis  = image_bgr.copy()
    h, w = vis.shape[:2]
    GREEN  = ( 34, 197,  94)
    RED    = ( 59,  52, 220)
    ORANGE = (  0, 165, 255)

    for word in kept:
        (x1, y1), (x2, y2) = word["geometry"]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, pt1, pt2, GREEN, 2)
        cv2.putText(vis, word["text"], (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)

    for word in discarded:
        (x1, y1), (x2, y2) = word["geometry"]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        color = RED if "confidence" in word["reason"] else ORANGE
        cv2.rectangle(vis, pt1, pt2, color, 2)

    cv2.imwrite(output_path, vis)
    print(f"\nOutput image saved to: {output_path}")
    print("  Green  = English text kept")
    print("  Red    = discarded by filter 1 (low confidence)")
    print("  Orange = discarded by filter 2 (Devanagari / Hindi)")


# ── report ────────────────────────────────────────────────────────────────────

def print_report(kept: list, discarded: list):
    total         = len(kept) + len(discarded)
    by_confidence = [w for w in discarded if "confidence" in w["reason"]]
    by_unicode    = [w for w in discarded if "Devanagari" in w["reason"]]

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"Total words detected by EasyOCR : {total}")
    print(f"Kept (English)                  : {len(kept)}")
    print(f"Discarded total                 : {len(discarded)}")
    print(f"  by filter 1 (confidence)      : {len(by_confidence)}")
    print(f"  by filter 2 (Devanagari)      : {len(by_unicode)}")

    print("\n── KEPT ENGLISH TEXT ────────────────────────────────────")
    for w in kept:
        print(f"  '{w['text']:<25}' conf={w['confidence']:.2f}")

    print("\n── DISCARDED ────────────────────────────────────────────")
    for w in discarded:
        print(f"  '{w['text']:<25}' reason: {w['reason']}")

    print("\n── RECONSTRUCTED TEXT ───────────────────────────────────")
    print(f"  {' '.join(w['text'] for w in kept)}")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filters Hindi/Devanagari from EasyOCR output, "
                    "keeping only English text for document validation."
    )
    parser.add_argument("--image",  required=True,
                        help="Path to the image or PDF file")
    parser.add_argument("--debug",  action="store_true",
                        help="Print filter result for every word")
    parser.add_argument("--output", default="result.jpg",
                        help="Path for the output image with colored bounding boxes")
    args = parser.parse_args()

    kept, discarded, image_bgr = process(args.image, args.debug)
    print_report(kept, discarded)
    visualize(image_bgr, kept, discarded, args.output)


if __name__ == "__main__":
    main()