import re
from difflib import SequenceMatcher

from ..schema import TextLine
from .patterns import extract_value


# ── Tunable constants ──────────────────────────────────────────────────────────

_FUZZY_THRESHOLD   = 0.82  # min similarity to accept a fuzzy label match
_FUZZY_MIN_LEN     = 6     # skip fuzzy for short labels (too noisy below this)
_ROW_Y_TOL         = 20    # px: y-distance to consider two lines on the same row
_BELOW_GAP_MAX     = 80    # px: max vertical gap for "directly below"
_BELOW_X_TOL       = 80    # px: max horizontal offset for "directly below"
_LOOKAHEAD         = 3     # lines to scan below a label in text order


# ── Bbox helpers ───────────────────────────────────────────────────────────────
# Bbox from PaddleOCR is a 4-point polygon: [top-left, top-right, bottom-right, bottom-left]

def _top_y(l: TextLine) -> int:    return int(l.bbox[0][1])
def _bot_y(l: TextLine) -> int:    return int(l.bbox[2][1])
def _left_x(l: TextLine) -> int:   return int(l.bbox[0][0])
def _right_x(l: TextLine) -> int:  return int(l.bbox[1][0])

def _same_row(a: TextLine, b: TextLine) -> bool:
    return abs(_top_y(a) - _top_y(b)) <= _ROW_Y_TOL

def _is_right_of(label: TextLine, cand: TextLine) -> bool:
    return _left_x(cand) >= _right_x(label) - 10

def _is_below(label: TextLine, cand: TextLine) -> bool:
    gap = _top_y(cand) - _bot_y(label)
    return 0 <= gap <= _BELOW_GAP_MAX and abs(_left_x(label) - _left_x(cand)) <= _BELOW_X_TOL


def _spatial_neighbor(
    label_line: TextLine,
    all_lines: list[TextLine],
    label_idx: int,
) -> str | None:
    """
    Find the nearest value line to label_line using bounding-box position.
    Priority: same row to the right → directly below.
    """
    others = [(j, l) for j, l in enumerate(all_lines) if j != label_idx]

    inline = sorted(
        [(j, l) for j, l in others if _same_row(label_line, l) and _is_right_of(label_line, l)],
        key=lambda jl: _left_x(jl[1]),
    )
    if inline:
        return inline[0][1].text

    below = sorted(
        [(j, l) for j, l in others if _is_below(label_line, l)],
        key=lambda jl: _top_y(jl[1]),
    )
    if below:
        return below[0][1].text

    return None


# ── Label matching ─────────────────────────────────────────────────────────────

def _fuzzy_label_end(label: str, text: str) -> int | None:
    """
    Sliding-window similarity search for label inside text.
    Returns the end position of the best match, or None if below threshold.
    Skipped for short labels where fuzzy matching produces too many false positives.
    """
    if len(label) < _FUZZY_MIN_LEN:
        return None

    label_l = label.lower()
    text_l  = text.lower()
    n = len(label_l)

    best_ratio, best_end = 0.0, None
    for i in range(max(1, len(text_l) - n + 1)):
        ratio = SequenceMatcher(None, label_l, text_l[i : i + n]).ratio()
        if ratio > best_ratio:
            best_ratio, best_end = ratio, i + n

    return best_end if best_ratio >= _FUZZY_THRESHOLD else None


def _find_label(label: str, text: str) -> int | None:
    """
    Return the position immediately after label in text.
    Tries exact word-boundary match first; falls back to fuzzy for OCR errors.
    """
    m = re.search(r"\b" + re.escape(label) + r"\b", text, re.IGNORECASE)
    if m:
        return m.end()
    return _fuzzy_label_end(label, text)


# ── KIE Engine ─────────────────────────────────────────────────────────────────

class KIEEngine:
    """
    Label-first Key Information Extraction.
    Stateless after construction; one instance can process many images.

    Value search strategy (in priority order):
      1. Same line  — text to the right of the label on the same OCR line
      2. Spatial    — nearest line by bbox (right of label, then directly below)
      3. Lookahead  — next _LOOKAHEAD lines in reading order

    Label matching: exact word-boundary first, then fuzzy (difflib) for OCR errors.
    """

    _DATE_STANDALONE = re.compile(r"^(\d{1,2}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{2,4})$")

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    def detect_doc_type(self, texts: list[str]) -> str | None:
        """Score each document type by identifier hits; return the best match."""
        full = " ".join(texts).upper()
        scores = {
            doc: sum(1 for ident in data["identifiers"] if ident.upper() in full)
            for doc, data in self._mapping["document_types"].items()
        }
        best = {k: v for k, v in scores.items() if v > 0}
        return max(best, key=best.get) if best else None

    def extract(
        self,
        english_lines: list[TextLine],
        unknown_lines: list[TextLine],
    ) -> dict:
        doc_type = self.detect_doc_type([l.text for l in english_lines])

        if doc_type:
            fields = self._mapping["document_types"][doc_type]["fields"]
        else:
            # Fallback: merge all known fields; first definition wins per name.
            fields: dict = {}
            for dt in self._mapping["document_types"].values():
                for name, data in dt["fields"].items():
                    if name not in fields:
                        fields[name] = data

        entities: dict = {}

        for i, line in enumerate(english_lines):
            for field_name, field_data in fields.items():
                if field_name in entities:
                    continue
                kind = field_data["type"]

                for label in field_data["labels"]:
                    end_pos = _find_label(label, line.text)
                    if end_pos is None:
                        continue

                    # 1. Same-line value (text after the label)
                    rest = line.text[end_pos:].lstrip(":/-| \t").strip()
                    if rest:
                        value = extract_value(rest, kind)
                        if value:
                            entities[field_name] = value
                            break

                    # 2. Spatial neighbor (right of label or directly below, by bbox)
                    neighbor = _spatial_neighbor(line, english_lines, i)
                    if neighbor:
                        value = extract_value(neighbor, kind)
                        if value:
                            entities[field_name] = value
                            break

                    # 3. Text-order lookahead
                    for j in range(i + 1, min(i + 1 + _LOOKAHEAD, len(english_lines))):
                        value = extract_value(english_lines[j].text, kind)
                        if value:
                            entities[field_name] = value
                            break

                    if field_name in entities:
                        break

        self._fill_date_gaps(unknown_lines, entities)
        return {"document_type": doc_type, "fields": entities}

    def _fill_date_gaps(self, unknown_lines: list[TextLine], entities: dict) -> None:
        """Assign standalone date-only lines to unfilled date fields."""
        dates = [
            m.group(1).strip()
            for line in unknown_lines
            if (m := self._DATE_STANDALONE.match(line.text.strip()))
        ]
        slots = [f for f in ("issue_date", "expiry_date", "date_of_birth") if f not in entities]
        for field_name, date in zip(slots, dates):
            entities[field_name] = date
