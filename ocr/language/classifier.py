from ..schema import TextLine


# ── Unicode ranges covering all major Indian scripts ──────────────────────────
_INDIAN_RANGES: list[tuple[int, int]] = [
    (0x0900, 0x097F),  # Devanagari  (Hindi, Sanskrit, Marathi…)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi   (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

# Three-tier thresholds (fraction of alphabetic chars that are Indian-script).
# Tune here without touching classify_lines logic.
_THRESH_ENGLISH = 0.10   # ratio below this → pure English
_THRESH_INDIAN  = 0.70   # ratio above this → pure Indian
# Between the two thresholds → mixed; both parts are extracted separately.


# ── Character-level helpers ───────────────────────────────────────────────────

def _is_indian_char(char: str) -> bool:
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _INDIAN_RANGES)


def _script_ratio(text: str) -> float | None:
    """Fraction of alphabetic chars that are Indian-script. None if no alpha chars."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return None
    return sum(_is_indian_char(c) for c in alpha) / len(alpha)


# ── Word-level extraction ─────────────────────────────────────────────────────

def _latin_words(text: str) -> str | None:
    """
    Return only the words that contain no Indian-script characters.
    Word-level (not char-level) so word boundaries are preserved.
    """
    kept = [
        word for word in text.split()
        if any(c.isalpha() for c in word)
        and not any(_is_indian_char(c) for c in word)
    ]
    return " ".join(kept) if kept else None


def _indian_words(text: str) -> str | None:
    """Return only the words that contain at least one Indian-script character."""
    kept = [word for word in text.split() if any(_is_indian_char(c) for c in word)]
    return " ".join(kept) if kept else None


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_lines(lines: list[TextLine]) -> dict[str, list[TextLine]]:
    """
    Split OCR lines into three buckets using a three-tier ratio test.

    pure English  (ratio < _THRESH_ENGLISH) → english
    pure Indian   (ratio > _THRESH_INDIAN)  → indian
    mixed                                   → Latin words → english
                                              Indian words → indian
    No alpha chars                          → unknown

    Mixed lines are split at word boundaries, so both buckets receive
    clean, script-homogeneous text instead of character-stripped fragments.
    """
    groups: dict[str, list[TextLine]] = {"english": [], "indian": [], "unknown": []}

    for line in lines:
        ratio = _script_ratio(line.text)

        if ratio is None or ratio < _THRESH_ENGLISH:
            line.lang = "english"
            groups["english"].append(line)

        elif ratio > _THRESH_INDIAN:
            # Mostly Indian — salvage any stray Latin words into english.
            latin = _latin_words(line.text)
            if latin:
                groups["english"].append(TextLine(
                    text=latin,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    lang="english",
                    original_text=line.text,
                    was_mixed=True,
                ))
            line.lang = "indian"
            groups["indian"].append(line)

        else:
            # Mixed — split by word so both buckets get clean text.
            latin  = _latin_words(line.text)
            indian = _indian_words(line.text)

            if latin:
                groups["english"].append(TextLine(
                    text=latin,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    lang="english",
                    original_text=line.text,
                    was_mixed=True,
                ))
            if indian:
                groups["indian"].append(TextLine(
                    text=indian,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    lang="indian",
                    original_text=line.text,
                    was_mixed=True,
                ))

    _log_mixed(groups)
    return groups


def _log_mixed(groups: dict[str, list[TextLine]]) -> None:
    mixed = [l for l in groups["english"] if l.was_mixed]
    if not mixed:
        return
    print("\n── RECOVERED MIXED LINES ──")
    for line in mixed:
        print(f"  original : {line.original_text}")
        print(f"  latin    : {line.text}")
