import re


# Each pattern captures the actual value in group 1.
# Applied to text after the field label has been stripped.
_VALUE_PATTERNS: dict[str, re.Pattern] = {
    # Priority: most specific format first.
    # DD-MON-YYYY is common on Indian passports; ISO (YYYY-MM-DD) added for digital docs.
    "date": re.compile(
        r"(\b\d{1,2}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{4}\b"
        r"|\b\d{4}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{1,2}\b"
        r"|\b\d{1,2}[\-\s](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\-\s]\d{4}\b"
        r"|\b\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER),?\s+\d{4}\b)",
        re.IGNORECASE,
    ),
    # Ordered most-specific → least-specific to prevent early false matches.
    # PAN: ABCDE1234F | Aadhaar: 4-4-4 | Voter ID: 3L+7D | DL
    "document_number": re.compile(
        r"\b([A-Z]{5}\d{4}[A-Z]"
        r"|\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"
        r"|[A-Z]{3}\d{7}"
        r"|[A-Z]{2}[\-]?\d{11,13})\b",
        re.IGNORECASE,
    ),
    # India standard: 1 uppercase letter + 7 digits (A1234567).
    # USA/UK style:   1 letter + 8 digits (A12345678).
    # ICAO generic:   9-char alphanumeric (covers most countries).
    "passport_number": re.compile(
        r"\b([A-Z]\d{7}"
        r"|[A-Z]\d{8}"
        r"|[A-Z0-9]{9})\b",
    ),
    # Indian sticker visa: 2-3 uppercase letters + 7 digits (VJ1234567).
    # Indian e-Visa:       12-13 digit numeric string (no letters).
    "visa_number": re.compile(
        r"\b([A-Z]{2}\d{7}"
        r"|\d{12,13})\b",
    ),
    
    # Matches 1–5 words: covers surname-only fields and full names alike.
    # Minimum 2 chars per word to filter out stray initials.
    "name": re.compile(
        r"\b([A-Z][A-Za-z\-\']{1,29}(?:\s+[A-Z][A-Za-z\-\']{1,29}){0,4})\b",
        re.IGNORECASE,
    ),
    "gender": re.compile(
        r"\b(MALE|FEMALE|TRANSGENDER|TRANS|NON[\s\-]BINARY|OTHER|[MF])\b",
        re.IGNORECASE,
    ),
    "address": re.compile(
        r"(\d+[A-Za-z]?[\s,]+[A-Za-z][\w\s\.,\-\/\#\&\']{10,120}"
        r"|[A-Za-z][\w\s\.,\-\/\#\&\']{15,120})",
        re.IGNORECASE,
    ),
    # ABO+Rh: A+, B-, AB+, O- (letter forms), or O POS / AB NEGATIVE (word forms).
    # Single wrapping group so m.group(1) is always the full match value.
    "blood_group": re.compile(
        r"\b((?:A|B|AB|O)[+\-]|(?:A|B|AB|O)\s+(?:POS(?:ITIVE)?|NEG(?:ATIVE)?))\b",
        re.IGNORECASE,
    ),
    # STR allele pairs (forensic): D3S1358:15,16  or  D3S1358 15 16
    # Also catches compact numeric haplotype strings (10+ digits).
    "dna_marker": re.compile(
        r"([A-Z0-9]{3,10}[\s:]?\d{1,2}[,\s]\d{1,2}"
        r"|\b\d{10,}\b)",
        re.IGNORECASE,
    ),
    # All 28 states + 8 UTs of India (ISO 3166-2:IN two-letter codes).
    "state_code": re.compile(
        r"\b(AP|AR|AS|BR|CG|GA|GJ|HR|HP|JH|KA|KL|MP|MH|MN|ML|MZ|NL|OD|PB|RJ|SK|TN|TG|TR|UP|UK|WB"
        r"|AN|CH|DN|DD|DL|JK|LA|LD|PY)\b",
    ),
    # Aadhaar Virtual ID (VID): 16 digits, optionally grouped 4-4-4-4.
    "vid_number": re.compile(r"(\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b)"),
    "number": re.compile(r"(\b\d{4,12}\b)"),
    # MRZ lines are exactly 44 chars (TD3 passport/visa) or 30 chars (TD1 ID card).
    "mrz":    re.compile(r"([A-Z0-9<]{30,44})"),
    "text":   re.compile(r"([A-Za-z][A-Za-z\s\.\-\,]{1,50})"),
}


def extract_value(text: str, kind: str) -> str | None:
    pattern = _VALUE_PATTERNS.get(kind, _VALUE_PATTERNS["text"])
    m = pattern.search(text)
    if not m:
        return None
    # Use the first non-None group; fall back to the full match if all groups are None.
    value = next((g for g in m.groups() if g is not None), m.group(0))
    return value.strip() if value else None
