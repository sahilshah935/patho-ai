"""Heuristic / regex text extractor for pathology report fields.

This module operates purely on plain text (from digital PDF extraction or
OCR postprocessing).  It does not require any ML model — all logic is
rule-based and deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pathology_ai.extraction.schema import ExtractedPatient, ExtractedSection, ExtractedTest
from pathology_ai.extraction.normalizer import (
    normalize_test_name,
    normalize_unit,
    parse_reference_range,
)

# ---------------------------------------------------------------------------
# Category keywords
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Hematology": [
        "haematology", "hematology", "hemoglobin", "haemoglobin",
        "complete blood count", "cbc", "haemogram", "leuco",
        "platelet", "rbc", "wbc", "esr",
    ],
    "Biochemistry": [
        "biochemistry", "liver function", "lft", "kidney function", "kft",
        "renal function", "lipid profile", "cholesterol", "creatinine",
        "bilirubin", "sgpt", "sgot", "albumin", "urea", "glucose",
        "electrolyte",
    ],
    "Urinalysis": [
        "urine", "urinalysis", "urine examination", "microscopy",
        "specific gravity", "pus cells",
    ],
    "Endocrinology": [
        "thyroid", "tsh", "t3", "t4", "endocrinology", "hormone",
        "vitamin d", "vitamin b12", "insulin", "diabetes profile",
    ],
    "Serology / Immunology": [
        "serology", "immunology", "crp", "c-reactive", "rheumatoid",
        "aso", "ferritin", "infectious", "autoimmune",
    ],
}

# ---------------------------------------------------------------------------
# Patient info patterns
# ---------------------------------------------------------------------------

_PATIENT_PATTERNS: List[Tuple[str, str, str]] = [
    # (field, regex_pattern, group_name)
    ("name",               r"(?<!report\s)(?<!rpt\s)(?<!doctor\s)(?<!dr\s)(?:patient\s*name|pt\.?\s*name|client\s*name|customer\s*name|full\s*name|name)\s*(?:/\s*age)?\s*(?:/\s*(?:sex|gender))?\s*[:/\-]?\s*(?:mrs?\.?|ms\.?|smt\.?|dr\.?|master|baby)?\s*(?P<v>[A-Za-z][A-Za-z .'-]{2,40}?)(?=\s*(?:/|Date|Patient\s*ID|Report\s*ID|Report\s*No|Lab\s*accession|Lab|Accession|Age|Sex|Gender|Sample|UHID|Ref|\n|$))", "v"),
    ("patient_id",         r"(?:patient\s*id|uhid|patient\s*no\.?|pid|lab\s*accession)\s*[:/\-]?\s*(?P<v>(?:UHID[-\s]?)?[A-Za-z0-9\-]{4,30})", "v"),
    ("accession_number",   r"(?:accession\s*(?:no\.?|number)|acc\.?\s*no\.?|lab\s*accession|report\s*no\.?)\s*[:/\-]?\s*(?P<v>[A-Za-z0-9/\-]{4,30})", "v"),
    ("sample_id",          r"(?:sample\s*(?:id|no\.?)|specimen\s*id)\s*[:/\-]?\s*(?P<v>(?:EDTA|SMP|SPEC)[-\w\d]+|[A-Za-z0-9\-]{4,30})", "v"),
    ("referred_by",        r"(?:referred\s*by|ordering\s*clinician|ref(?:erring)?\s*(?:doctor|physician|dr\.?))\s*[:/\-]?\s*(?P<v>(?:Dr\.?\s+)?[A-Za-z .,'-]{3,50}?)(?=\s*(?:Location|Date|Sample|Accession|Reported|\n|$))", "v"),
    ("sample_type",        r"(?:sample|specimen)\s*(?:type|collected?)\s*[:/\-]?\s*(?P<v>[A-Za-z/\(\) ]{3,40}?)(?=\s*(?:Collection|Report|\n|$))", "v"),
    ("collection_datetime",r"(?:sample\s*collected|collected|collection\s*(?:date|time|datetime?))\s*[:/\-]?\s*(?P<v>\d{1,2}[/\-\.][A-Za-z0-9/\-\., ]{4,25})", "v"),
    ("received_datetime",  r"(?:received|receipt)\s*(?:date|time)?\s*[:/\-]?\s*(?P<v>\d{1,2}[/\-\.][A-Za-z0-9/\-\., ]{4,25})", "v"),
    ("report_datetime",    r"(?:report\s*(?:date|time|datetime?|generated)|reported)\s*[:/\-]?\s*(?P<v>\d{1,2}[/\-\.][A-Za-z0-9/\-\., ]{4,25})", "v"),
]

_AGE_SEX_PATTERN = re.compile(
    r"(?:age(?:\s*/\s*(?:gender|sex))?\s*[:/\-]?\s*\n?\s*(?P<age1>\b(?:1[0-1]\d|[1-9]\d?)\b)\s*(?:years?|yrs?|y/o)?\s*[/|,]?\s*(?P<sex1>female|male|f|m)?)"
    r"|(?:(?P<age2>\b(?:1[0-1]\d|[1-9]\d)\b)\s*(?:years?|yrs?|y/o)\s*[/|,]?\s*(?P<sex2>female|male|f|m)?)"
    r"|(?:(?:gender|sex)\s*[:/\-]?\s*(?P<sex3>female|male|f|m))",
    re.IGNORECASE,
)

_FLAG_RE = re.compile(
    r"^(?:[HLhl]\*?|CRITICAL|HIGH|LOW|NORMAL|ABNORMAL|BORDERLINE|ALERT|POSITIVE|NEGATIVE|BL|BH|CRIT|-)$",
    re.IGNORECASE,
)

_FLAG_WORDS = {
    "high", "low", "normal", "abnormal", "borderline", "critical",
    "alert", "positive", "negative", "bl", "bh", "crit", "h", "l"
}

def _clean_unit_and_flag(unit: str, flag: str) -> Tuple[str, str]:
    """Clean unit and flag, preventing flag terms from contaminating unit."""
    unit_tokens = []
    extracted_flag = flag.strip("-*").strip() if flag else ""
    for tok in (unit or "").split():
        tok_clean = tok.strip(".,;:*()[]").lower()
        if tok_clean in _FLAG_WORDS or _FLAG_RE.match(tok):
            if not extracted_flag:
                extracted_flag = tok.strip(".,;:*()[]-")
        else:
            unit_tokens.append(tok)
    clean_unit = " ".join(unit_tokens).strip(".,;:*()[]-")
    return clean_unit, extracted_flag


# ---------------------------------------------------------------------------
# Test row detection
# ---------------------------------------------------------------------------

# Typical table columns: TestName | Value | Unit | RefRange | Flag
# Accept both tab-separated and multi-space-separated rows.
_TEST_ROW_PATTERN = re.compile(
    r"^(?P<name>.{3,60?}?)"           # test name (non-greedy, 3–60 chars)
    r"[\t ]{1,6}"
    r"(?P<value>-?(?:\d+\.?\d*|[A-Za-z+]{1,15}))"  # numeric or qualitative
    r"(?:[\t ]{1,6}(?P<unit>[µ%\w/\.]{0,15}))?"    # optional unit
    r"(?:[\t ]{1,6}(?P<refrange>[\d\.\s\-–<>≤≥]+))?"  # optional ref range
    r"(?:[\t ]{1,6}(?P<flag>[HLhl*]{1,3}))?",       # optional flag
    re.MULTILINE,
)

# Section headers: all-caps lines of 4+ chars, optionally followed by colon
_SECTION_HEADER = re.compile(r"^([A-Z][A-Z\s\(\)/&\-]{3,60}?)(?:\s*:)?\s*$", re.MULTILINE)

# Remarks / comments blocks
_REMARKS_KEYWORDS = re.compile(
    r"(?:laboratory\s*remarks?|remarks?|notes?|comments?)\s*[:\-]?\s*\n"
    r"(?P<body>(?:.+\n?){1,10})",
    re.IGNORECASE,
)

# Laboratory name is typically the first non-empty line in the report
_LAB_NAME_LINES = 3  # check the first N lines

# Panel / report title patterns
_PANEL_PATTERNS = re.compile(
    r"(?:^|\n)(?P<panel>[A-Z][A-Z0-9 \(\)&/\-]{5,60}?)\s*\n",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_document_category(text: str) -> str:
    """Detect the report category from keyword presence in the text."""
    lower = text.lower()
    scores: Dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in lower)
    if not any(scores.values()):
        return ""
    return max(scores, key=lambda c: scores[c])


def extract_laboratory_name(text: str) -> str:
    """Best-effort extraction of the laboratory name (skipping page tags and generic headers)."""
    skip_keywords = (
        "[page", "page 1", "page 2", "page 3", "page 4", "page 5",
        "laboratory report", "investigation report", "pathology report",
        "diagnostic report", "confidential report", "medical report",
        "report no", "report date", "patient report", "end of report"
    )
    for line in text.splitlines():
        stripped = line.strip().strip("`").strip()
        if not stripped or len(stripped) <= 3:
            continue
        low = stripped.lower()
        if any(low.startswith(sk) or low == sk for sk in skip_keywords):
            continue
        # Skip pure contact, address, URL, barcode, markdown, or metadata lines
        if any(kw in low for kw in ["tel:", "phone:", "email:", "www.", "http:", "iso 15189", "nabl accredited", "qr code", "barcode", "```"]):
            continue
        return stripped
    return ""


def extract_panel_name(text: str) -> str:
    """Extract the panel/report title (typically an ALL-CAPS heading near the top)."""
    # Look for known panel keywords first
    known_panels = [
        "COMPLETE BLOOD COUNT", "CBC", "HAEMATOLOGY PROFILE", "HAEMOGRAM",
        "LIVER FUNCTION", "KIDNEY FUNCTION", "RENAL FUNCTION", "LIPID PROFILE",
        "BIOCHEMISTRY PROFILE", "BASIC METABOLIC",
        "COMPLETE URINE", "URINE ROUTINE", "URINALYSIS",
        "THYROID PROFILE", "THYROID FUNCTION", "DIABETES PROFILE",
        "HORMONE PROFILE", "ENDOCRINOLOGY PROFILE",
        "SEROLOGY PROFILE", "IMMUNOLOGY PROFILE", "INFECTIOUS DISEASE",
        "AUTOIMMUNE SCREENING",
    ]
    upper = text.upper()
    for panel in known_panels:
        if panel in upper:
            # Find the full line containing this keyword
            for line in text.splitlines():
                if panel in line.upper():
                    return line.strip()
    return ""


def _clean_human_name(raw: str) -> str:
    if not raw:
        return ""
    # Strip any trailing '/ Age / Sex' or '/ 34 Yrs' or 'Age:' tokens
    cleaned = re.sub(r"(?:/|Age|Sex|Gender|UHID|Date|Ref|Acc|Lab|SMP).*", "", raw, flags=re.IGNORECASE).strip()
    # Strip common title prefixes (Mr., Mrs., Ms., Smt., Dr., Master, Baby)
    cleaned = re.sub(r"^(?:mrs?\.?|ms\.?|smt\.?|dr\.?|master|baby)\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip(",;:- ")


def _is_valid_human_name(name: str) -> bool:
    cleaned = _clean_human_name(name)
    if not cleaned or len(cleaned) < 3:
        return False
    # Reject if it contains digits, slashes, backslashes, or colons
    if any(c.isdigit() for c in cleaned) or "/" in cleaned or "\\" in cleaned or ":" in cleaned:
        return False
    # Reject if any word matches ID / registration / header keywords
    words = [w.strip(".,:-()[]|").lower() for w in cleaned.split()]
    junk_words = {
        "uhid", "acc", "accession", "reg", "registration", "smp", "sample", "specimen",
        "bill", "ref", "no", "number", "id", "pid", "lab", "hospital", "hosp", "mrn",
        "result", "unit", "biological", "reference", "interval", "investigation", "parameter",
        "test", "status", "range", "flag", "observation", "findings", "laboratory",
        "report", "rpt", "email", "phone", "tel", "fax", "contact", "doctor", "physician",
        "dr", "clinic", "diagnostics", "group", "street", "road", "suite", "page", "name",
        "patient", "full", "male", "female", "yrs", "years", "adp", "hm", "dept", "department", "gender"
    }
    if any(w in junk_words for w in words):
        return False
    # Must contain at least one valid alphabetic name word
    alpha_words = [w for w in words if w.isalpha() and len(w) >= 2]
    if not alpha_words:
        return False
    return True


def extract_patient_info(text: str) -> ExtractedPatient:
    """Extract patient demographic and sample information from report text."""
    patient = ExtractedPatient()

    for field, pattern, group in _PATIENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(group).strip().rstrip(",;:")
            cand_clean = _clean_human_name(value) if field == "name" else value
            setattr(patient, field, cand_clean)

    # Validate extracted patient name against strict human name rules
    if patient.name and not _is_valid_human_name(patient.name):
        patient.name = ""

    # Spatial fallback for Patient Name: scan lines near 'Patient Name', 'Pt. Name', 'Name / Age' or 'Name:' labels
    if not patient.name:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            line_low = line.lower()
            if any(k in line_low for k in ["patient name", "pt. name", "pt name", "customer name", "client name", "name /", "name:"]):
                # Extract value after colon if present
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        cand = parts[1].strip()
                        cand_clean = _clean_human_name(cand)
                        if _is_valid_human_name(cand_clean):
                            patient.name = cand_clean
                            break

                # Check adjacent lines (strip leading/trailing ID tokens like EDTA-8042816)
                for offset in (1, -1, 2):
                    if 0 <= idx + offset < len(lines):
                        target_line = lines[idx + offset]
                        # Strip leading sample/accession/UHID codes like EDTA-8042816 or ADP/HM/123
                        cand_raw = re.sub(r"^(?:EDTA|UHID|SMP|ACC|ADP|REG|PID|ID|REF)[-\w\d/]+\s+", "", target_line, flags=re.IGNORECASE).strip()
                        cand_clean = _clean_human_name(cand_raw)
                        if _is_valid_human_name(cand_clean):
                            patient.name = cand_clean
                            break
                if patient.name:
                    break

    # Validate and filter IDs to prevent header tokens from being treated as values
    _JUNK_IDS = {
        "patient", "name", "uhid", "sample", "accession", "report", "date",
        "dr", "male", "female", "pid", "id", "ref", "specimen", "reg", "registration",
        "condition", "department", "general", "acceptable", "accredited", "nabl"
    }
    if patient.patient_id and patient.patient_id.lower() in _JUNK_IDS:
        patient.patient_id = ""
    if patient.sample_id and patient.sample_id.lower() in _JUNK_IDS:
        patient.sample_id = ""
    if patient.accession_number and (patient.accession_number.lower() in _JUNK_IDS or "accredit" in patient.accession_number.lower()):
        patient.accession_number = ""

    # Fallback pattern matchers for Indian pathology ID formats:
    if not patient.patient_id:
        m_uhid = re.search(r"\b(UHID[-_\s]?[A-Za-z0-9\-]{5,20}|PID[-_\s]?[A-Za-z0-9\-]{5,20})\b", text, re.IGNORECASE)
        if m_uhid:
            patient.patient_id = m_uhid.group(1).strip()

    if not patient.sample_id:
        m_smp = re.search(r"\b(EDTA[-_\s]?[A-Za-z0-9]{5,20}|SMP[-_\s]?[A-Za-z0-9]{5,20}|SPEC[-_\s]?[A-Za-z0-9]{5,20})\b", text, re.IGNORECASE)
        if m_smp:
            patient.sample_id = m_smp.group(1).strip()

    if not patient.accession_number:
        m_acc = re.search(r"\b(ACC[-_\s]?[0-9][A-Za-z0-9\-]{3,19}|ADP/[A-Z0-9/\-_]+)\b", text, re.IGNORECASE)
        if m_acc:
            cand_acc = m_acc.group(1).strip()
            if "accredit" not in cand_acc.lower() and cand_acc.lower() not in _JUNK_IDS:
                patient.accession_number = cand_acc

    # Clean referred_by from trailing department / location tokens
    if patient.referred_by:
        patient.referred_by = re.sub(
            r"(?:Department|Sample|Location|Condition|Accession|Registration|Date).*",
            "", patient.referred_by, flags=re.IGNORECASE
        ).strip().rstrip(",;:- ")
        if len(patient.referred_by.split()) < 2 and not patient.referred_by.lower().startswith("dr"):
            patient.referred_by = ""

    if not patient.referred_by:
        m_dr = re.search(r"\b(Dr\.?\s+[A-Za-z .,'-]{3,40}(?:,\s*(?:MD|MBBS|DNB)[A-Za-z0-9 ()\-_/]*)?)\b", text)
        if m_dr:
            cand_dr = m_dr.group(1).strip()
            if not patient.name or cand_dr.lower() not in patient.name.lower():
                patient.referred_by = cand_dr

    # Age / Sex (smart candidate selector prioritizing explicit units over bare numbers like 1)
    age_candidates = []
    for m in _AGE_SEX_PATTERN.finditer(text):
        age_raw = m.group("age1") or m.group("age2")
        sex_raw = m.group("sex1") or m.group("sex2") or m.group("sex3")
        if age_raw:
            try:
                val = int(age_raw)
                has_unit = any(u in m.group(0).lower() for u in ["year", "yr", "y/o"])
                if 0 < val <= 120:
                    age_candidates.append((val, has_unit))
            except ValueError:
                pass
        if sex_raw and not patient.sex:
            s = sex_raw.strip().lower()
            if s in ("m", "male"):
                patient.sex = "Male"
            elif s in ("f", "female"):
                patient.sex = "Female"

    if age_candidates:
        with_unit = [v for v, u in age_candidates if u]
        if with_unit:
            patient.age = with_unit[0]
        else:
            valid_bare = [v for v, u in age_candidates if v != 1]
            if valid_bare:
                patient.age = valid_bare[0]
            else:
                patient.age = age_candidates[0][0]

    return patient


def extract_sections(text: str) -> List[str]:
    """Return list of section-header strings detected in the text."""
    headers = []
    for m in _SECTION_HEADER.finditer(text):
        header = m.group(1).strip()
        # Skip very short all-cap words that aren't real section headers
        if len(header.split()) >= 1 and len(header) >= 4:
            headers.append(header)
    return headers


def extract_remarks(text: str) -> List[str]:
    """Extract laboratory remarks/comments from the report text."""
    remarks = []
    m = _REMARKS_KEYWORDS.search(text)
    if m:
        body = m.group("body")
        for line in body.splitlines():
            stripped = line.strip().strip("-•·*").strip()
            if stripped and len(stripped) > 4:
                remarks.append(stripped)
    return remarks


def _parse_value(raw: str) -> Optional[float]:
    """Attempt to convert a raw value string to a float."""
    if raw is None:
        return None
    cleaned = raw.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_test_rows(text: str) -> List[Dict[str, Any]]:
    """Extract raw test rows (name / value / unit / ref-range / flag) from text.

    The extractor looks for lines that match the typical layout of a
    pathology report result table.  Both tab-separated and space-aligned
    tables are supported.

    Returns:
        List of dicts with keys: ``name``, ``value``, ``unit``,
        ``reference_range``, ``flag``.
    """
    rows: List[Dict[str, Any]] = []

    # Known table header/footer words that indicate a non-data row
    _HEADER_WORDS = {
        "investigation", "test", "result", "unit", "reference", "flag",
        "biological", "interval", "range", "normal", "observation",
        "parameter", "analyte", "laboratory", "report", "page",
        "processed", "verified", "authorized", "technologist",
    }

    def _is_header_row(name: str, value: str) -> bool:
        """Return True if the row looks like a column header rather than data."""
        name_lower = name.lower().strip()
        value_lower = value.lower().strip()
        # Any of these combos → header row
        if value_lower in ("result", "value", "observation", "findings"):
            return True
        if name_lower in _HEADER_WORDS or value_lower in _HEADER_WORDS:
            return True
        # Multi-word names containing header keywords
        name_words = set(name_lower.split())
        if len(name_words & _HEADER_WORDS) >= 1 and not any(c.isdigit() for c in value):
            return True
        return False

    _VAL_RE = re.compile(r"^-?\d+\.?\d*$|^(?:Nil|Negative|Positive|Trace|\+{1,3})$", re.IGNORECASE)
    _FLAG_RE = re.compile(r"^(?:[HLhl]\*?|CRITICAL|-)$")
    _HYPHEN_REF = re.compile(r"(?:\d+\.?\d*\s*[-–—to]+\s*\d+\.?\d*|[<≤>≥]\s*\d+\.?\d*|up\s*to\s*\d+\.?\d*)", re.IGNORECASE)
    _SPACE_REF = re.compile(r"(?:\b\d+\.\d+\s+\d+\.\d+\b|\b\d+\s+\d+\b)", re.IGNORECASE)

    all_lines = text.splitlines()
    in_remarks_section = False

    for line in all_lines:
        line_clean = line.strip()
        line_low = line_clean.lower()
        if not line_clean or len(line_clean.split()) < 2:
            continue

        # Skip header/footer contact, address, email, phone, website lines
        if any(kw in line_low for kw in [
            "email", "tel:", "phone", "fax", "contact", "@", "www.", "http",
            "suite", "road", "street", "building"
        ]):
            continue

        # Stop processing rows if we enter remarks / clinical interpretation footer block
        if any(h in line_low for h in ["laboratory remarks", "clinical interpretation", "analyst:", "authorized signatory"]):
            in_remarks_section = True

        if in_remarks_section:
            continue

        tokens = line_clean.split()

        # Find index of first numeric or qualitative value token
        val_idx = -1
        for idx, tok in enumerate(tokens):
            if idx > 0 and _VAL_RE.match(tok):
                val_idx = idx
                break

        if val_idx == -1:
            continue

        name = " ".join(tokens[:val_idx]).strip()
        value = tokens[val_idx].strip()

        # Filter header/metadata/address/remarks lines
        if not name or len(name) < 2 or _is_header_row(name, value):
            continue
        if any(kw in name.lower() for kw in [
            "patient", "contact", "phone", "road", "street", "way", "suite", "biotech", "park", "tel:", "fax:",
            "page", "technologist", "department", "doctor", "physician", "clinician", "sample", "specimen",
            "status", "collected", "reported", "accession", "date", "age", "leukocytosis", "neutrophilia",
            "anemia", "suggestive", "observed", "correlation", "remarks", "interpretation", "analyst", "signatory", "director"
        ]):
            continue

        flag = ""
        ref = ""
        unit_tokens = []

        rest_str = " ".join(tokens[val_idx + 1:])

        # Extract reference range if present (prioritize explicit hyphen-separated ranges!)
        m_ref = _HYPHEN_REF.search(rest_str)
        if not m_ref:
            m_ref = _SPACE_REF.search(rest_str)

        if m_ref:
            ref = m_ref.group(0).strip()
            rest_str = rest_str.replace(ref, "").strip()

        # Classify remaining tokens into FLAG vs UNIT
        for tok in rest_str.split():
            clean_tok = tok.strip(".,;:*()[]-")
            if (clean_tok.lower() in _FLAG_WORDS or _FLAG_RE.match(tok)) and not flag:
                flag = clean_tok
            elif clean_tok.lower() in _FLAG_WORDS or _FLAG_RE.match(tok):
                pass
            else:
                unit_tokens.append(tok)

        unit = " ".join(unit_tokens).strip()
        unit, flag = _clean_unit_and_flag(unit, flag)

        rows.append({
            "name": name,
            "value": value,
            "unit": unit,
            "reference_range": ref,
            "flag": flag,
        })

    # Strategy 2: space-aligned rows — fallback when tabs are absent
    if not rows:
        # Match lines like: "Hemoglobin     15.4     g/dL     13 - 17     "
        space_pattern = re.compile(
            r"^(?P<name>[A-Za-z][A-Za-z \-\(\)/\.%+]{2,50?}?)"
            r"\s{2,}"
            r"(?P<value>-?(?:\d+\.?\d*|(?:Nil|Negative|Positive|Trace)))"
            r"(?:\s{2,}(?P<unit>[µ%\w/\.]{0,15}))?"
            r"(?:\s{2,}(?P<refrange>\d[\d\.\s\-–]+\d))?"
            r"(?:\s+(?P<flag>[HLhl*]{1,3}|High|Low|Normal|Critical|Abnormal|Borderline|-))?",
            re.MULTILINE | re.IGNORECASE,
        )
        for m in space_pattern.finditer(text):
            name = m.group("name").strip()
            value = (m.group("value") or "").strip()
            if not name or len(name) < 2:
                continue
            if _is_header_row(name, value):
                continue
            unit_val = (m.group("unit") or "").strip()
            flag_val = (m.group("flag") or "").strip()
            unit_val, flag_val = _clean_unit_and_flag(unit_val, flag_val)
            rows.append({
                "name": name,
                "value": value,
                "unit": unit_val,
                "reference_range": (m.group("refrange") or "").strip(),
                "flag": flag_val,
            })

    # Strategy 3: one-value-per-line layout (common in reportlab-generated PDFs
    # extracted by PyMuPDF where each table column appears on its own line).
    # Pattern:  TestName\nValue\nUnit\nRefRange\nFlag\nTestName2\n...
    if not rows:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Locate the table header sentinel to know where data starts
        table_start = 0
        for i, ln in enumerate(lines):
            if ln.lower() in ("investigation", "test", "parameter", "analyte"):
                # Skip the header block (Investigation / Result / Unit / ... / Flag)
                j = i + 1
                while j < len(lines) and lines[j].lower() in (
                    "result", "value", "unit", "biological reference interval",
                    "reference interval", "reference range", "flag", "normal range", "observation",
                ):
                    j += 1
                table_start = j
                break

        # Sliding window: name, value, and remaining tokens (unit, ref range, flag)
        i = table_start
        _val_re = re.compile(r"^-?\d+\.?\d*$|^(?:Nil|Negative|Positive|Trace|\+{1,3})$", re.IGNORECASE)
        _flag_token_re = re.compile(r"^(?:[HLhl]\*?|CRITICAL)$")
        _ref_token_re = re.compile(r"^\d[\d\.\s\-–<>≤≥]+\d$|^\d+\.\d+\s*-\s*\d+\.\d+$")
        _unit_token_re = re.compile(r"^(?:x10\^\d\s*/\s*uL|g\s*/\s*dL|mg\s*/\s*dL|fL|pg|%|mm\s*/\s*hr|/cmm|million/cmm|µg/dL|IU/mL|[µ%\w/\.\-]{1,15})$")
        _name_re = re.compile(r"^[A-Za-z]")

        while i < len(lines):
            ln = lines[i]
            if not _name_re.match(ln) or _is_header_row(ln, ""):
                i += 1
                continue

            name = ln
            j = i + 1

            # Peek ahead for value
            if j >= len(lines) or not _val_re.match(lines[j]):
                i += 1
                continue
            value = lines[j]; j += 1

            unit = ""
            ref = ""
            flag = ""

            # Examine up to 3 remaining tokens for this test row
            tokens_consumed = 0
            while j < len(lines) and tokens_consumed < 3:
                tok = lines[j]
                if _name_re.match(tok) and not _flag_token_re.match(tok) and not _unit_token_re.match(tok) and len(tok) > 3 and not "/" in tok and not "%" in tok and not "10^" in tok:
                    # Hit the next test name!
                    break

                if _flag_token_re.match(tok) and not flag:
                    flag = tok.strip("-").strip()
                    j += 1
                    tokens_consumed += 1
                elif _ref_token_re.match(tok) and not ref:
                    ref = tok
                    j += 1
                    tokens_consumed += 1
                elif not unit and (_unit_token_re.match(tok) or "/" in tok or "%" in tok or "10^" in tok or tok.lower() in ("fl", "pg", "g/dl", "%", "mm/hr")):
                    unit = tok
                    j += 1
                    tokens_consumed += 1
                else:
                    break

            if _is_header_row(name, value):
                i += 1
                continue

            unit, flag = _clean_unit_and_flag(unit, flag)
            rows.append({
                "name": name,
                "value": value,
                "unit": unit,
                "reference_range": ref,
                "flag": flag,
            })
            i = j

    # Strategy 4: single-space line parser (specifically for EasyOCR line-reconstructed text)
    # EasyOCR joins tokens on the same horizontal line with single spaces.
    # Line format: Test Name <space> Value <space> [Unit] <space> [Ref Range] <space> [Flag]
    if not rows:
        # Match lines like: "Haemoglobin 13.5 gldL 12 - 15.5" or "Mean Corpuscular Hb Concentration 29.5 gldL 31 36 BL"
        ocr_line_pattern = re.compile(
            r"^(?P<name>[A-Za-z][A-Za-z0-9 \-\(\)/%\.]{2,45}?)"
            r"\s+"
            r"(?P<value>-?\d+\.?\d*|(?:Nil|Negative|Positive|Trace|\+{1,3}))"
            r"(?:\s+(?P<unit>[µ%\w/\.]{1,15}))?"
            r"(?:\s+(?P<refrange>\d[\d\.\s\-–<>≤≥]+\d|\d+\s+\d+))?"
            r"(?:\s+(?P<flag>[HLhl*]{1,3}|High|Low|Normal|Critical|Abnormal|Borderline|BL|BH|-))?$",
            re.MULTILINE | re.IGNORECASE,
        )
        for m in ocr_line_pattern.finditer(text):
            name = m.group("name").strip()
            value = (m.group("value") or "").strip()
            if not name or len(name) < 2:
                continue
            if _is_header_row(name, value):
                continue
            unit_val = (m.group("unit") or "").strip()
            flag_val = (m.group("flag") or "").strip()
            unit_val, flag_val = _clean_unit_and_flag(unit_val, flag_val)
            rows.append({
                "name": name,
                "value": value,
                "unit": unit_val,
                "reference_range": (m.group("refrange") or "").strip(),
                "flag": flag_val,
            })

    return rows


_TRAILING_NUMS_RE = re.compile(r"\s+[-+]?\d+\.?\d*.*$")

def build_extracted_section(
    section_name: str,
    rows: List[Dict[str, Any]],
) -> ExtractedSection:
    """Convert raw row dicts into a typed ExtractedSection."""
    tests: List[ExtractedTest] = []
    for row in rows:
        raw_name = row.get("name", "").strip()
        raw_value = row.get("value", "").strip()
        raw_unit = row.get("unit", "").strip()
        raw_range = row.get("reference_range", "").strip()
        raw_flag = row.get("flag", "").strip()

        if not raw_name:
            continue

        raw_unit, detected_flag = _clean_unit_and_flag(raw_unit, raw_flag)
        if not raw_flag and detected_flag:
            raw_flag = detected_flag

        # Clean trailing numbers accidentally attached to raw_name
        clean_name = _TRAILING_NUMS_RE.sub("", raw_name).strip() if any(c.isdigit() for c in raw_name) else raw_name
        if not clean_name:
            clean_name = raw_name

        normalized_name = normalize_test_name(clean_name)
        normalized_unit = normalize_unit(raw_unit)
        ref_low, ref_high = parse_reference_range(raw_range)
        value_numeric = _parse_value(raw_value)

        # Basic confidence: 1.0 if we got a numeric value + known test name
        confidence = 1.0
        if value_numeric is None:
            confidence -= 0.3
        if normalized_name == clean_name:  # name wasn't recognised
            confidence -= 0.2
        confidence = max(0.0, min(1.0, confidence))

        tests.append(
            ExtractedTest(
                test_name_raw=clean_name,
                test_name_normalized=normalized_name,
                value_raw=raw_value,
                value_numeric=value_numeric,
                unit_raw=raw_unit,
                unit_normalized=normalized_unit,
                reference_range_raw=raw_range,
                reference_low=ref_low,
                reference_high=ref_high,
                flag_raw=raw_flag,
                confidence=confidence,
            )
        )
    return ExtractedSection(section_name=section_name, tests=tests)
