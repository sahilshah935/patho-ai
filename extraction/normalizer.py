"""Rule-based normalization for test names, units, and reference range strings.

Driven by the analyte vocabulary defined in the dataset generator.  Uses
``rapidfuzz`` for fuzzy test-name matching so we can handle OCR noise and
spelling variations gracefully.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Analyte vocabulary
# All variant names → canonical normalised name.
# Derived from the ANALYTES dict in pathology_generator_v4.py.
# ---------------------------------------------------------------------------

_ANALYTE_ALIASES: Dict[str, str] = {
    # Hematology
    "hemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "haemoglobin": "Hemoglobin",
    "rbc count": "RBC Count",
    "rbc": "RBC Count",
    "red blood cell count": "RBC Count",
    "packed cell volume": "Packed Cell Volume",
    "pcv": "Packed Cell Volume",
    "haematocrit": "Packed Cell Volume",
    "hematocrit": "Packed Cell Volume",
    "hct": "Packed Cell Volume",
    "mcv": "MCV",
    "mean corpuscular volume": "MCV",
    "mch": "MCH",
    "mean corpuscular haemoglobin": "MCH",
    "mean corpuscular hemoglobin": "MCH",
    "mchc": "MCHC",
    "mean corpuscular hb concentration": "MCHC",
    "mean corpuscular haemoglobin concentration": "MCHC",
    "rdw-cv": "RDW-CV",
    "rdw": "RDW-CV",
    "total leucocyte count": "Total Leucocyte Count",
    "tlc": "Total Leucocyte Count",
    "total wbc count": "Total Leucocyte Count",
    "wbc": "Total Leucocyte Count",
    "wbc count": "Total Leucocyte Count",
    "white blood cell count": "Total Leucocyte Count",
    "white blood cell count (wbc)": "Total Leucocyte Count",
    "white blood cell": "Total Leucocyte Count",
    "total wbc": "Total Leucocyte Count",
    "neutrophils": "Neutrophils",
    "neutrophil %": "Neutrophils",
    "lymphocyte %": "Lymphocyte %",
    "lymphocytes": "Lymphocyte %",
    "monocytes": "Monocytes",
    "monocyte %": "Monocytes",
    "eosinophils": "Eosinophils",
    "eosinophil %": "Eosinophils",
    "basophils": "Basophils",
    "basophil %": "Basophils",
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "mpv": "MPV",
    "mean platelet volume": "MPV",
    "esr": "ESR",
    "erythrocyte sedimentation rate": "ESR",
    # Biochemistry
    "glucose - fasting": "Glucose - Fasting",
    "fasting blood sugar": "Glucose - Fasting",
    "fbs": "Glucose - Fasting",
    "hba1c": "HbA1c",
    "glycated haemoglobin": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "blood urea": "Blood Urea",
    "urea": "Blood Urea",
    "creatinine": "Creatinine",
    "serum creatinine": "Creatinine",
    "uric acid": "Uric Acid",
    "total bilirubin": "Total Bilirubin",
    "direct bilirubin": "Direct Bilirubin",
    "sgot": "SGOT",
    "ast": "SGOT",
    "sgpt": "SGPT",
    "alt": "SGPT",
    "alkaline phosphatase": "Alkaline Phosphatase",
    "alp": "Alkaline Phosphatase",
    "total protein": "Total Protein",
    "albumin": "Albumin",
    "sodium": "Sodium",
    "na+": "Sodium",
    "potassium": "Potassium",
    "k+": "Potassium",
    "total cholesterol": "Total Cholesterol",
    "triglycerides": "Triglycerides",
    "hdl cholesterol": "HDL Cholesterol",
    "hdl-c": "HDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "ldl-c": "LDL Cholesterol",
    "ldl": "LDL Cholesterol",
    # Urinalysis
    "ph": "pH",
    "urine ph": "pH",
    "specific gravity": "Specific Gravity",
    "sp. gravity": "Specific Gravity",
    "protein": "Protein",
    "urine protein": "Protein",
    "glucose": "Urine Glucose",
    "urine glucose": "Urine Glucose",
    "rbcs": "RBCs",
    "rbc / hpf": "RBCs",
    "pus cells": "Pus Cells",
    "wbc / hpf": "Pus Cells",
    # Endocrinology
    "tsh": "TSH",
    "thyroid stimulating hormone": "TSH",
    "free t4": "Free T4",
    "ft4": "Free T4",
    "free t3": "Free T3",
    "ft3": "Free T3",
    "25-oh vitamin d": "25-OH Vitamin D",
    "vitamin d total": "25-OH Vitamin D",
    "vitamin d": "25-OH Vitamin D",
    "vitamin b12": "Vitamin B12",
    "cobalamin": "Vitamin B12",
    "fasting insulin": "Fasting Insulin",
    "insulin": "Fasting Insulin",
    # Serology
    "c-reactive protein": "C-Reactive Protein",
    "crp": "C-Reactive Protein",
    "rheumatoid factor": "Rheumatoid Factor",
    "rf": "Rheumatoid Factor",
    "aso titre": "ASO Titre",
    "antistreptolysin o": "ASO Titre",
    "ferritin": "Ferritin",
}

# Sorted list of canonical names for fuzzy matching
_CANONICAL_NAMES = sorted(set(_ANALYTE_ALIASES.values()))

# ---------------------------------------------------------------------------
# Unit normalisation map
# ---------------------------------------------------------------------------

_UNIT_MAP: Dict[str, str] = {
    # Hemoglobin / protein
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "grams/dl": "g/dL",
    "g/l": "g/L",
    # Counts & volumes
    "/cmm": "/cmm",
    "cells/cmm": "/cmm",
    "million/cmm": "million/cmm",
    "mill/cmm": "million/cmm",
    "/ul": "/µL",
    "/µl": "/µL",
    "cells/ul": "/µL",
    "cells/µl": "/µL",
    "million/ul": "million/µL",
    "million/µl": "million/µL",
    "x10^3/ul": "x10^3/µL",
    "x10^3/µl": "x10^3/µL",
    "x10^6/ul": "x10^6/µL",
    "x10^6/µl": "x10^6/µL",
    # Percentage
    "%": "%",
    "percent": "%",
    # Volume
    "fl": "fL",
    "ml": "mL",
    # Mass
    "pg": "pg",
    "pg/ml": "pg/mL",
    "ng/ml": "ng/mL",
    "ng/dl": "ng/dL",
    "mg/dl": "mg/dL",
    "mg/l": "mg/L",
    "ug/dl": "µg/dL",
    "µg/dl": "µg/dL",
    "mcg/dl": "µg/dL",
    # Molar
    "mmol/l": "mmol/L",
    "µmol/l": "µmol/L",
    "umol/l": "µmol/L",
    # Endocrinology
    "µiu/ml": "µIU/mL",
    "uiu/ml": "µIU/mL",
    "miu/ml": "mIU/mL",
    "iu/ml": "IU/mL",
    "iu/l": "IU/L",
    "u/l": "U/L",
    # Urine
    "/hpf": "/HPF",
    "per hpf": "/HPF",
    "lakh/cmm": "lakh/cmm",
    # Speed
    "mm/hr": "mm/hr",
    "mm/h": "mm/hr",
    # OCR typos
    "gldl": "g/dL",
    "gmldl": "g/dL",
    "icmm": "/cmm",
    "per cmm": "/cmm",
    "lakhlcmm": "lakh/cmm",
    "lakhicmm": "lakh/cmm",
    "millionlcmm": "million/cmm",
    "milllcmm": "million/cmm",
    "lhpf": "/HPF",
    "perlhpf": "/HPF",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_test_name(raw: str, score_threshold: int = 70) -> str:
    """Normalise a raw test name to a canonical name.

    First checks the alias dictionary (exact, lower-cased).  If not found,
    falls back to fuzzy matching against canonical names using rapidfuzz.

    Args:
        raw: Raw test name string from the document.
        score_threshold: Minimum fuzzy score (0–100) to accept a match.

    Returns:
        Normalised canonical name, or the original ``raw`` string if no
        confident match is found.
    """
    if not raw:
        return raw

    key = raw.strip().lower()

    # Exact alias lookup (fast path)
    if key in _ANALYTE_ALIASES:
        return _ANALYTE_ALIASES[key]

    if not HAS_RAPIDFUZZ:
        return raw

    # Fuzzy match against alias keys
    alias_result = rf_process.extractOne(
        key,
        _ANALYTE_ALIASES.keys(),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_threshold,
    )
    if alias_result:
        matched_key = alias_result[0]
        return _ANALYTE_ALIASES[matched_key]

    # Fuzzy match directly against canonical names
    canonical_result = rf_process.extractOne(
        raw.strip(),
        _CANONICAL_NAMES,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_threshold,
    )
    if canonical_result:
        return canonical_result[0]

    return raw


_FLAG_WORDS_SET = {
    "high", "low", "normal", "abnormal", "borderline", "critical",
    "alert", "positive", "negative", "bl", "bh", "crit", "h", "l"
}


def normalize_unit(raw: str) -> str:
    """Normalise a unit string to a canonical form.

    Strips any trailing/leading flag annotations and normalises whitespace/slashes.

    Args:
        raw: Raw unit string from the document (e.g. ``"gm/dl"``, ``"g/dL Low"``).

    Returns:
        Normalised unit string, or the stripped original if not in the map.
    """
    if raw is None:
        return ""

    tokens = [tok for tok in raw.strip().split() if tok.strip(".,;:*()[]").lower() not in _FLAG_WORDS_SET]
    cleaned = " ".join(tokens).strip().strip(".,;:*()[]")
    if not cleaned:
        return ""

    key = cleaned.lower()
    # Normalize internal slashes: e.g. "g / dl" -> "g/dl"
    key = re.sub(r"\s*/\s*", "/", key)

    if key in _UNIT_MAP:
        return _UNIT_MAP[key]

    # Try mapping bare token if single slash present
    tokens_sub = [t.lower() for t in cleaned.split()]
    for t in tokens_sub:
        t_clean = re.sub(r"\s*/\s*", "/", t)
        if t_clean in _UNIT_MAP:
            return _UNIT_MAP[t_clean]

    return cleaned


# Reference-range patterns
_RANGE_BETWEEN = re.compile(
    r"(?P<lo>-?\d+\.?\d*)\s*[-–—to]+\s*(?P<hi>-?\d+\.?\d*)", re.IGNORECASE
)
_RANGE_TWO_NUMS = re.compile(
    r"^(?P<lo>\d+\.?\d*)\s+(?P<hi>\d+\.?\d*)$"
)
_RANGE_LESS = re.compile(r"[<≤]\s*(?P<hi>\d+\.?\d*)")
_RANGE_GREATER = re.compile(r"[>≥]\s*(?P<lo>\d+\.?\d*)")
_RANGE_UPTO = re.compile(r"up\s*to\s*(?P<hi>\d+\.?\d*)", re.IGNORECASE)
_RANGE_SINGLE = re.compile(r"^\d+\.?\d*$")


def parse_reference_range(raw: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a raw reference-range string into (low, high) float pair.

    Handles patterns like:
    - ``"13 - 17"`` → (13.0, 17.0)
    - ``"36 46"`` → (36.0, 46.0) (space-separated OCR range)
    - ``"< 200"`` → (None, 200.0)
    - ``"> 40"`` → (40.0, None)
    - ``"Up to 5"`` → (None, 5.0)
    - ``"Negative"`` / qualitative → (None, None)

    Args:
        raw: Raw reference range string.

    Returns:
        Tuple ``(low, high)``, where either may be ``None``.
    """
    if not raw or not raw.strip():
        return None, None

    text = raw.strip()

    m = _RANGE_BETWEEN.search(text)
    if m:
        return float(m.group("lo")), float(m.group("hi"))

    m = _RANGE_TWO_NUMS.search(text)
    if m:
        return float(m.group("lo")), float(m.group("hi"))

    m = _RANGE_LESS.search(text)
    if m:
        return None, float(m.group("hi"))

    m = _RANGE_UPTO.search(text)
    if m:
        return None, float(m.group("hi"))

    m = _RANGE_GREATER.search(text)
    if m:
        return float(m.group("lo")), None

    if _RANGE_SINGLE.match(text):
        return None, float(text)

    # Qualitative (Negative, Positive, Trace, etc.) → no numeric range
    return None, None
