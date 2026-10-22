"""Deterministic clinical interpretation rules.

All rules here are deliberately simplified and operate on numeric values only.
They are designed for decision-support — NOT for independent clinical diagnosis.

The status taxonomy used:
    NORMAL           — value within reference range
    BORDERLINE_HIGH  — within 10 % above the upper limit
    BORDERLINE_LOW   — within 10 % below the lower limit
    HIGH             — value above the upper reference limit
    LOW              — value below the lower reference limit
    CRITICAL_HIGH    — value at or above the critical high threshold
    CRITICAL_LOW     — value at or below the critical low threshold
    QUALITATIVE      — qualitative result (Positive / Negative / Trace)
    UNKNOWN          — cannot be determined (missing reference range or value)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Critical thresholds
# Derived from the ANALYTES dict in pathology_generator_v4.py.
# Values here are illustrative; do NOT use as clinical reference.
# ---------------------------------------------------------------------------

_CRITICAL_THRESHOLDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "Hemoglobin":            (7.0,    20.0),
    "RBC Count":             (2.0,    7.0),
    "Packed Cell Volume":    (20.0,   65.0),
    "MCV":                   (55.0,   125.0),
    "MCH":                   (15.0,   40.0),
    "MCHC":                  (25.0,   40.0),
    "Total Leucocyte Count": (2000,   30000),
    "Platelet Count":        (0.5,    10.0),
    "Glucose - Fasting":     (45.0,   400.0),
    "HbA1c":                 (None,   14.0),
    "Blood Urea":            (5.0,    150.0),
    "Creatinine":            (0.2,    10.0),
    "Total Bilirubin":       (None,   15.0),
    "Direct Bilirubin":      (None,   8.0),
    "Sodium":                (120.0,  160.0),
    "Potassium":             (2.5,    6.5),
    "TSH":                   (0.05,   50.0),
    "Free T4":               (0.2,    5.0),
    "25-OH Vitamin D":       (None,   200.0),
    "Vitamin B12":           (80.0,   2000.0),
    "Albumin":               (2.0,    7.0),
    "C-Reactive Protein":    (None,   200.0),
    "Rheumatoid Factor":     (None,   500.0),
    "Ferritin":              (None,   2000.0),
}

# Borderline zone: fraction of the reference interval width
_BORDERLINE_FRACTION = 0.10

# Qualitative values that cannot be numerically classified
_QUALITATIVE_VALUES = {
    "negative", "positive", "trace", "nil", "absent",
    "present", "few", "occasional", "moderate", "many", "rare",
    "+", "++", "+++",
}


# ---------------------------------------------------------------------------
# Multi-test pattern rules
# ---------------------------------------------------------------------------


class PatternRule:
    """A multi-test pattern that fires when all required tests satisfy a predicate."""

    def __init__(self, name: str, description: str, required_statuses: Dict[str, str]):
        self.name = name
        self.description = description
        self.required_statuses = required_statuses  # {normalized_name: status}

    def matches(self, tests_by_name: Dict[str, "ExtractedTest"]) -> bool:  # noqa: F821
        for test_name, required_status in self.required_statuses.items():
            test = tests_by_name.get(test_name)
            if test is None or test.status != required_status:
                return False
        return True


MULTI_TEST_PATTERNS: List[PatternRule] = [
    PatternRule(
        name="Microcytic Anaemia Pattern",
        description="Low Hemoglobin, low MCV, and low MCH suggest microcytic anaemia (possible iron deficiency or thalassaemia).",
        required_statuses={"Hemoglobin": "LOW", "MCV": "LOW", "MCH": "LOW"},
    ),
    PatternRule(
        name="Macrocytic Anaemia Pattern",
        description="Low Hemoglobin with high MCV suggests macrocytic anaemia (possible B12 or folate deficiency).",
        required_statuses={"Hemoglobin": "LOW", "MCV": "HIGH"},
    ),
    PatternRule(
        name="Thrombocytopenia",
        description="Platelet count is below the lower reference limit. Monitoring and clinical correlation advised.",
        required_statuses={"Platelet Count": "LOW"},
    ),
    PatternRule(
        name="Leukocytosis",
        description="Total leucocyte count is elevated. May indicate infection, inflammation, or other conditions.",
        required_statuses={"Total Leucocyte Count": "HIGH"},
    ),
    PatternRule(
        name="Diabetic Glycaemia Pattern",
        description="Both fasting glucose and HbA1c are above the normal range, consistent with poor glycaemic control.",
        required_statuses={"Glucose - Fasting": "HIGH", "HbA1c": "HIGH"},
    ),
    PatternRule(
        name="Dyslipidaemia Pattern",
        description="Elevated total cholesterol and LDL with low HDL indicate dyslipidaemia — a cardiovascular risk factor.",
        required_statuses={"Total Cholesterol": "HIGH", "LDL Cholesterol": "HIGH", "HDL Cholesterol": "LOW"},
    ),
    PatternRule(
        name="Hypothyroidism Pattern",
        description="Elevated TSH with low Free T4 suggests primary hypothyroidism.",
        required_statuses={"TSH": "HIGH", "Free T4": "LOW"},
    ),
    PatternRule(
        name="Hyperthyroidism Pattern",
        description="Low TSH with high Free T4 suggests hyperthyroidism.",
        required_statuses={"TSH": "LOW", "Free T4": "HIGH"},
    ),
    PatternRule(
        name="Renal Impairment Pattern",
        description="Elevated urea and creatinine suggest impaired renal function. Further evaluation recommended.",
        required_statuses={"Blood Urea": "HIGH", "Creatinine": "HIGH"},
    ),
]


# ---------------------------------------------------------------------------
# Core classification functions
# ---------------------------------------------------------------------------

def classify_status(
    value: Optional[float],
    ref_low: Optional[float],
    ref_high: Optional[float],
    value_raw: str = "",
    critical_low: Optional[float] = None,
    critical_high: Optional[float] = None,
) -> str:
    """Classify a test value against its reference range.

    Args:
        value: Numeric test value.
        ref_low: Lower reference limit (None if unbounded).
        ref_high: Upper reference limit (None if unbounded).
        value_raw: Raw string value (used to detect qualitative results).
        critical_low: Critical low threshold (overrides ref_low for CRITICAL).
        critical_high: Critical high threshold (overrides ref_high for CRITICAL).

    Returns:
        Status string from the taxonomy documented above.
    """
    # Check for qualitative values
    if value_raw.strip().lower() in _QUALITATIVE_VALUES:
        return "QUALITATIVE"

    if value is None:
        return "UNKNOWN"

    if ref_low is None and ref_high is None:
        return "UNKNOWN"

    # Check if value is within normal reference range
    is_normal = False
    if ref_low is not None and ref_high is not None:
        if ref_low <= value <= ref_high:
            is_normal = True
    elif ref_low is not None and ref_high is None:
        if value >= ref_low:
            is_normal = True
    elif ref_high is not None and ref_low is None:
        if value <= ref_high:
            is_normal = True

    if is_normal:
        return "NORMAL"

    # Critical checks: only apply if critical threshold scale aligns with reference range scale
    if critical_high is not None and value >= critical_high:
        if ref_high is None or (ref_high / 10.0 <= critical_high <= ref_high * 10.0):
            return "CRITICAL_HIGH"
    if critical_low is not None and value <= critical_low:
        if ref_low is None or (ref_low / 10.0 <= critical_low <= ref_low * 10.0):
            return "CRITICAL_LOW"

    # Compute borderline zone width
    if ref_low is not None and ref_high is not None:
        zone = (ref_high - ref_low) * _BORDERLINE_FRACTION
    else:
        zone = 0.0

    # High
    if ref_high is not None and value > ref_high:
        if zone > 0 and value <= ref_high + zone:
            return "BORDERLINE_HIGH"
        return "HIGH"

    # Low
    if ref_low is not None and value < ref_low:
        if zone > 0 and value >= ref_low - zone:
            return "BORDERLINE_LOW"
        return "LOW"

    return "NORMAL"


def determine_flag(status: str) -> str:
    """Map a status string to a display flag character."""
    return {
        "HIGH": "H",
        "BORDERLINE_HIGH": "H",
        "CRITICAL_HIGH": "H*",
        "LOW": "L",
        "BORDERLINE_LOW": "L",
        "CRITICAL_LOW": "L*",
    }.get(status, "")


def is_critical(status: str) -> bool:
    """Return True for critical-level statuses."""
    return status in ("CRITICAL_HIGH", "CRITICAL_LOW")


def detect_patterns(tests_by_name: Dict) -> List[PatternRule]:
    """Return all multi-test patterns that match the given test dict.

    Args:
        tests_by_name: Dict mapping normalised test name → ExtractedTest.
    """
    return [rule for rule in MULTI_TEST_PATTERNS if rule.matches(tests_by_name)]


def get_critical_thresholds(test_name_normalized: str) -> Tuple[Optional[float], Optional[float]]:
    """Look up critical thresholds for a normalised test name."""
    return _CRITICAL_THRESHOLDS.get(test_name_normalized, (None, None))
