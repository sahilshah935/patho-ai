"""Plain-language explanation and clinical interpretation generator for pathology results.

All explanations are clearly framed as report-understanding support and
include the mandatory medical disclaimer.

IMPORTANT: These explanations are for decision support only.
They are NOT a clinical diagnosis and must be confirmed by a qualified
healthcare professional.
"""

from __future__ import annotations

from typing import List, Dict

from pathology_ai.extraction.schema import ExtractedTest
from pathology_ai.inference.interpreter import InterpretationResult

# ---------------------------------------------------------------------------
# Mandatory disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "⚠️  IMPORTANT DISCLAIMER: This output is for pathology report "
    "understanding and clinical decision support only. It is NOT a medical "
    "diagnosis. All results must be reviewed and confirmed by a qualified "
    "healthcare professional before any clinical action is taken."
)

# ---------------------------------------------------------------------------
# Status descriptors
# ---------------------------------------------------------------------------

_STATUS_PHRASES = {
    "NORMAL":          ("is within the normal range", ""),
    "HIGH":            ("is above the normal range", "This may require clinical attention."),
    "LOW":             ("is below the normal range", "This may require clinical attention."),
    "BORDERLINE_HIGH": ("is slightly above the upper reference limit", "Monitor and correlate clinically."),
    "BORDERLINE_LOW":  ("is slightly below the lower reference limit", "Monitor and correlate clinically."),
    "CRITICAL_HIGH":   ("is CRITICALLY ELEVATED", "⚠️  Urgent clinical attention is recommended."),
    "CRITICAL_LOW":    ("is CRITICALLY LOW", "⚠️  Urgent clinical attention is recommended."),
    "QUALITATIVE":     ("is a qualitative result", ""),
    "UNKNOWN":         ("could not be classified", "Reference range or numeric value unavailable."),
}


# ---------------------------------------------------------------------------
# Per-test explanation
# ---------------------------------------------------------------------------

def explain_test(test: ExtractedTest) -> str:
    """Generate a one-sentence plain-language explanation for a single test."""
    name = test.test_name_normalized or test.test_name_raw
    value = test.value_raw
    unit = test.unit_normalized or test.unit_raw
    status = test.status or "UNKNOWN"

    phrase, note = _STATUS_PHRASES.get(status, ("has an unclassified result", ""))

    value_str = f"{value} {unit}".strip() if unit else value

    if test.reference_low is not None and test.reference_high is not None:
        ref_str = f"(normal range: {test.reference_low}–{test.reference_high} {unit})".strip()
    elif test.reference_high is not None:
        ref_str = f"(normal: < {test.reference_high} {unit})".strip()
    elif test.reference_low is not None:
        ref_str = f"(normal: > {test.reference_low} {unit})".strip()
    else:
        ref_str = ""

    parts = [f"{name} is {value_str}, which {phrase}"]
    if ref_str:
        parts[0] += f" {ref_str}"
    parts[0] += "."
    if note:
        parts.append(note)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------

def generate_summary(result: InterpretationResult) -> str:
    """Generate an overall plain-language report summary."""
    tests = result.all_tests
    category = result.document_category or "laboratory"
    total = len(tests)

    if total == 0:
        return (
            f"No test results could be extracted from this {category} report. "
            "Manual review of the original document is advised."
        )

    normal = sum(1 for t in tests if t.status == "NORMAL")
    high = sum(1 for t in tests if t.status in ("HIGH", "BORDERLINE_HIGH"))
    low = sum(1 for t in tests if t.status in ("LOW", "BORDERLINE_LOW"))
    critical = sum(1 for t in tests if t.critical)

    lines: List[str] = []

    lines.append(
        f"This {category} report contains {total} test result(s). "
        f"{normal} are within the normal range."
    )

    if high:
        high_names = [t.test_name_normalized or t.test_name_raw
                      for t in tests if t.status in ("HIGH", "BORDERLINE_HIGH")]
        lines.append(
            f"{high} result(s) are elevated: {', '.join(high_names)}."
        )

    if low:
        low_names = [t.test_name_normalized or t.test_name_raw
                     for t in tests if t.status in ("LOW", "BORDERLINE_LOW")]
        lines.append(
            f"{low} result(s) are below the reference range: {', '.join(low_names)}."
        )

    if critical:
        crit_names = [t.test_name_normalized or t.test_name_raw
                      for t in tests if t.critical]
        lines.append(
            f"⚠️  {critical} CRITICAL value(s) detected: {', '.join(crit_names)}. "
            "Immediate clinical review is strongly advised."
        )

    if result.patterns:
        lines.append(
            "Pattern(s) detected: " +
            "; ".join(p.name for p in result.patterns) + "."
        )

    if result.requires_review:
        lines.append(
            "This report is flagged for healthcare professional review."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Dynamic Structured Clinical Triad Generator
# ---------------------------------------------------------------------------

def generate_clinical_triad(result: InterpretationResult) -> Dict[str, str]:
    """Dynamically generate Clinical Interpretation, Pathologist Remarks, and Suggested Clinical Correlation.

    Customized for each unique uploaded report based on its exact biomarkers and patterns.
    """
    tests = result.all_tests
    category = result.extraction.document_category or "Laboratory"
    patient = result.extraction.patient

    high_tests = [t for t in tests if t.status in ("HIGH", "BORDERLINE_HIGH", "CRITICAL_HIGH")]
    low_tests = [t for t in tests if t.status in ("LOW", "BORDERLINE_LOW", "CRITICAL_LOW")]
    critical_tests = [t for t in tests if t.critical]
    patterns = result.patterns

    tests_by_name = {t.test_name_normalized: t for t in tests}

    # 1. Clinical Interpretation
    interp_parts = []

    if not high_tests and not low_tests:
        interp_parts.append(
            f"{category} evaluation shows all {len(tests)} tested parameter(s) within physiological biological reference intervals. "
            "No isolated or combined biomarker abnormalities were detected."
        )
    else:
        abnormal_summary = []
        if low_tests:
            low_strs = [f"{t.test_name_normalized} ({t.value_raw} {t.unit_normalized or t.unit_raw})".strip() for t in low_tests]
            abnormal_summary.append(f"reduced levels of {', '.join(low_strs)}")
        if high_tests:
            high_strs = [f"{t.test_name_normalized} ({t.value_raw} {t.unit_normalized or t.unit_raw})".strip() for t in high_tests]
            abnormal_summary.append(f"elevated levels of {', '.join(high_strs)}")

        interp_parts.append(
            f"{category} panel demonstrates {' accompanied by '.join(abnormal_summary)}."
        )

        if patterns:
            pattern_names = [p.name for p in patterns]
            interp_parts.append(f"Primary pattern features align with {', '.join(pattern_names)}.")

    clinical_interpretation = " ".join(interp_parts)

    # 2. Pathologist Remarks
    remark_parts = []

    if not high_tests and not low_tests:
        remark_parts.append(
            "Normal laboratory profile. Parameters are consistent with healthy baseline status. "
            "No specific pathophysiological pattern identified on current testing."
        )
    else:
        if patterns:
            for p in patterns:
                remark_parts.append(f"Pattern is compatible with {p.description}")
        else:
            ab_names = [t.test_name_normalized for t in (high_tests + low_tests)]
            remark_parts.append(
                f"Isolated or non-classical deviation observed in {', '.join(ab_names[:4])}. "
                "May reflect mild physiological variation, early stage response, or subclinical fluctuation."
            )

        if critical_tests:
            crit_names = [t.test_name_normalized for t in critical_tests]
            remark_parts.append(
                f"ALERT: Critical value threshold triggered for {', '.join(crit_names)}. Urgent clinical evaluation recommended."
            )

        remark_parts.append("Correlation with clinical history, medication, and baseline laboratory trends is advised.")

    pathologist_remarks = " ".join(remark_parts)

    # 3. Suggested Clinical Correlation
    corr_parts = []

    if not high_tests and not low_tests:
        corr_parts.append(
            "Maintain routine health monitoring. Correlate with general well-being and schedule routine annual health check-up as advised by primary healthcare provider."
        )
    else:
        symptoms = []
        followup_tests = []

        # Category & Pattern specific correlations
        if "Hemoglobin" in tests_by_name and tests_by_name["Hemoglobin"].status == "LOW":
            symptoms.extend(["fatigue", "pallor", "shortness of breath on exertion", "dizziness"])
            followup_tests.extend(["Iron Profile (Ferritin, TIBC, Serum Iron)", "Peripheral Blood Smear", "Reticulocyte Count"])

        if "Total Leucocyte Count" in tests_by_name and tests_by_name["Total Leucocyte Count"].status == "HIGH":
            symptoms.extend(["fever", "localized pain or inflammation", "recent infection history"])
            followup_tests.extend(["C-Reactive Protein (CRP)", "Infectious Workup / Cultures"])

        if "Glucose - Fasting" in tests_by_name and tests_by_name["Glucose - Fasting"].status == "HIGH":
            symptoms.extend(["increased thirst (polydipsia)", "frequent urination (polyuria)", "unexplained weight changes"])
            followup_tests.extend(["HbA1c", "Post-prandial Blood Glucose", "Lipid Profile"])

        if "Creatinine" in tests_by_name and tests_by_name["Creatinine"].status == "HIGH":
            symptoms.extend(["decreased urine output", "peripheral edema (swelling)", "hypertension"])
            followup_tests.extend(["Serum Electrolytes", "Urine Routine & Microalbumin", "Renal Ultrasound"])

        if "TSH" in tests_by_name and tests_by_name["TSH"].status != "NORMAL":
            symptoms.extend(["weight changes", "temperature intolerance", "fatigue or anxiety"])
            followup_tests.extend(["Free T3 & Free T4", "Anti-TPO Antibodies"])

        if "Total Cholesterol" in tests_by_name and tests_by_name["Total Cholesterol"].status == "HIGH":
            symptoms.extend(["cardiovascular risk factors", "family history of premature CAD"])
            followup_tests.extend(["Fasting Lipid Subfractions", "ECG", "High-Sensitivity CRP"])

        # Fallback general correlation if specific ones not triggered
        if not symptoms:
            symptoms.extend(["presenting symptoms", "relevant clinical findings", "medication history"])
        if not followup_tests:
            followup_tests.extend(["Repeat targeted testing in 4-6 weeks", "Relevant specialty consultation"])

        corr_parts.append(
            f"Correlate laboratory findings with patient clinical presentation, specifically evaluating for {', '.join(symptoms[:4])}."
        )
        corr_parts.append(
            f"Recommended follow-up consideration: {'; '.join(followup_tests[:3])}."
        )

    suggested_clinical_correlation = " ".join(corr_parts)

    return {
        "clinical_interpretation": clinical_interpretation,
        "pathologist_remarks": pathologist_remarks,
        "suggested_clinical_correlation": suggested_clinical_correlation,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_disclaimer() -> str:
    """Return the mandatory medical disclaimer string."""
    return DISCLAIMER


def explain_and_summarise(result: InterpretationResult) -> InterpretationResult:
    """Populate per-test explanations, overall summary, and clinical triad on ``result``.

    Mutates ``result`` in-place and returns it for chaining.
    """
    for test in result.all_tests:
        test.explanation = explain_test(test)

    result.summary = generate_summary(result)

    # Populate dynamic clinical triad
    triad = generate_clinical_triad(result)
    result.clinical_interpretation = triad["clinical_interpretation"]
    result.pathologist_remarks = triad["pathologist_remarks"]
    result.suggested_clinical_correlation = triad["suggested_clinical_correlation"]

    result.disclaimer = DISCLAIMER
    result.extraction.summary = result.summary

    return result
