"""Interpreter: applies clinical rules to an ExtractionResult.

Fills in status, flag, and critical on each ExtractedTest; detects
multi-test patterns; and produces an InterpretationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from pathology_ai.extraction.schema import ExtractionResult, ExtractedTest
from pathology_ai.inference.rules import (
    PatternRule,
    classify_status,
    detect_patterns,
    determine_flag,
    get_critical_thresholds,
    is_critical,
)


# ---------------------------------------------------------------------------
# Interpretation output
# ---------------------------------------------------------------------------

@dataclass
class InterpretationResult:
    """Extends ExtractionResult with interpreted statuses and patterns."""

    extraction: ExtractionResult

    # Detected multi-test patterns
    patterns: List[PatternRule] = field(default_factory=list)

    # True when any test is critical OR extraction_confidence < 0.6
    requires_review: bool = False

    # Per-test explanations are populated by the Explainer
    # Overall summary and clinical triad are populated by the Explainer
    summary: str = ""
    clinical_interpretation: str = ""
    pathologist_remarks: str = ""
    suggested_clinical_correlation: str = ""
    disclaimer: str = ""

    @property
    def document_category(self) -> str:
        return self.extraction.document_category

    @property
    def all_tests(self) -> List[ExtractedTest]:
        return self.extraction.all_tests()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction": self.extraction.to_dict(),
            "patterns": [
                {"name": p.name, "description": p.description}
                for p in self.patterns
            ],
            "requires_review": self.requires_review,
            "summary": self.summary,
            "clinical_interpretation": self.clinical_interpretation,
            "pathologist_remarks": self.pathologist_remarks,
            "suggested_clinical_correlation": self.suggested_clinical_correlation,
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Applies deterministic clinical rules to a fully extracted result.

    Usage::

        interpreter = Interpreter()
        interpretation = interpreter.interpret(extraction_result)
    """

    def interpret(self, extraction: ExtractionResult) -> InterpretationResult:
        """Classify each test and detect multi-test patterns.

        Args:
            extraction: ``ExtractionResult`` from the extraction pipeline.

        Returns:
            ``InterpretationResult`` with status, flag, critical, and patterns.
        """
        # --- Classify each test ---
        any_critical = False
        for test in extraction.all_tests():
            crit_low, crit_high = get_critical_thresholds(test.test_name_normalized)
            status = classify_status(
                value=test.value_numeric,
                ref_low=test.reference_low,
                ref_high=test.reference_high,
                value_raw=test.value_raw,
                critical_low=crit_low,
                critical_high=crit_high,
            )
            test.status = status
            test.flag = determine_flag(status)
            test.critical = is_critical(status)
            if test.critical:
                any_critical = True

        # --- Detect multi-test patterns ---
        tests_by_name: Dict[str, ExtractedTest] = {
            t.test_name_normalized: t for t in extraction.all_tests()
        }
        patterns = detect_patterns(tests_by_name)

        requires_review = any_critical or extraction.extraction_confidence < 0.6

        return InterpretationResult(
            extraction=extraction,
            patterns=patterns,
            requires_review=requires_review,
        )
