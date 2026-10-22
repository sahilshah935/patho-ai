"""Dataclasses for structured extraction output.

These mirror the ground-truth annotation schema produced by the dataset
generator, so that extraction results can be directly compared against
annotations for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-field extracted entities
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPatient:
    name: str = ""
    patient_id: str = ""
    age: Optional[int] = None
    sex: str = ""
    sample_id: str = ""
    accession_number: str = ""
    referred_by: str = ""
    sample_type: str = ""
    collection_datetime: str = ""
    received_datetime: str = ""
    report_datetime: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "patient_id": self.patient_id,
            "age": self.age,
            "sex": self.sex,
            "sample_id": self.sample_id,
            "accession_number": self.accession_number,
            "referred_by": self.referred_by,
            "sample_type": self.sample_type,
            "collection_datetime": self.collection_datetime,
            "received_datetime": self.received_datetime,
            "report_datetime": self.report_datetime,
        }


@dataclass
class ExtractedTest:
    # Raw values as they appear in the document
    test_name_raw: str = ""
    value_raw: str = ""
    unit_raw: str = ""
    reference_range_raw: str = ""
    flag_raw: str = ""

    # Normalised / parsed values
    test_name_normalized: str = ""
    value_numeric: Optional[float] = None
    unit_normalized: str = ""
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None

    # Interpretation (filled by InferencePipeline)
    status: str = ""          # NORMAL / HIGH / LOW / CRITICAL_HIGH / CRITICAL_LOW / BORDERLINE_HIGH / BORDERLINE_LOW / UNKNOWN
    flag: str = ""            # H / L / H* / L* / ""
    critical: bool = False

    # Meta
    confidence: float = 1.0
    explanation: str = ""     # filled by Explainer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name_raw": self.test_name_raw,
            "test_name_normalized": self.test_name_normalized,
            "value_raw": self.value_raw,
            "value_numeric": self.value_numeric,
            "unit_raw": self.unit_raw,
            "unit_normalized": self.unit_normalized,
            "reference_range_raw": self.reference_range_raw,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "reference_range_display": self.reference_range_display,
            "status": self.status,
            "flag": self.flag,
            "critical": self.critical,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    @property
    def reference_range_display(self) -> str:
        if self.reference_low is not None and self.reference_high is not None:
            return f"{self._fmt(self.reference_low)} – {self._fmt(self.reference_high)}"
        elif self.reference_high is not None:
            return f"< {self._fmt(self.reference_high)}"
        elif self.reference_low is not None:
            return f"> {self._fmt(self.reference_low)}"
        return self.reference_range_raw or "-"

    @staticmethod
    def _fmt(val: float) -> str:
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}".rstrip("0").rstrip(".")


@dataclass
class ExtractedSection:
    section_name: str = ""
    tests: List[ExtractedTest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_name": self.section_name,
            "tests": [t.to_dict() for t in self.tests],
        }


# ---------------------------------------------------------------------------
# Top-level extraction result
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Output of the ExtractionPipeline for one document."""

    document_category: str = ""         # Hematology / Biochemistry / etc.
    panel_name: str = ""                # e.g. "COMPLETE BLOOD COUNT (CBC)"
    laboratory_name: str = ""
    patient: ExtractedPatient = field(default_factory=ExtractedPatient)
    sections: List[ExtractedSection] = field(default_factory=list)
    remarks: List[str] = field(default_factory=list)

    # Quality indicators
    extraction_confidence: float = 1.0  # 0–1 overall confidence
    requires_review: bool = False       # True when any critical value OR low confidence

    # Summary (filled by Explainer)
    summary: str = ""

    def all_tests(self) -> List[ExtractedTest]:
        """Flat list of every test across all sections."""
        return [t for s in self.sections for t in s.tests]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_category": self.document_category,
            "panel_name": self.panel_name,
            "laboratory_name": self.laboratory_name,
            "patient": self.patient.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "remarks": self.remarks,
            "extraction_confidence": self.extraction_confidence,
            "requires_review": self.requires_review,
            "summary": self.summary,
        }
