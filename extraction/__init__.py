"""pathology_ai.extraction — structured field extraction from pathology reports."""

from pathology_ai.extraction.schema import (
    ExtractionResult,
    ExtractedPatient,
    ExtractedSection,
    ExtractedTest,
)
from pathology_ai.extraction.pipeline import ExtractionPipeline

__all__ = [
    "ExtractionPipeline",
    "ExtractionResult",
    "ExtractedPatient",
    "ExtractedSection",
    "ExtractedTest",
]
