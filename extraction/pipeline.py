"""Extraction pipeline: orchestrates text extraction from a PDFDocument.

Takes a fully preprocessed + OCR'd ``PDFDocument`` and returns a structured
``ExtractionResult`` with patient info, test rows, sections, and remarks.
The interpretation (status classification, explanations) is applied later by
the inference module.
"""

from __future__ import annotations

from typing import Optional

from pathology_ai.preprocessing.document import PDFDocument
from pathology_ai.extraction.schema import ExtractionResult, ExtractedSection
from pathology_ai.extraction.text_extractor import (
    build_extracted_section,
    extract_document_category,
    extract_laboratory_name,
    extract_panel_name,
    extract_patient_info,
    extract_remarks,
    extract_test_rows,
)


from pathology_ai.extraction.ml_extractor import MLExtractor


class ExtractionPipeline:
    """Orchestrates structured field extraction from a preprocessed PDFDocument.

    Usage::

        pipeline = ExtractionPipeline()
        result = pipeline.extract(document)
    """

    def __init__(self, use_ml_model: bool = True):
        self.ml_extractor = None
        if use_ml_model:
            try:
                self.ml_extractor = MLExtractor()
                print("[ExtractionPipeline] Loaded ML Transformer model for NER extraction.")
            except Exception:
                # Silently fall back to pure rule-based extractor if model weights aren't present
                self.ml_extractor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, document: PDFDocument) -> ExtractionResult:
        """Extract structured information from a ``PDFDocument``.

        Selects the best text source for each page (digital text preferred;
        falls back to OCR processed text).  Concatenates all pages into a
        single text body and runs the extraction pipeline.

        Args:
            document: A ``PDFDocument`` produced by ``PDFParser``, optionally
                with OCR results attached.

        Returns:
            ``ExtractionResult`` with all extracted fields populated.
        """
        full_text = self._get_full_text(document)

        category = extract_document_category(full_text)
        laboratory_name = extract_laboratory_name(full_text)
        panel_name = extract_panel_name(full_text)
        patient = extract_patient_info(full_text)
        remarks = extract_remarks(full_text)

        rows = extract_test_rows(full_text)

        # Ensemble with fine-tuned BERT Transformer NER model if available
        if self.ml_extractor is not None:
            try:
                ml_entities = self.ml_extractor.extract_entities(full_text)
                # 1. Backfill patient name / ID if rule-based extractor missed them
                if not patient.name and ml_entities.get("PATIENT_NAME"):
                    patient.name = ml_entities["PATIENT_NAME"][0]
                if not patient.patient_id and ml_entities.get("PATIENT_ID"):
                    patient.patient_id = ml_entities["PATIENT_ID"][0]

                # 2. Backfill missing test rows if rule-based extractor found none
                if len(rows) == 0 and ml_entities.get("TEST_NAME") and ml_entities.get("VALUE"):
                    test_names = ml_entities["TEST_NAME"]
                    values = ml_entities["VALUE"]
                    units = ml_entities.get("UNIT", [])
                    ref_ranges = ml_entities.get("REF_RANGE", [])
                    for idx, t_name in enumerate(test_names):
                        val_str = values[idx] if idx < len(values) else ""
                        unit_str = units[idx] if idx < len(units) else ""
                        ref_str = ref_ranges[idx] if idx < len(ref_ranges) else ""
                        if val_str:
                            rows.append({
                                "name": t_name,
                                "value": val_str,
                                "unit": unit_str,
                                "reference_range": ref_str,
                                "flag": "",
                            })
            except Exception as e:
                print(f"[ExtractionPipeline] ML ensemble notice: {e}")

        sections = self._group_into_sections(rows, full_text)

        all_tests = [t for s in sections for t in s.tests]
        extraction_confidence = self._compute_overall_confidence(all_tests, rows)

        result = ExtractionResult(
            document_category=category,
            panel_name=panel_name,
            laboratory_name=laboratory_name,
            patient=patient,
            sections=sections,
            remarks=remarks,
            extraction_confidence=extraction_confidence,
            requires_review=(extraction_confidence < 0.6),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_full_text(document: PDFDocument) -> str:
        """Concatenate best available text from every page."""
        parts = []
        for page in document.pages:
            # Prefer digital text (higher fidelity); fall back to OCR
            text = page.text.strip()
            if not text:
                text = (page.ocr_processed_text or page.ocr_text or "").strip()
            if text:
                parts.append(f"[PAGE {page.page_number}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _group_into_sections(rows, full_text: str):
        """Group extracted test rows under a single section.

        For now all rows are placed in one section named after the panel or a
        generic "Investigation Results" heading.  Future iterations can segment
        by section headers detected in the text.
        """
        if not rows:
            return []

        # Try to infer a meaningful section name from the text
        section_name = "Investigation Results"
        for candidate in ["INVESTIGATION RESULTS", "TEST RESULTS", "REPORT RESULTS",
                          "BIOCHEMISTRY RESULTS", "HAEMATOLOGY RESULTS"]:
            if candidate in full_text.upper():
                section_name = candidate.title()
                break

        section = build_extracted_section(section_name, rows)
        return [section] if section.tests else []

    @staticmethod
    def _compute_overall_confidence(all_tests, raw_rows) -> float:
        """Compute an overall extraction confidence score.

        A simple heuristic:
        - If no rows were extracted at all: 0.0
        - Average confidence across extracted tests
        - Penalise if we lost rows (rows detected but not converted to tests)
        """
        if not raw_rows:
            return 0.0
        if not all_tests:
            return 0.1

        avg_test_confidence = sum(t.confidence for t in all_tests) / len(all_tests)

        # Penalise row-drop rate
        conversion_rate = len(all_tests) / len(raw_rows)
        score = avg_test_confidence * (0.5 + 0.5 * conversion_rate)
        return round(max(0.0, min(1.0, score)), 4)
