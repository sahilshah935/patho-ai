"""Dataset loader for the pathology_dataset directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

from pathology_ai.configs.paths import DATASET_ROOT


# ---------------------------------------------------------------------------
# Typed dataclasses matching the annotation schema
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedPatient:
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
    contact: str = ""
    pregnancy_status: Optional[str] = None
    menstrual_phase: Optional[str] = None
    menopausal_status: Optional[str] = None


@dataclass
class AnnotatedTest:
    test_name_raw: str = ""
    test_name_normalized: str = ""
    value_raw: str = ""
    value_numeric: Optional[float] = None
    unit_raw: str = ""
    unit_normalized: str = ""
    reference_range_raw: str = ""
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    status: str = ""
    flag: str = ""
    critical: bool = False
    confidence: float = 1.0


@dataclass
class AnnotatedSection:
    section_name: str = ""
    tests: List[AnnotatedTest] = field(default_factory=list)


@dataclass
class AnnotationRecord:
    report_id: str
    category: str
    format: str  # "digital" or "scanned"
    laboratory_id: str
    template_id: str
    panel_name: str
    split: str
    difficulty: str
    patient: AnnotatedPatient
    sections: List[AnnotatedSection]
    remarks: List[str]
    interpretation: str
    ground_truth_text: str
    ocr_ground_truth: Optional[Any]
    # Raw source dict for fields not captured in typed fields
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRecord":
        patient_data = data.get("patient", {})
        patient = AnnotatedPatient(
            name=patient_data.get("name", ""),
            patient_id=patient_data.get("patient_id", ""),
            age=patient_data.get("age"),
            sex=patient_data.get("sex", ""),
            sample_id=patient_data.get("sample_id", ""),
            accession_number=patient_data.get("accession_number", ""),
            referred_by=patient_data.get("referred_by", ""),
            sample_type=patient_data.get("sample_type", ""),
            collection_datetime=patient_data.get("collection_datetime", ""),
            received_datetime=patient_data.get("received_datetime", ""),
            report_datetime=patient_data.get("report_datetime", ""),
            contact=patient_data.get("contact", ""),
            pregnancy_status=patient_data.get("pregnancy_status"),
            menstrual_phase=patient_data.get("menstrual_phase"),
            menopausal_status=patient_data.get("menopausal_status"),
        )

        sections = []
        for sec_data in data.get("sections", []):
            tests = []
            for t in sec_data.get("tests", []):
                tests.append(
                    AnnotatedTest(
                        test_name_raw=t.get("test_name_raw", ""),
                        test_name_normalized=t.get("test_name_normalized", ""),
                        value_raw=t.get("value_raw", ""),
                        value_numeric=t.get("value_numeric"),
                        unit_raw=t.get("unit_raw", ""),
                        unit_normalized=t.get("unit_normalized", ""),
                        reference_range_raw=t.get("reference_range_raw", ""),
                        reference_low=t.get("reference_low"),
                        reference_high=t.get("reference_high"),
                        status=t.get("status", ""),
                        flag=t.get("flag", ""),
                        critical=t.get("critical", False),
                        confidence=t.get("confidence", 1.0),
                    )
                )
            sections.append(
                AnnotatedSection(
                    section_name=sec_data.get("section_name", ""),
                    tests=tests,
                )
            )

        return cls(
            report_id=data.get("report_id", ""),
            category=data.get("category", ""),
            format=data.get("format", "digital"),
            laboratory_id=data.get("laboratory_id", ""),
            template_id=data.get("template_id", ""),
            panel_name=data.get("panel_name", ""),
            split=data.get("split", ""),
            difficulty=data.get("difficulty", "easy"),
            patient=patient,
            sections=sections,
            remarks=data.get("remarks", []),
            interpretation=data.get("interpretation", ""),
            ground_truth_text=data.get("ground_truth_text", ""),
            ocr_ground_truth=data.get("ocr_ground_truth"),
            _raw=data,
        )

    def all_tests(self) -> List[AnnotatedTest]:
        """Flat list of every test across all sections."""
        return [test for section in self.sections for test in section.tests]

    def pdf_path(self, dataset_root: Optional[Path] = None) -> Path:
        """Resolve the path to this report's PDF file."""
        root = Path(dataset_root) if dataset_root else DATASET_ROOT
        category_slug = self.category.lower().replace(" / ", "_").replace(" ", "_")
        return root / "raw_documents" / self.format / category_slug / f"{self.report_id}.pdf"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

VALID_SPLITS = {"train", "validation", "standard_test", "hard_generalization_test"}
VALID_CATEGORIES = {
    "Hematology",
    "Biochemistry",
    "Urinalysis",
    "Endocrinology",
    "Serology / Immunology",
}


class AnnotationLoader:
    """Lightweight loader for the pathology_dataset annotations directory.

    Example usage::

        loader = AnnotationLoader()
        record = loader.load("RPT000001")
        for record in loader.iterate(split="train", category="Hematology", limit=100):
            ...
    """

    def __init__(self, dataset_root: Optional[Path] = None):
        self.dataset_root = Path(dataset_root) if dataset_root else DATASET_ROOT
        self.annotations_root = self.dataset_root / "annotations"

        if not self.annotations_root.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {self.annotations_root}"
            )

    # ------------------------------------------------------------------
    # Single record access
    # ------------------------------------------------------------------

    def load(self, report_id: str) -> AnnotationRecord:
        """Load a single annotation record by report_id, searching all splits."""
        for split_dir in sorted(self.annotations_root.iterdir()):
            if not split_dir.is_dir():
                continue
            candidate = split_dir / f"{report_id}.json"
            if candidate.exists():
                return self._load_file(candidate)
        raise FileNotFoundError(
            f"Annotation not found for report_id={report_id!r} in {self.annotations_root}"
        )

    def load_from_path(self, path: Path) -> AnnotationRecord:
        """Load a single annotation record from an explicit path."""
        return self._load_file(Path(path))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iterate(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
        format: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[AnnotationRecord]:
        """Iterate over annotation records with optional filters.

        Args:
            split: One of ``train``, ``validation``, ``standard_test``,
                ``hard_generalization_test``, or ``None`` for all.
            category: Category name filter, or ``None`` for all.
            format: ``"digital"`` or ``"scanned"``, or ``None`` for all.
            difficulty: ``"easy"``, ``"moderate"``, ``"difficult"``, ``"hard"``.
            limit: Maximum number of records to yield.
        """
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r}. Choose from {sorted(VALID_SPLITS)}")

        splits_to_search = (
            [self.annotations_root / split]
            if split
            else sorted(self.annotations_root.iterdir())
        )

        yielded = 0
        for split_dir in splits_to_search:
            if not split_dir.is_dir():
                continue
            for json_path in sorted(split_dir.glob("*.json")):
                if limit is not None and yielded >= limit:
                    return
                record = self._load_file(json_path)
                if category is not None and record.category != category:
                    continue
                if format is not None and record.format != format:
                    continue
                if difficulty is not None and record.difficulty != difficulty:
                    continue
                yield record
                yielded += 1

    def count(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        """Count annotation files matching the given filters."""
        return sum(1 for _ in self.iterate(split=split, category=category))

    # ------------------------------------------------------------------
    # Master index (fast lookup, no JSON parsing)
    # ------------------------------------------------------------------

    def iter_index(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate the master_dataset_index.jsonl for fast metadata scanning
        without loading full annotation JSON files."""
        index_path = self.dataset_root / "metadata" / "master_dataset_index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Master index not found: {index_path}")
        with index_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> AnnotationRecord:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AnnotationRecord.from_dict(data)
