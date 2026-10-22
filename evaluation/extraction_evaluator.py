"""Evaluate extraction results against ground-truth annotations.

Compares an ``ExtractionResult`` (or a prediction JSON produced by
``predict.py --extract``) against the corresponding annotation JSON from
the dataset, computing field-level and document-level accuracy metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pathology_ai.configs.paths import DATASET_ROOT
from pathology_ai.datasets.loader import AnnotationLoader, AnnotationRecord, AnnotatedTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _str_match(pred: str, ref: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return pred.strip().lower() == ref.strip().lower()


def _numeric_match(pred: Optional[float], ref: Optional[float], tol: float = 1e-4) -> bool:
    """Numeric match within tolerance."""
    if pred is None and ref is None:
        return True
    if pred is None or ref is None:
        return False
    return abs(pred - ref) <= tol


# ---------------------------------------------------------------------------
# Patient field evaluation
# ---------------------------------------------------------------------------

PATIENT_FIELDS = ["name", "age", "sex", "patient_id", "accession_number",
                  "sample_id", "referred_by", "sample_type"]


def evaluate_patient(pred_patient: Dict, gt_patient: Dict) -> Dict[str, Any]:
    """Evaluate extracted patient fields against ground truth."""
    results = {}
    correct = 0
    total = 0

    for field in PATIENT_FIELDS:
        pred_val = str(pred_patient.get(field, "") or "").strip()
        gt_val = str(gt_patient.get(field, "") or "").strip()

        if not gt_val:  # skip fields that are empty in ground truth
            continue

        total += 1
        match = _str_match(pred_val, gt_val)
        if match:
            correct += 1

        results[field] = {
            "predicted": pred_val,
            "ground_truth": gt_val,
            "match": match,
        }

    accuracy = correct / total if total > 0 else None
    return {"fields": results, "accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Test-level evaluation
# ---------------------------------------------------------------------------

def _tests_by_name(tests: List[Dict]) -> Dict[str, Dict]:
    """Index test list by normalised name for O(1) lookup."""
    index = {}
    for t in tests:
        key = t.get("test_name_normalized") or t.get("test_name_raw", "")
        if key:
            index[key.strip().lower()] = t
    return index


def evaluate_tests(
    pred_tests: List[Dict],
    gt_tests: List[Dict],
) -> Dict[str, Any]:
    """Evaluate extracted test results against ground-truth tests.

    Matches by normalised test name. Computes per-test and aggregate metrics.

    Returns:
        Dict with per-test details and aggregate precision / recall / F1
        for name, value, unit, and status matching.
    """
    pred_index = _tests_by_name(pred_tests)
    gt_index = _tests_by_name(gt_tests)

    matched_names = set(pred_index.keys()) & set(gt_index.keys())
    only_pred = set(pred_index.keys()) - set(gt_index.keys())
    only_gt = set(gt_index.keys()) - set(pred_index.keys())

    per_test = []
    value_matches = 0
    unit_matches = 0
    status_matches = 0
    total_matched = len(matched_names)

    for name in matched_names:
        pred_t = pred_index[name]
        gt_t = gt_index[name]

        val_match = _numeric_match(pred_t.get("value_numeric"), gt_t.get("value_numeric"))
        unit_match = _str_match(
            pred_t.get("unit_normalized") or pred_t.get("unit_raw", ""),
            gt_t.get("unit_normalized") or gt_t.get("unit_raw", ""),
        )
        status_match = _str_match(pred_t.get("status", ""), gt_t.get("status", ""))

        if val_match:
            value_matches += 1
        if unit_match:
            unit_matches += 1
        if status_match:
            status_matches += 1

        per_test.append({
            "test_name": name,
            "value_match": val_match,
            "unit_match": unit_match,
            "status_match": status_match,
            "predicted_value": pred_t.get("value_raw"),
            "ground_truth_value": gt_t.get("value_raw"),
            "predicted_status": pred_t.get("status"),
            "ground_truth_status": gt_t.get("status"),
        })

    precision = total_matched / len(pred_index) if pred_index else 0.0
    recall = total_matched / len(gt_index) if gt_index else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "per_test": per_test,
        "total_predicted": len(pred_index),
        "total_ground_truth": len(gt_index),
        "matched": total_matched,
        "only_in_prediction": sorted(only_pred),
        "only_in_ground_truth": sorted(only_gt),
        "name_precision": round(precision, 4),
        "name_recall": round(recall, 4),
        "name_f1": round(f1, 4),
        "value_accuracy": round(value_matches / total_matched, 4) if total_matched else None,
        "unit_accuracy": round(unit_matches / total_matched, 4) if total_matched else None,
        "status_accuracy": round(status_matches / total_matched, 4) if total_matched else None,
    }


# ---------------------------------------------------------------------------
# Document-level evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction_prediction(
    prediction_path,
    ground_truth_path,
) -> Dict[str, Any]:
    """Evaluate a single prediction JSON against its ground-truth annotation.

    Args:
        prediction_path: Path to the prediction JSON produced by ``predict.py``.
        ground_truth_path: Path to the annotation JSON from the dataset.

    Returns:
        Dict with patient and test evaluation metrics.
    """
    pred = _load_json(prediction_path)
    gt = _load_json(ground_truth_path)

    # Extract the 'extraction' sub-key if predict.py format
    extraction = pred.get("extraction") or pred

    pred_patient = extraction.get("patient", {})
    gt_patient = gt.get("patient", {})

    # Flatten pred tests from all sections
    pred_tests: List[Dict] = []
    for section in extraction.get("sections", []):
        pred_tests.extend(section.get("tests", []))

    # Flatten gt tests from all sections
    gt_tests: List[Dict] = []
    for section in gt.get("sections", []):
        gt_tests.extend(section.get("tests", []))

    patient_metrics = evaluate_patient(pred_patient, gt_patient)
    test_metrics = evaluate_tests(pred_tests, gt_tests)

    return {
        "report_id": gt.get("report_id", Path(ground_truth_path).stem),
        "prediction_path": str(Path(prediction_path).resolve()),
        "ground_truth_path": str(Path(ground_truth_path).resolve()),
        "category": gt.get("category", ""),
        "format": gt.get("format", ""),
        "patient": patient_metrics,
        "tests": test_metrics,
        "extraction_confidence": extraction.get("extraction_confidence"),
        "requires_review": extraction.get("requires_review"),
    }


def find_annotation(report_id: str, dataset_root=None) -> Path:
    """Locate the annotation JSON for a given report_id in any split."""
    root = Path(dataset_root) if dataset_root else DATASET_ROOT
    loader = AnnotationLoader(root)
    record = loader.load(report_id)
    # Reconstruct the annotation file path
    annotations_root = root / "annotations"
    for split_dir in annotations_root.iterdir():
        candidate = split_dir / f"{report_id}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Annotation file not found for {report_id}")


# ---------------------------------------------------------------------------
# Directory-level evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction_directory(
    predictions_dir,
    dataset_root=None,
) -> Dict[str, Any]:
    """Evaluate all extraction prediction JSONs in a directory.

    Args:
        predictions_dir: Directory containing prediction JSON files from ``predict.py``.
        dataset_root: Root of the pathology_dataset directory.

    Returns:
        Aggregate and per-report evaluation results.
    """
    predictions_dir = Path(predictions_dir)
    prediction_paths = sorted(predictions_dir.glob("*.json"))

    reports = []
    skipped = []

    for pred_path in prediction_paths:
        report_id = pred_path.stem
        try:
            gt_path = find_annotation(report_id, dataset_root)
            reports.append(evaluate_extraction_prediction(pred_path, gt_path))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append({"prediction_path": str(pred_path.resolve()), "reason": str(exc)})

    n = len(reports)

    def _avg(key_path):
        vals = []
        for r in reports:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "predictions_dir": str(predictions_dir.resolve()),
        "reports_evaluated": n,
        "reports_skipped": len(skipped),
        "aggregate": {
            "avg_patient_accuracy": _avg(["patient", "accuracy"]),
            "avg_name_precision": _avg(["tests", "name_precision"]),
            "avg_name_recall": _avg(["tests", "name_recall"]),
            "avg_name_f1": _avg(["tests", "name_f1"]),
            "avg_value_accuracy": _avg(["tests", "value_accuracy"]),
            "avg_unit_accuracy": _avg(["tests", "unit_accuracy"]),
            "avg_status_accuracy": _avg(["tests", "status_accuracy"]),
        },
        "reports": reports,
        "skipped": skipped,
    }
