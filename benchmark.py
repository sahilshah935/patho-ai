"""Phase 2 Benchmark: Digital vs. Scanned PDF Extraction Accuracy.

Runs the full pipeline (PDF → OCR → Extraction → Interpretation) on a sample
of reports from the dataset and compares extraction results against
ground-truth annotations.

Produces a JSON benchmark report with per-category and per-format metrics.

Usage:
    python pathology_ai/benchmark.py --sample 20 --output pathology_ai/outputs/evaluation/benchmark_report.json
    python pathology_ai/benchmark.py --sample 50 --categories Hematology Biochemistry
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathology_ai.configs.paths import DATASET_ROOT, EVALUATION_DIR, RENDERED_PAGES_DIR
from pathology_ai.datasets.loader import AnnotationLoader
from pathology_ai.preprocessing.pdf_parser import PDFParser
from pathology_ai.ocr.engines import create_ocr_engine
from pathology_ai.ocr.pipeline import apply_ocr
from pathology_ai.extraction.pipeline import ExtractionPipeline
from pathology_ai.inference.interpreter import Interpreter
from pathology_ai.inference.explainer import explain_and_summarise


# ---------------------------------------------------------------------------
# Evaluation helpers (inline to keep the script self-contained)
# ---------------------------------------------------------------------------

def _str_match(pred: str, ref: str) -> bool:
    return pred.strip().lower() == ref.strip().lower()


def _numeric_match(pred, ref, tol=0.05) -> bool:
    if pred is None and ref is None:
        return True
    if pred is None or ref is None:
        return False
    return abs(float(pred) - float(ref)) <= tol


def evaluate_single(extraction_dict: Dict, annotation: Dict) -> Dict[str, Any]:
    """Compare a single extraction result against its ground-truth annotation."""
    # Flatten predicted tests
    pred_tests = []
    for section in extraction_dict.get("sections", []):
        pred_tests.extend(section.get("tests", []))

    # Flatten ground-truth tests
    gt_tests = []
    for section in annotation.get("sections", []):
        gt_tests.extend(section.get("tests", []))

    # Index by normalised name
    pred_by_name = {}
    for t in pred_tests:
        key = (t.get("test_name_normalized") or t.get("test_name_raw", "")).strip().lower()
        if key:
            pred_by_name[key] = t

    gt_by_name = {}
    for t in gt_tests:
        key = (t.get("test_name_normalized") or t.get("test_name_raw", "")).strip().lower()
        if key:
            gt_by_name[key] = t

    matched = set(pred_by_name.keys()) & set(gt_by_name.keys())
    total_matched = len(matched)
    total_pred = len(pred_by_name)
    total_gt = len(gt_by_name)

    value_matches = 0
    status_matches = 0
    unit_matches = 0

    for name in matched:
        pt = pred_by_name[name]
        gt = gt_by_name[name]

        if _numeric_match(pt.get("value_numeric"), gt.get("value_numeric")):
            value_matches += 1
        if _str_match(pt.get("status", ""), gt.get("status", "")):
            status_matches += 1
        if _str_match(
            pt.get("unit_normalized") or pt.get("unit_raw", ""),
            gt.get("unit_normalized") or gt.get("unit_raw", ""),
        ):
            unit_matches += 1

    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Patient field accuracy
    pred_patient = extraction_dict.get("patient", {})
    gt_patient = annotation.get("patient", {})
    patient_fields = ["name", "age", "sex", "patient_id"]
    patient_correct = 0
    patient_total = 0
    for field in patient_fields:
        gt_val = str(gt_patient.get(field, "") or "").strip()
        if not gt_val:
            continue
        patient_total += 1
        pred_val = str(pred_patient.get(field, "") or "").strip()
        if _str_match(pred_val, gt_val):
            patient_correct += 1

    return {
        "total_predicted": total_pred,
        "total_ground_truth": total_gt,
        "matched": total_matched,
        "name_precision": round(precision, 4),
        "name_recall": round(recall, 4),
        "name_f1": round(f1, 4),
        "value_accuracy": round(value_matches / total_matched, 4) if total_matched else None,
        "unit_accuracy": round(unit_matches / total_matched, 4) if total_matched else None,
        "status_accuracy": round(status_matches / total_matched, 4) if total_matched else None,
        "patient_accuracy": round(patient_correct / patient_total, 4) if patient_total else None,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    sample_per_format: int = 10,
    categories: Optional[List[str]] = None,
    split: str = "validation",
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Run the full pipeline on a sample of digital + scanned PDFs and measure accuracy."""

    loader = AnnotationLoader()

    if categories is None:
        categories = ["Hematology", "Biochemistry", "Urinalysis", "Endocrinology", "Serology / Immunology"]

    # Initialize pipeline components
    parser = PDFParser(output_dir=RENDERED_PAGES_DIR, dpi=300, render_pages=True)
    ocr_engine = create_ocr_engine(name="easyocr", languages=["en"], gpu=use_gpu)
    extraction_pipeline = ExtractionPipeline()
    interpreter = Interpreter()

    results = {
        "digital": [],
        "scanned": [],
    }
    errors = []

    for fmt in ["digital", "scanned"]:
        print(f"\n{'='*60}")
        print(f"  Benchmarking {fmt.upper()} PDFs")
        print(f"{'='*60}")

        # In this dataset, digital PDFs are in 'train', scanned in 'validation'/'standard_test'
        target_split = "train" if fmt == "digital" else split

        count = 0
        for record in loader.iterate(split=target_split, format=fmt):
            if categories and record.category not in categories:
                continue
            if count >= sample_per_format:
                break

            report_id = record.report_id
            pdf_path = record.pdf_path()

            if not pdf_path.exists():
                errors.append({"report_id": report_id, "error": f"PDF not found: {pdf_path}"})
                continue

            print(f"  [{fmt}] {report_id} ({record.category})...", end=" ", flush=True)
            t0 = time.time()

            try:
                # Full pipeline
                document = parser.parse(pdf_path)
                apply_ocr(document, ocr_engine, force=(fmt == "scanned"))
                extraction_result = extraction_pipeline.extract(document)
                interpretation_result = interpreter.interpret(extraction_result)
                interpretation_result = explain_and_summarise(interpretation_result)

                extraction_dict = interpretation_result.extraction.to_dict()

                # Compare against ground truth
                metrics = evaluate_single(extraction_dict, record._raw)

                elapsed = round(time.time() - t0, 2)
                metrics["report_id"] = report_id
                metrics["category"] = record.category
                metrics["format"] = fmt
                metrics["difficulty"] = record.difficulty
                metrics["elapsed_seconds"] = elapsed
                metrics["extraction_confidence"] = extraction_dict.get("extraction_confidence")

                results[fmt].append(metrics)
                count += 1

                f1_str = f"F1={metrics['name_f1']:.2f}"
                val_str = f"Val={metrics['value_accuracy']:.2f}" if metrics['value_accuracy'] is not None else "Val=N/A"
                stat_str = f"Stat={metrics['status_accuracy']:.2f}" if metrics['status_accuracy'] is not None else "Stat=N/A"
                print(f"[OK] {f1_str} {val_str} {stat_str} ({elapsed}s)")

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                print(f"[FAIL] ERROR ({elapsed}s): {e}")
                errors.append({"report_id": report_id, "format": fmt, "error": str(e), "traceback": traceback.format_exc()})
                count += 1

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    def _aggregate(report_list):
        if not report_list:
            return {}
        n = len(report_list)

        def _avg(key):
            vals = [r[key] for r in report_list if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "count": n,
            "avg_name_precision": _avg("name_precision"),
            "avg_name_recall": _avg("name_recall"),
            "avg_name_f1": _avg("name_f1"),
            "avg_value_accuracy": _avg("value_accuracy"),
            "avg_unit_accuracy": _avg("unit_accuracy"),
            "avg_status_accuracy": _avg("status_accuracy"),
            "avg_patient_accuracy": _avg("patient_accuracy"),
            "avg_extraction_confidence": _avg("extraction_confidence"),
            "avg_elapsed_seconds": _avg("elapsed_seconds"),
        }

    # Per-category breakdowns
    all_categories = set()
    for fmt in ["digital", "scanned"]:
        for r in results[fmt]:
            all_categories.add(r["category"])

    per_category = {}
    for cat in sorted(all_categories):
        per_category[cat] = {
            "digital": _aggregate([r for r in results["digital"] if r["category"] == cat]),
            "scanned": _aggregate([r for r in results["scanned"] if r["category"] == cat]),
        }

    report = {
        "benchmark_config": {
            "sample_per_format": sample_per_format,
            "split": split,
            "categories": categories,
        },
        "aggregate": {
            "digital": _aggregate(results["digital"]),
            "scanned": _aggregate(results["scanned"]),
        },
        "per_category": per_category,
        "per_report": {
            "digital": results["digital"],
            "scanned": results["scanned"],
        },
        "errors": errors,
    }

    return report


def print_summary(report: Dict):
    """Print a human-readable summary of the benchmark report."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    for fmt in ["digital", "scanned"]:
        agg = report["aggregate"].get(fmt, {})
        if not agg:
            continue
        print(f"\n  {fmt.upper()} ({agg.get('count', 0)} reports)")
        print(f"  {'-' * 40}")
        print(f"  Test Name F1:        {agg.get('avg_name_f1', 'N/A')}")
        print(f"  Value Accuracy:      {agg.get('avg_value_accuracy', 'N/A')}")
        print(f"  Unit Accuracy:       {agg.get('avg_unit_accuracy', 'N/A')}")
        print(f"  Status Accuracy:     {agg.get('avg_status_accuracy', 'N/A')}")
        print(f"  Patient Accuracy:    {agg.get('avg_patient_accuracy', 'N/A')}")
        print(f"  Extraction Conf.:    {agg.get('avg_extraction_confidence', 'N/A')}")
        print(f"  Avg Time:            {agg.get('avg_elapsed_seconds', 'N/A')}s")

    # Comparison
    dig = report["aggregate"].get("digital", {})
    scn = report["aggregate"].get("scanned", {})
    if dig and scn:
        print(f"\n  {'-' * 40}")
        print(f"  ACCURACY DROP (Digital -> Scanned)")
        print(f"  {'-' * 40}")
        for metric in ["avg_name_f1", "avg_value_accuracy", "avg_status_accuracy"]:
            d = dig.get(metric)
            s = scn.get(metric)
            if d is not None and s is not None:
                drop = round((d - s) * 100, 1)
                label = metric.replace("avg_", "").replace("_", " ").title()
                print(f"  {label:<22} {d:.4f} -> {s:.4f}  (Delta {drop:+.1f}%)")

    if report.get("errors"):
        print(f"\n  [!] {len(report['errors'])} error(s) encountered. See the JSON report for details.")

    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark extraction pipeline on digital vs. scanned PDFs.")
    parser.add_argument("--sample", type=int, default=10, help="Number of reports to test per format (digital/scanned).")
    parser.add_argument("--split", default="validation", help="Dataset split to benchmark against.")
    parser.add_argument("--categories", nargs="*", help="Specific categories to benchmark. Defaults to all.")
    parser.add_argument("--cpu-ocr", action="store_true", help="Run OCR on CPU instead of GPU.")
    parser.add_argument("--output", help="Output JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()

    report = run_benchmark(
        sample_per_format=args.sample,
        categories=args.categories,
        split=args.split,
        use_gpu=not args.cpu_ocr,
    )

    print_summary(report)

    output_path = Path(args.output) if args.output else EVALUATION_DIR / "benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Full report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
