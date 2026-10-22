import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathology_ai.configs.paths import EVALUATION_DIR, PREDICTIONS_DIR
from pathology_ai.evaluation.ocr_evaluator import evaluate_ocr_directory, evaluate_ocr_prediction
from pathology_ai.evaluation.extraction_evaluator import (
    evaluate_extraction_prediction,
    evaluate_extraction_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pathology AI predictions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # OCR evaluation sub-commands (unchanged)
    # ------------------------------------------------------------------
    ocr_parser = subparsers.add_parser("ocr", help="Evaluate OCR prediction JSON.")
    ocr_parser.add_argument("--prediction", required=True, help="Prediction JSON path.")
    ocr_parser.add_argument("--ground-truth", required=True, help="OCR ground-truth JSON path.")
    ocr_parser.add_argument(
        "--output",
        help="Optional output metrics JSON path. Defaults to outputs/evaluation/<report_id>_ocr_metrics.json.",
    )

    ocr_dir_parser = subparsers.add_parser(
        "ocr-dir",
        help="Evaluate every prediction JSON in a directory against OCR ground truth.",
    )
    ocr_dir_parser.add_argument(
        "--predictions-dir",
        default=str(PREDICTIONS_DIR),
        help="Directory containing prediction JSON files.",
    )
    ocr_dir_parser.add_argument(
        "--ground-truth-root",
        help="Root directory containing OCR ground-truth category folders.",
    )
    ocr_dir_parser.add_argument(
        "--output",
        help="Optional output metrics JSON path. Defaults to outputs/evaluation/ocr_directory_metrics.json.",
    )

    # ------------------------------------------------------------------
    # Extraction evaluation sub-commands (new)
    # ------------------------------------------------------------------
    ext_parser = subparsers.add_parser(
        "extraction",
        help="Evaluate a single extraction prediction JSON against its annotation.",
    )
    ext_parser.add_argument("--prediction", required=True, help="Prediction JSON path (from predict.py --extract).")
    ext_parser.add_argument("--ground-truth", required=True, help="Annotation JSON path from the dataset.")
    ext_parser.add_argument(
        "--output",
        help="Optional output metrics JSON path. Defaults to outputs/evaluation/<report_id>_extraction_metrics.json.",
    )

    ext_dir_parser = subparsers.add_parser(
        "extraction-dir",
        help="Evaluate all extraction prediction JSONs in a directory.",
    )
    ext_dir_parser.add_argument(
        "--predictions-dir",
        default=str(PREDICTIONS_DIR),
        help="Directory containing prediction JSON files.",
    )
    ext_dir_parser.add_argument(
        "--dataset-root",
        help="Root of the pathology_dataset directory. Defaults to the configured DATASET_ROOT.",
    )
    ext_dir_parser.add_argument(
        "--output",
        help="Optional output metrics JSON path. Defaults to outputs/evaluation/extraction_directory_metrics.json.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # OCR — single
    # ------------------------------------------------------------------
    if args.command == "ocr":
        metrics = evaluate_ocr_prediction(args.prediction, args.ground_truth)

        output_path = (
            Path(args.output)
            if args.output
            else EVALUATION_DIR / f"{metrics['report_id']}_ocr_metrics.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"Report:  {metrics['report_id']}")
        print(f"Pages:   {metrics['pages_evaluated']}")
        print(f"CER:     {metrics['cer']:.4f}")
        print(f"WER:     {metrics['wer']:.4f}")
        print(f"Output:  {output_path.resolve()}")

    # ------------------------------------------------------------------
    # OCR — directory
    # ------------------------------------------------------------------
    if args.command == "ocr-dir":
        metrics = evaluate_ocr_directory(args.predictions_dir, args.ground_truth_root)

        output_path = Path(args.output) if args.output else EVALUATION_DIR / "ocr_directory_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"Reports evaluated: {metrics['reports_evaluated']}")
        print(f"Reports skipped:   {metrics['reports_skipped']}")
        if metrics["average_cer"] is not None:
            print(f"Average CER:       {metrics['average_cer']:.4f}")
            print(f"Average WER:       {metrics['average_wer']:.4f}")
        print(f"Output:            {output_path.resolve()}")

    # ------------------------------------------------------------------
    # Extraction — single
    # ------------------------------------------------------------------
    if args.command == "extraction":
        metrics = evaluate_extraction_prediction(args.prediction, args.ground_truth)
        report_id = metrics["report_id"]

        output_path = (
            Path(args.output)
            if args.output
            else EVALUATION_DIR / f"{report_id}_extraction_metrics.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        t = metrics["tests"]
        p = metrics["patient"]
        print(f"Report:             {report_id}")
        print(f"Category:           {metrics.get('category', '')}")
        print(f"Patient accuracy:   {p['accuracy']:.2%}" if p["accuracy"] is not None else "Patient accuracy:   N/A")
        print(f"Test name F1:       {t['name_f1']:.4f}")
        print(f"Value accuracy:     {t['value_accuracy']:.4f}" if t["value_accuracy"] is not None else "Value accuracy:     N/A")
        print(f"Unit accuracy:      {t['unit_accuracy']:.4f}" if t["unit_accuracy"] is not None else "Unit accuracy:      N/A")
        print(f"Status accuracy:    {t['status_accuracy']:.4f}" if t["status_accuracy"] is not None else "Status accuracy:    N/A")
        print(f"Output:             {output_path.resolve()}")

    # ------------------------------------------------------------------
    # Extraction — directory
    # ------------------------------------------------------------------
    if args.command == "extraction-dir":
        metrics = evaluate_extraction_directory(
            args.predictions_dir,
            dataset_root=args.dataset_root if hasattr(args, "dataset_root") else None,
        )

        output_path = (
            Path(args.output)
            if args.output
            else EVALUATION_DIR / "extraction_directory_metrics.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        agg = metrics["aggregate"]
        print(f"Reports evaluated:    {metrics['reports_evaluated']}")
        print(f"Reports skipped:      {metrics['reports_skipped']}")
        print(f"Avg patient accuracy: {agg['avg_patient_accuracy']}")
        print(f"Avg test name F1:     {agg['avg_name_f1']}")
        print(f"Avg value accuracy:   {agg['avg_value_accuracy']}")
        print(f"Avg unit accuracy:    {agg['avg_unit_accuracy']}")
        print(f"Avg status accuracy:  {agg['avg_status_accuracy']}")
        print(f"Output:               {output_path.resolve()}")


if __name__ == "__main__":
    main()
