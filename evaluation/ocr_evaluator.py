import json
from pathlib import Path
from typing import Any, Dict, List

from pathology_ai.configs.paths import DATASET_ROOT
from pathology_ai.evaluation.text_metrics import character_error_rate, word_error_rate


def load_json(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pages_by_number(pages: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(page["page_number"]): page for page in pages}


def evaluate_ocr_prediction(prediction_path, ground_truth_path) -> Dict[str, Any]:
    prediction = load_json(prediction_path)
    ground_truth = load_json(ground_truth_path)

    prediction_pages = _pages_by_number(prediction["document"]["pages"])
    ground_truth_pages = _pages_by_number(ground_truth["pages"])

    page_metrics = []
    reference_all = []
    prediction_all = []

    for page_number, gt_page in sorted(ground_truth_pages.items()):
        pred_page = prediction_pages.get(page_number, {})
        reference_text = gt_page.get("full_page_text", "")
        predicted_text = (
            pred_page.get("ocr_processed_text")
            or pred_page.get("ocr_text")
            or pred_page.get("text", "")
        )

        reference_all.append(reference_text)
        prediction_all.append(predicted_text)

        page_metrics.append(
            {
                "page_number": page_number,
                "reference_characters": len(reference_text),
                "predicted_characters": len(predicted_text),
                "cer": character_error_rate(reference_text, predicted_text),
                "wer": word_error_rate(reference_text, predicted_text),
            }
        )

    reference_text = "\n".join(reference_all)
    predicted_text = "\n".join(prediction_all)

    return {
        "prediction_path": str(Path(prediction_path).resolve()),
        "ground_truth_path": str(Path(ground_truth_path).resolve()),
        "report_id": ground_truth.get("report_id"),
        "pages_evaluated": len(page_metrics),
        "reference_characters": len(reference_text),
        "predicted_characters": len(predicted_text),
        "cer": character_error_rate(reference_text, predicted_text),
        "wer": word_error_rate(reference_text, predicted_text),
        "page_metrics": page_metrics,
    }


def find_ocr_ground_truth(report_id: str, ocr_ground_truth_root=None) -> Path:
    root = Path(ocr_ground_truth_root) if ocr_ground_truth_root else DATASET_ROOT / "ocr_ground_truth"
    matches = list(root.glob(f"*/{report_id}.json"))

    if not matches:
        raise FileNotFoundError(f"OCR ground truth not found for report id: {report_id}")

    if len(matches) > 1:
        raise ValueError(f"Multiple OCR ground-truth files found for report id: {report_id}")

    return matches[0]


def evaluate_ocr_directory(predictions_dir, ocr_ground_truth_root=None) -> Dict[str, Any]:
    predictions_dir = Path(predictions_dir)
    prediction_paths = sorted(predictions_dir.glob("*.json"))

    reports = []
    skipped = []

    for prediction_path in prediction_paths:
        report_id = prediction_path.stem

        try:
            ground_truth_path = find_ocr_ground_truth(report_id, ocr_ground_truth_root)
            reports.append(evaluate_ocr_prediction(prediction_path, ground_truth_path))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append({"prediction_path": str(prediction_path.resolve()), "reason": str(exc)})

    evaluated_count = len(reports)
    average_cer = sum(report["cer"] for report in reports) / evaluated_count if evaluated_count else None
    average_wer = sum(report["wer"] for report in reports) / evaluated_count if evaluated_count else None

    return {
        "predictions_dir": str(predictions_dir.resolve()),
        "reports_evaluated": evaluated_count,
        "reports_skipped": len(skipped),
        "average_cer": average_cer,
        "average_wer": average_wer,
        "reports": reports,
        "skipped": skipped,
    }
