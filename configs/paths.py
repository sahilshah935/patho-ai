from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "pathology_dataset"
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
UPLOADS_DIR = OUTPUT_ROOT / "uploads"
RENDERED_PAGES_DIR = OUTPUT_ROOT / "rendered_pages"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
EVALUATION_DIR = OUTPUT_ROOT / "evaluation"
EXTRACTION_DIR = OUTPUT_ROOT / "extractions"
INTERPRETATION_DIR = OUTPUT_ROOT / "interpretations"
OCR_MODELS_DIR = PACKAGE_ROOT / "models" / "easyocr"
OCR_USER_NETWORKS_DIR = OCR_MODELS_DIR / "user_network"

# Auto-create output directories
for d in (OUTPUT_ROOT, UPLOADS_DIR, RENDERED_PAGES_DIR, PREDICTIONS_DIR, EVALUATION_DIR, EXTRACTION_DIR, INTERPRETATION_DIR, OCR_MODELS_DIR, OCR_USER_NETWORKS_DIR):
    d.mkdir(parents=True, exist_ok=True)
