import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathology_ai.configs.paths import (
    EXTRACTION_DIR,
    INTERPRETATION_DIR,
    OCR_MODELS_DIR,
    PREDICTIONS_DIR,
    RENDERED_PAGES_DIR,
)
from pathology_ai.ocr.engines import create_ocr_engine
from pathology_ai.ocr.pipeline import apply_ocr
from pathology_ai.preprocessing.pdf_parser import PDFParser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a pathology report PDF into structured document JSON."
    )
    parser.add_argument("pdf_path", help="Path to a pathology report PDF.")
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to outputs/predictions/<pdf_stem>.json.",
    )
    parser.add_argument(
        "--render-dir",
        default=str(RENDERED_PAGES_DIR),
        help="Directory where rendered page images will be saved.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for page images.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip page image rendering and only extract PDF metadata/text.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR on scanned pages after rendering them.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Run OCR even when the PDF has embedded digital text.",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr"],
        help="OCR engine to use.",
    )
    parser.add_argument(
        "--ocr-langs",
        default="en",
        help="Comma-separated OCR language codes, for example: en,hi.",
    )
    parser.add_argument(
        "--cpu-ocr",
        action="store_true",
        help="Run OCR on CPU instead of GPU.",
    )
    parser.add_argument(
        "--ocr-model-dir",
        default=str(OCR_MODELS_DIR),
        help="Directory for OCR model files/cache.",
    )
    # --- New flags ---
    parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "Run structured field extraction after OCR. "
            "Extracts patient info, test results, units, and reference ranges."
        ),
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help=(
            "Run clinical interpretation after extraction (implies --extract). "
            "Classifies test statuses, detects patterns, and generates "
            "plain-language explanations and an overall summary."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = Path(args.pdf_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if args.no_render and (args.ocr or args.force_ocr):
        raise SystemExit("OCR requires rendered page images. Remove --no-render.")

    if args.interpret:
        args.extract = True  # --interpret implies --extract

    output_path = Path(args.output) if args.output else PREDICTIONS_DIR / f"{pdf_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: PDF parsing + rendering
    # ------------------------------------------------------------------
    parser = PDFParser(
        output_dir=args.render_dir,
        dpi=args.dpi,
        render_pages=not args.no_render,
    )
    document = parser.parse(pdf_path)

    # ------------------------------------------------------------------
    # Stage 2: OCR
    # ------------------------------------------------------------------
    ocr_summary = None
    if args.ocr or args.force_ocr:
        languages = [lang.strip() for lang in args.ocr_langs.split(",") if lang.strip()]
        engine = create_ocr_engine(
            args.ocr_engine,
            languages=languages,
            gpu=not args.cpu_ocr,
            model_dir=args.ocr_model_dir,
        )
        ocr_summary = apply_ocr(document, engine, force=args.force_ocr)

    # ------------------------------------------------------------------
    # Stage 3: Extraction
    # ------------------------------------------------------------------
    extraction_dict = None
    if args.extract:
        from pathology_ai.extraction.pipeline import ExtractionPipeline

        extraction_pipeline = ExtractionPipeline()
        extraction_result = extraction_pipeline.extract(document)
        extraction_dict = extraction_result.to_dict()

        # Save standalone extraction JSON
        ext_path = EXTRACTION_DIR / f"{pdf_path.stem}_extraction.json"
        ext_path.parent.mkdir(parents=True, exist_ok=True)
        ext_path.write_text(json.dumps(extraction_dict, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Stage 4: Interpretation (optional)
    # ------------------------------------------------------------------
    interpretation_dict = None
    if args.interpret and extraction_result is not None:
        from pathology_ai.inference.interpreter import Interpreter
        from pathology_ai.inference.explainer import explain_and_summarise

        interpreter = Interpreter()
        interpretation_result = interpreter.interpret(extraction_result)
        interpretation_result = explain_and_summarise(interpretation_result)
        interpretation_dict = interpretation_result.to_dict()

        # Update extraction dict with per-test explanations and summary
        extraction_dict = interpretation_result.extraction.to_dict()

        # Save standalone interpretation JSON
        interp_path = INTERPRETATION_DIR / f"{pdf_path.stem}_interpretation.json"
        interp_path.parent.mkdir(parents=True, exist_ok=True)
        interp_path.write_text(json.dumps(interpretation_dict, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Build output payload
    # ------------------------------------------------------------------
    if args.interpret:
        stage = "pdf_preprocessing_ocr_extraction_interpretation"
    elif args.extract:
        stage = "pdf_preprocessing_ocr_extraction"
    elif args.ocr or args.force_ocr:
        stage = "pdf_preprocessing_ocr"
    else:
        stage = "pdf_preprocessing"

    payload = {
        "pipeline_stage": stage,
        "disclaimer": (
            "This output is for pathology report understanding and decision "
            "support only. It is not a diagnosis and should be confirmed by "
            "an appropriate healthcare professional."
        ),
        "ocr": ocr_summary,
        "document": document.to_dict(),
        "extraction": extraction_dict,
        "interpretation": interpretation_dict,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"Parsed:             {document.file_name}")
    print(f"Type:               {document.pdf_type}")
    print(f"Pages:              {document.page_count}")
    print(f"Text chars:         {payload['document']['total_text_characters']}")
    print(f"OCR chars:          {payload['document']['total_ocr_text_characters']}")

    if extraction_dict:
        all_tests = [
            t for s in extraction_dict.get("sections", [])
            for t in s.get("tests", [])
        ]
        print(f"Category:           {extraction_dict.get('document_category', '')}")
        print(f"Tests extracted:    {len(all_tests)}")
        print(f"Extr. confidence:   {extraction_dict.get('extraction_confidence', 'N/A')}")
        print(f"Requires review:    {extraction_dict.get('requires_review', False)}")

    if interpretation_dict:
        patterns = interpretation_dict.get("patterns", [])
        if patterns:
            print(f"Patterns detected:  {', '.join(p['name'] for p in patterns)}")
        print(f"Summary:            {interpretation_dict.get('summary', '')[:120]}...")

    print(f"Output:             {output_path.resolve()}")


if __name__ == "__main__":
    main()
