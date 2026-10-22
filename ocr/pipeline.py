from typing import Any, Dict

from pathology_ai.ocr.postprocessing import postprocess_ocr_blocks


def apply_ocr(document, engine, force: bool = False) -> Dict[str, Any]:
    pages_processed = 0
    total_blocks = 0
    total_lines = 0

    for page in document.pages:
        should_ocr = force or document.pdf_type == "scanned" or not page.text.strip()

        if not should_ocr:
            continue

        if not page.image_path:
            raise ValueError("OCR requires rendered page images. Run without --no-render.")

        result = engine.read_image(page.image_path)
        processed = postprocess_ocr_blocks(result["blocks"])

        page.ocr_text = result["text"]
        page.ocr_processed_text = processed["text"]
        page.ocr_blocks = result["blocks"]
        page.ocr_lines = processed["lines"]
        page.ocr_engine = result["engine"]
        page.ocr_languages = result["languages"]

        pages_processed += 1
        total_blocks += result["block_count"]
        total_lines += processed["line_count"]

    return {
        "pages_processed": pages_processed,
        "total_blocks": total_blocks,
        "total_lines": total_lines,
        "engine": engine.__class__.__name__,
    }
