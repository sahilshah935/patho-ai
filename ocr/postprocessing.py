import re
from statistics import median
from typing import Any, Dict, List


def _bbox_stats(block: Dict[str, Any]) -> Dict[str, float]:
    points = block.get("bbox") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    if not xs or not ys:
        return {
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.0,
            "y_max": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "height": 0.0,
        }

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "x_center": (x_min + x_max) / 2,
        "y_center": (y_min + y_max) / 2,
        "height": y_max - y_min,
    }


def _correct_ocr_text(text: str) -> str:
    text = text.replace('gldL', 'g/dL')
    text = text.replace('gldl', 'g/dL')
    text = text.replace('gmldl', 'g/dL')
    text = text.replace('icmm', '/cmm')
    text = text.replace('Icmm', '/cmm')
    text = text.replace('lhpf', '/HPF')
    text = text.replace('perlhpf', '/HPF')
    text = text.replace('lakhlcmm', 'lakh/cmm')
    text = text.replace('lakhicmm', 'lakh/cmm')
    text = text.replace('millionlcmm', 'million/cmm')
    text = text.replace('milllcmm', 'million/cmm')
    text = text.replace('x103', 'x10^3')
    text = text.replace('x106', 'x10^6')
    text = re.sub(r'\buL\b', 'µL', text)

    # Normalize spaced units common in OCR bounding boxes
    text = re.sub(r'\bg\s*/\s*d[Ll]\b', 'g/dL', text)
    text = re.sub(r'\bgm\s*/\s*d[Ll]\b', 'g/dL', text)
    text = re.sub(r'\bmg\s*/\s*d[Ll]\b', 'mg/dL', text)
    text = re.sub(r'\bng\s*/\s*m[Ll]\b', 'ng/mL', text)
    text = re.sub(r'\bpg\s*/\s*m[Ll]\b', 'pg/mL', text)
    text = re.sub(r'\bmm\s*/\s*hr\b', 'mm/hr', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[uµ]iu\s*/\s*m[Ll]\b', 'µIU/mL', text, flags=re.IGNORECASE)
    text = re.sub(r'\biu\s*/\s*[Ll]\b', 'IU/L', text, flags=re.IGNORECASE)
    text = re.sub(r'\bu\s*/\s*[Ll]\b', 'U/L', text, flags=re.IGNORECASE)
    text = re.sub(r'/\s*cmm\b', '/cmm', text, flags=re.IGNORECASE)
    text = re.sub(r'/\s*hpf\b', '/HPF', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmillion\s*/\s*cmm\b', 'million/cmm', text, flags=re.IGNORECASE)
    text = re.sub(r'\blakh\s*/\s*cmm\b', 'lakh/cmm', text, flags=re.IGNORECASE)
    text = re.sub(r'x\s*10\s*\^\s*(\d)', r'x10^\1', text)
    return text


def _enrich_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []

    for index, block in enumerate(blocks):
        text = (block.get("text") or "").strip()
        text = _correct_ocr_text(text)

        if not text:
            continue

        enriched_block = dict(block)
        enriched_block["text"] = text
        enriched_block["original_index"] = index
        enriched_block.update(_bbox_stats(block))
        enriched.append(enriched_block)

    return enriched


def reconstruct_lines(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched_blocks = _enrich_blocks(blocks)

    if not enriched_blocks:
        return []

    heights = [block["height"] for block in enriched_blocks if block["height"] > 0]
    typical_height = median(heights) if heights else 12.0
    y_tolerance = max(typical_height * 0.5, 6.0)

    lines = []

    for block in sorted(enriched_blocks, key=lambda item: (item["y_center"], item["x_min"])):
        best_line = None
        best_distance = None

        for line in lines:
            distance = abs(line["y_center"] - block["y_center"])

            if distance <= y_tolerance and (best_distance is None or distance < best_distance):
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append({"blocks": [block], "y_center": block["y_center"]})
            continue

        best_line["blocks"].append(block)
        best_line["y_center"] = sum(item["y_center"] for item in best_line["blocks"]) / len(
            best_line["blocks"]
        )

    reconstructed = []

    for line_number, line in enumerate(
        sorted(lines, key=lambda item: min(block["y_min"] for block in item["blocks"])),
        start=1,
    ):
        line_blocks = sorted(line["blocks"], key=lambda item: item["x_min"])
        text = " ".join(block["text"] for block in line_blocks)

        reconstructed.append(
            {
                "line_number": line_number,
                "text": text,
                "block_count": len(line_blocks),
                "bbox": [
                    min(block["x_min"] for block in line_blocks),
                    min(block["y_min"] for block in line_blocks),
                    max(block["x_max"] for block in line_blocks),
                    max(block["y_max"] for block in line_blocks),
                ],
                "blocks": [
                    {
                        "text": block["text"],
                        "confidence": block.get("confidence", 0.0),
                        "bbox": block.get("bbox", []),
                        "original_index": block["original_index"],
                    }
                    for block in line_blocks
                ],
            }
        )

    return reconstructed


def _restore_hyphens(line_text: str) -> str:
    # Match two numbers that look like a reference range near line end, followed optionally by a flag
    pattern = re.compile(
        r'(\b\d+\.?\d*)\s+(\d+\.?\d*)\b(?:\s+(?:[HLhl]\*?|High|Low|Normal|Critical|Abnormal|Borderline|BL|BH|-))?\s*$',
        re.IGNORECASE
    )
    m = pattern.search(line_text)
    if m:
        try:
            num1 = float(m.group(1))
            num2 = float(m.group(2))
            if num2 > num1 and num1 >= 0:
                span_start, span_end = m.span(1)[0], m.span(2)[1]
                replacement = f"{m.group(1)} - {m.group(2)}"
                return line_text[:span_start] + replacement + line_text[span_end:]
        except ValueError:
            pass
    return line_text


def postprocess_ocr_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    lines = reconstruct_lines(blocks)

    for line in lines:
        line["text"] = _restore_hyphens(line["text"])

    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "line_count": len(lines),
    }

