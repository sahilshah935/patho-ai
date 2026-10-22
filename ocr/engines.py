from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pathology_ai.configs.paths import OCR_MODELS_DIR, OCR_USER_NETWORKS_DIR


@dataclass
class OCRBlock:
    text: str
    confidence: float
    bbox: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


from pathology_ai.ocr.image_enhancer import ImageEnhancer


class EasyOCREngine:
    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: bool = True,
        model_storage_directory=None,
        user_network_directory=None,
    ):
        import easyocr

        self.languages = list(languages)
        self.gpu = gpu
        self.enhancer = ImageEnhancer(apply_clahe=True, apply_denoise=True)
        self.model_storage_directory = Path(model_storage_directory or OCR_MODELS_DIR)
        self.user_network_directory = Path(user_network_directory or OCR_USER_NETWORKS_DIR)
        self.model_storage_directory.mkdir(parents=True, exist_ok=True)
        self.user_network_directory.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            self.languages,
            gpu=gpu,
            model_storage_directory=str(self.model_storage_directory),
            user_network_directory=str(self.user_network_directory),
            verbose=False,
        )

    def read_image(self, image_path) -> Dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Rendered page image not found: {image_path}")

        # Preprocess / Enhance image first
        try:
            enhanced_path = self.enhancer.enhance(image_path)
            target_image = str(enhanced_path)
        except Exception as e:
            target_image = str(image_path)

        results = self.reader.readtext(
            target_image,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            canvas_size=2560,
        )
        blocks = [
            OCRBlock(
                text=str(text),
                confidence=float(confidence),
                bbox=[[float(x), float(y)] for x, y in bbox],
            )
            for bbox, text, confidence in results
        ]

        return {
            "engine": "easyocr",
            "languages": self.languages,
            "text": "\n".join(block.text for block in blocks),
            "blocks": [block.to_dict() for block in blocks],
            "block_count": len(blocks),
        }


def create_ocr_engine(name: str, languages: Sequence[str], gpu: bool, model_dir=None):
    normalized_name = name.lower().strip()

    if normalized_name == "easyocr":
        return EasyOCREngine(languages=languages, gpu=gpu, model_storage_directory=model_dir)

    raise ValueError(f"Unsupported OCR engine: {name}")
