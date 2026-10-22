from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PDFPage:

    page_number: int

    text: str = ""

    image_path: str = ""

    width: int = 0

    height: int = 0

    dpi: int = 300

    ocr_text: str = ""

    ocr_processed_text: str = ""

    ocr_blocks: List[Dict[str, Any]] = field(default_factory=list)

    ocr_lines: List[Dict[str, Any]] = field(default_factory=list)

    ocr_engine: str = ""

    ocr_languages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "text_characters": len(self.text or ""),
            "ocr_text": self.ocr_text,
            "ocr_text_characters": len(self.ocr_text or ""),
            "ocr_processed_text": self.ocr_processed_text,
            "ocr_processed_text_characters": len(self.ocr_processed_text or ""),
            "ocr_blocks": self.ocr_blocks,
            "ocr_lines": self.ocr_lines,
            "ocr_engine": self.ocr_engine,
            "ocr_languages": self.ocr_languages,
        }


@dataclass
class PDFDocument:

    file_name: str

    file_path: str

    page_count: int

    pdf_type: str

    metadata: Dict[str, Any] = field(default_factory=dict)

    pages: List[PDFPage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_count": self.page_count,
            "pdf_type": self.pdf_type,
            "metadata": self.metadata,
            "total_text_characters": sum(len(page.text or "") for page in self.pages),
            "total_ocr_text_characters": sum(len(page.ocr_text or "") for page in self.pages),
            "total_ocr_processed_text_characters": sum(
                len(page.ocr_processed_text or "") for page in self.pages
            ),
            "pages": [page.to_dict() for page in self.pages],
        }
