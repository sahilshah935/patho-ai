from pathlib import Path

import fitz

from pathology_ai.configs.paths import RENDERED_PAGES_DIR
from pathology_ai.preprocessing.document import PDFDocument
from pathology_ai.preprocessing.document import PDFPage

from pathology_ai.preprocessing.page_renderer import PageRenderer


class PDFParser:

    def __init__(self, output_dir=None, dpi=300, render_pages=True):

        self.output_dir = Path(output_dir) if output_dir else RENDERED_PAGES_DIR
        self.render_pages = render_pages
        self.renderer = PageRenderer(dpi=dpi)

    def detect_pdf_type(self, doc):

        characters = 0

        for page in doc:

            characters += len(page.get_text())

        if characters > 50:
            return "digital"

        return "scanned"

    @staticmethod
    def _extract_spatial_text(page) -> str:
        """Extract text from PyMuPDF page using spatial y-coordinate line reconstruction.
        
        Prevents column-merging artifacts by grouping words on the same horizontal line.
        """
        words = page.get_text("words")
        if not words:
            return page.get_text()

        # Group words by rounded y0 coordinate (vertical tolerance ~3 points)
        lines_by_y = {}
        for w in words:
            y_key = round(w[1] / 3.0) * 3
            lines_by_y.setdefault(y_key, []).append(w)

        lines = []
        for y in sorted(lines_by_y.keys()):
            # Sort words in each line from left to right (x0)
            line_words = sorted(lines_by_y[y], key=lambda item: item[0])
            line_str = " ".join(w[4] for w in line_words)
            if line_str.strip():
                lines.append(line_str.strip())

        return "\n".join(lines)

    def parse(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        try:
            pdf_type = self.detect_pdf_type(doc)

            document = PDFDocument(
                file_name=pdf_path.name,
                file_path=str(pdf_path.resolve()),
                page_count=len(doc),
                pdf_type=pdf_type,
                metadata=doc.metadata
            )

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for i, page in enumerate(doc):

                text = self._extract_spatial_text(page)

                image_path = ""

                if self.render_pages:
                    image_path = self.output_dir / f"{pdf_path.stem}_page_{i+1}.png"
                    self.renderer.render(page, image_path)

                rect = page.rect

                document.pages.append(
                    PDFPage(
                        page_number=i + 1,
                        text=text,
                        image_path=str(image_path),
                        width=int(rect.width),
                        height=int(rect.height),
                        dpi=self.renderer.dpi,
                    )
                )

            return document

        finally:
            doc.close()
