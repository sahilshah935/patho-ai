from pathlib import Path
import fitz


class PDFLoader:

    def __init__(self, pdf_path):

        self.pdf_path = Path(pdf_path)

    def load(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        return fitz.open(self.pdf_path)
