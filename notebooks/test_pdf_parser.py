import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathology_ai.preprocessing.pdf_parser import PDFParser

parser = PDFParser()

sample_pdf = Path("pathology_dataset/raw_documents/digital/hematology/RPT000001.pdf")

doc = parser.parse(sample_pdf)

print("=" * 60)

print("File:", doc.file_name)

print("Type:", doc.pdf_type)

print("Pages:", doc.page_count)

print()

for page in doc.pages:

    print("Page:", page.page_number)

    print("Image:", page.image_path)

    print("Characters:", len(page.text))

    print("-" * 40)
