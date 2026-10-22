import torch
import fitz
import pdfplumber
import cv2
import easyocr
import transformers
import datasets
import ultralytics
import rapidfuzz
import paddleocr

print("=" * 60)
print("PATHOLOGY AI INSTALLATION CHECK")
print("=" * 60)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("PyMuPDF: OK")
print("pdfplumber: OK")
print("OpenCV: OK")
print("EasyOCR: OK")
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Ultralytics:", ultralytics.__version__)
print("RapidFuzz:", rapidfuzz.__version__)
print("PaddleOCR: OK")

print("=" * 60)
print("Everything Installed Successfully")
print("=" * 60)