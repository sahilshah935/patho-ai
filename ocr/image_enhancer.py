"""Computer Vision Image Preprocessing & Enhancement for High-Accuracy OCR.

Applies CLAHE contrast enhancement and bilateral noise filtering to rendered
page images prior to running OCR, preserving tiny text features like decimal
points and hyphens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ImageEnhancer:
    """Computer vision pipeline for preprocessing scanned document images."""

    def __init__(
        self,
        apply_deskew: bool = True,
        apply_clahe: bool = True,
        apply_denoise: bool = True,
        apply_threshold: bool = False,
    ):
        self.apply_deskew = apply_deskew
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise
        self.apply_threshold = apply_threshold

    def deskew(self, image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Correct document scanning rotation/tilt using text bounding angle."""
        try:
            # Invert threshold so text is white on black background
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if coords.shape[0] < 100:
                return image

            rect = cv2.minAreaRect(coords)
            angle = rect[-1]

            # Adjust OpenCV angle conventions
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Only correct if tilt is noticeable (between 0.3 and max_angle degrees)
            if abs(angle) < 0.25 or abs(angle) > max_angle:
                return image

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, rot_mat, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
            return rotated
        except Exception as e:
            # Never let deskew failure break image enhancement
            return image

    def enhance(self, image_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
        """Enhance a rendered page image for maximum OCR text recognition.

        Args:
            image_path: Path to input image file.
            output_path: Path to save enhanced image (optional).

        Returns:
            Path to the saved enhanced image.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Automated Deskewing for tilted scans
        if self.apply_deskew:
            gray = self.deskew(gray)

        # 3. Gentle CLAHE contrast boost (preserves faint decimal dots)
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 4. Gentle bilateral noise filter (preserves sharp text edges & dots)
        if self.apply_denoise:
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # 5. Optional thresholding for extremely dark/washed-out scans
        if self.apply_threshold:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )

        out_file = (
            Path(output_path)
            if output_path
            else img_path.parent / f"{img_path.stem}_enhanced{img_path.suffix}"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), gray)

        return out_file
