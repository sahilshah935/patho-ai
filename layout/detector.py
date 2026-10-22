"""Rule-based computer vision layout detector for pathology reports.

Analyzes rendered page images using OpenCV to locate horizontal dividing
lines, text blocks, and table regions, segmenting the page into:
- HEADER (lab name, address, logo)
- PATIENT_INFO (demographics box)
- REPORT_TITLE (panel name)
- TABLE_BODY (investigation result tables)
- REMARKS / FOOTER (signatures, comments)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pathology_ai.layout.schema import BoundingBox, DocumentLayout, LayoutRegion, RegionType


class LayoutDetector:
    """Computer-vision layout detector operating on rendered page images."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def detect_layout(
        self,
        image_path: str | Path,
        page_number: int = 1,
        page_text: str = "",
    ) -> DocumentLayout:
        """Detect visual layout regions from a rendered page image.

        Args:
            image_path: Path to rendered page PNG/JPG image.
            page_number: Page number in document.
            page_text: Plain text extracted from page (optional, for text alignment).

        Returns:
            ``DocumentLayout`` containing list of ``LayoutRegion`` objects.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Page image not found: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        h, w, _ = img.shape
        layout = DocumentLayout(page_number=page_number, width=w, height=h)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find horizontal dividing lines (common in pathology report templates)
        horizontal_lines = self._detect_horizontal_lines(gray, min_width_pct=0.4)

        # Segment page vertically into 4 primary bands if horizontal lines exist:
        # Band 1: Top 0-18% -> Header
        # Band 2: 18-35% -> Patient Info
        # Band 3: 35-82% -> Table Body
        # Band 4: 82-100% -> Footer / Remarks
        
        y_header_end = int(h * 0.18)
        y_patient_end = int(h * 0.35)
        y_table_end = int(h * 0.82)

        # Refine boundaries using detected horizontal lines if available
        if len(horizontal_lines) >= 2:
            y_lines = sorted(line[1] for line in horizontal_lines)
            for y_line in y_lines:
                if 0.12 * h < y_line < 0.25 * h:
                    y_header_end = y_line
                elif 0.25 * h < y_line < 0.42 * h:
                    y_patient_end = y_line
                elif 0.75 * h < y_line < 0.90 * h:
                    y_table_end = y_line

        # Header
        layout.regions.append(
            LayoutRegion(
                region_type=RegionType.HEADER,
                bbox=BoundingBox(x=0, y=0, w=w, h=y_header_end),
            )
        )

        # Patient Info
        layout.regions.append(
            LayoutRegion(
                region_type=RegionType.PATIENT_INFO,
                bbox=BoundingBox(x=0, y=y_header_end, w=w, h=y_patient_end - y_header_end),
            )
        )

        # Table Body
        layout.regions.append(
            LayoutRegion(
                region_type=RegionType.TABLE_BODY,
                bbox=BoundingBox(x=0, y=y_patient_end, w=w, h=y_table_end - y_patient_end),
            )
        )

        # Footer
        layout.regions.append(
            LayoutRegion(
                region_type=RegionType.FOOTER,
                bbox=BoundingBox(x=0, y=y_table_end, w=w, h=h - y_table_end),
            )
        )

        return layout

    @staticmethod
    def _detect_horizontal_lines(
        gray_img: np.ndarray,
        min_width_pct: float = 0.4,
    ) -> List[Tuple[int, int, int]]:
        """Find long horizontal dividing lines in a grayscale page image."""
        h, w = gray_img.shape
        min_len = int(w * min_width_pct)

        # Binary inverse threshold
        _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

        # Horizontal morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw >= min_len:
                lines.append((x, y + ch // 2, cw))
        return lines
