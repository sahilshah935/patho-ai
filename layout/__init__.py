"""pathology_ai.layout — Document layout analysis and region segmentation."""

from pathology_ai.layout.schema import BoundingBox, DocumentLayout, LayoutRegion, RegionType
from pathology_ai.layout.detector import LayoutDetector

__all__ = [
    "BoundingBox",
    "DocumentLayout",
    "LayoutRegion",
    "RegionType",
    "LayoutDetector",
]
