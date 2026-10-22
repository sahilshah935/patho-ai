"""Dataclasses for Document Layout Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class RegionType(str, Enum):
    HEADER = "header"
    PATIENT_INFO = "patient_info"
    REPORT_TITLE = "report_title"
    TABLE_BODY = "table_body"
    REMARKS = "remarks"
    FOOTER = "footer"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class LayoutRegion:
    region_type: RegionType
    bbox: BoundingBox
    confidence: float = 1.0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_type": self.region_type.value,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "text": self.text,
        }


@dataclass
class DocumentLayout:
    page_number: int
    width: int
    height: int
    regions: List[LayoutRegion] = field(default_factory=list)

    def get_regions_by_type(self, region_type: RegionType) -> List[LayoutRegion]:
        return [r for r in self.regions if r.region_type == region_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
        }
