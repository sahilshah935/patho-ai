"""Dataclasses for Table Detection & Extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from pathology_ai.layout.schema import BoundingBox


@dataclass
class TableCell:
    row_index: int
    col_index: int
    bbox: BoundingBox
    text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "col_index": self.col_index,
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class TableRow:
    row_index: int
    cells: List[TableCell] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [c.to_dict() for c in self.cells],
        }


@dataclass
class DetectedTable:
    table_index: int
    bbox: BoundingBox
    rows: List[TableRow] = field(default_factory=list)
    col_count: int = 0
    row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_index": self.table_index,
            "bbox": self.bbox.to_dict(),
            "row_count": self.row_count,
            "col_count": self.col_count,
            "rows": [r.to_dict() for r in self.rows],
        }
