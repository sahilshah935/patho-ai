"""pathology_ai.table_detection — Table boundary and cell grid detection."""

from pathology_ai.table_detection.schema import DetectedTable, TableCell, TableRow
from pathology_ai.table_detection.detector import TableDetector

__all__ = [
    "DetectedTable",
    "TableCell",
    "TableRow",
    "TableDetector",
]
