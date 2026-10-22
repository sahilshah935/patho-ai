"""Unit tests for normalizer module: test name aliases, units, and reference intervals."""

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import unittest
from pathology_ai.extraction.normalizer import (
    normalize_test_name,
    normalize_unit,
    parse_reference_range,
)


class TestNormalizer(unittest.TestCase):

    def test_analyte_name_normalization(self):
        # Exact alias mappings
        self.assertEqual(normalize_test_name("haemoglobin"), "Hemoglobin")
        self.assertEqual(normalize_test_name("hb"), "Hemoglobin")
        self.assertEqual(normalize_test_name("fasting blood sugar"), "Glucose - Fasting")
        self.assertEqual(normalize_test_name("fbs"), "Glucose - Fasting")
        self.assertEqual(normalize_test_name("total wbc count"), "Total Leucocyte Count")
        self.assertEqual(normalize_test_name("platelets"), "Platelet Count")
        self.assertEqual(normalize_test_name("sgpt"), "SGPT")
        self.assertEqual(normalize_test_name("alt"), "SGPT")
        self.assertEqual(normalize_test_name("tsh"), "TSH")
        self.assertEqual(normalize_test_name("hba1c"), "HbA1c")

    def test_unit_normalization_and_flag_stripping(self):
        # Standard units
        self.assertEqual(normalize_unit("g/dL"), "g/dL")
        self.assertEqual(normalize_unit("gm/dl"), "g/dL")
        self.assertEqual(normalize_unit("/cmm"), "/cmm")
        self.assertEqual(normalize_unit("cells/cmm"), "/cmm")
        self.assertEqual(normalize_unit("lakh/cmm"), "lakh/cmm")

        # Spaced units
        self.assertEqual(normalize_unit("g / dl"), "g/dL")
        self.assertEqual(normalize_unit("mg / dl"), "mg/dL")
        self.assertEqual(normalize_unit("/ cmm"), "/cmm")
        self.assertEqual(normalize_unit("/ hpf"), "/HPF")

        # Units contaminated by flags (critical bug fix test)
        self.assertEqual(normalize_unit("g/dL Low"), "g/dL")
        self.assertEqual(normalize_unit("million/uL Normal"), "million/µL")
        self.assertEqual(normalize_unit("% High"), "%")
        self.assertEqual(normalize_unit("fL Normal"), "fL")
        self.assertEqual(normalize_unit("pg Low"), "pg")

    def test_reference_range_parsing(self):
        # Range between numbers
        self.assertEqual(parse_reference_range("13 - 17"), (13.0, 17.0))
        self.assertEqual(parse_reference_range("12.0 – 15.5"), (12.0, 15.5))
        self.assertEqual(parse_reference_range("4.0 to 11.0"), (4.0, 11.0))
        self.assertEqual(parse_reference_range("4000 - 11000"), (4000.0, 11000.0))

        # Space separated OCR ranges
        self.assertEqual(parse_reference_range("36 46"), (36.0, 46.0))

        # Upper bound only
        self.assertEqual(parse_reference_range("< 200"), (None, 200.0))
        self.assertEqual(parse_reference_range("Up to 5"), (None, 5.0))

        # Lower bound only
        self.assertEqual(parse_reference_range("> 40"), (40.0, None))

        # Qualitative
        self.assertEqual(parse_reference_range("Negative"), (None, None))
        self.assertEqual(parse_reference_range(""), (None, None))


if __name__ == "__main__":
    unittest.main()
