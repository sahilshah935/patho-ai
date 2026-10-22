"""Unit tests for clinical rules: status classification, critical thresholds, and multi-test patterns."""

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import unittest
from pathology_ai.extraction.schema import ExtractedTest, ExtractionResult, ExtractedSection
from pathology_ai.inference.rules import classify_status, is_critical
from pathology_ai.inference.interpreter import Interpreter
from pathology_ai.inference.explainer import explain_and_summarise, DISCLAIMER


class TestClinicalRules(unittest.TestCase):

    def test_status_classification(self):
        # Within normal range
        self.assertEqual(classify_status(14.0, 13.0, 17.0), "NORMAL")
        # High and Low
        self.assertEqual(classify_status(18.5, 13.0, 17.0), "HIGH")
        self.assertEqual(classify_status(11.0, 13.0, 17.0), "LOW")

        # Critical threshold checks via classify_status
        crit_low_status = classify_status(6.0, 13.0, 17.0, critical_low=7.0, critical_high=20.0)
        self.assertEqual(crit_low_status, "CRITICAL_LOW")
        self.assertTrue(is_critical(crit_low_status))

        crit_high_status = classify_status(22.0, 13.0, 17.0, critical_low=7.0, critical_high=20.0)
        self.assertEqual(crit_high_status, "CRITICAL_HIGH")
        self.assertTrue(is_critical(crit_high_status))

        normal_status = classify_status(14.0, 13.0, 17.0, critical_low=7.0, critical_high=20.0)
        self.assertFalse(is_critical(normal_status))

    def test_multi_test_pattern_detection(self):
        # Simulate microcytic anemia (Low Hb, Low MCV, Low MCH)
        tests = [
            ExtractedTest(
                test_name_raw="Hemoglobin",
                test_name_normalized="Hemoglobin",
                value_numeric=9.5,
                reference_low=12.0,
                reference_high=15.5,
                unit_normalized="g/dL",
            ),
            ExtractedTest(
                test_name_raw="MCV",
                test_name_normalized="MCV",
                value_numeric=72.0,
                reference_low=80.0,
                reference_high=100.0,
                unit_normalized="fL",
            ),
            ExtractedTest(
                test_name_raw="MCH",
                test_name_normalized="MCH",
                value_numeric=22.0,
                reference_low=27.0,
                reference_high=33.0,
                unit_normalized="pg",
            ),
        ]
        section = ExtractedSection(section_name="Hematology", tests=tests)
        extraction = ExtractionResult(sections=[section], document_category="Hematology")

        interpreter = Interpreter()
        result = interpreter.interpret(extraction)

        pattern_names = [p.name for p in result.patterns]
        self.assertIn("Microcytic Anaemia Pattern", pattern_names)

        # Check explanation and disclaimer
        result = explain_and_summarise(result)
        self.assertIn("DISCLAIMER", result.disclaimer)
        self.assertTrue(len(result.summary) > 20)


if __name__ == "__main__":
    unittest.main()
