"""Unit tests for text extraction: lab name, demographics, row parsing, and flag isolation."""

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import unittest
from pathology_ai.extraction.text_extractor import (
    extract_laboratory_name,
    extract_document_category,
    extract_panel_name,
    extract_patient_info,
    extract_test_rows,
    build_extracted_section,
    _clean_unit_and_flag,
)


class TestTextExtractor(unittest.TestCase):

    def test_extract_laboratory_name_skips_page_and_header_tags(self):
        text = (
            "[PAGE 1]\n"
            "Laboratory Report\n"
            "Page 1 of 1\n"
            "AAROGYA REFERENCE LABORATORY\n"
            "Indore Main Branch\n"
            "143, Ring Road, Indore\n"
            "HAEMATOLOGY PROFILE\n"
        )
        lab_name = extract_laboratory_name(text)
        self.assertEqual(lab_name, "AAROGYA REFERENCE LABORATORY")
        self.assertNotIn("[PAGE", lab_name)
        self.assertNotIn("Page 1", lab_name)

    def test_extract_patient_info_no_junk_ids(self):
        text = (
            "Report No. UHID / Patient ID\n"
            "ADP/HM/26/0802/4176 UHID-26-771984\n"
            "Sample ID Patient Name\n"
            "EDTA-8042816 Meera S. Kulkarni\n"
            "Age / Gender 34 Years / Female Referring Doctor\n"
            "Dr. Rohan Deshpande, MD (Internal Medicine)\n"
            "Department General Medicine Sample Type Whole Blood (EDTA)\n"
        )
        patient = extract_patient_info(text)
        self.assertEqual(patient.name, "Meera S. Kulkarni")
        self.assertEqual(patient.age, 34)
        self.assertEqual(patient.sex, "Female")
        # Ensure 'Patient' and 'UHID' are not used as IDs
        self.assertNotEqual(patient.patient_id.lower(), "patient")
        self.assertNotEqual(patient.patient_id.lower(), "uhid")
        self.assertEqual(patient.patient_id, "UHID-26-771984")
        self.assertEqual(patient.sample_id, "EDTA-8042816")
        self.assertEqual(patient.accession_number, "ADP/HM/26/0802/4176")
        self.assertIn("Rohan Deshpande", patient.referred_by)

    def test_clean_unit_and_flag_isolation(self):
        unit, flag = _clean_unit_and_flag("g/dL Low", "")
        self.assertEqual(unit, "g/dL")
        self.assertEqual(flag.lower(), "low")

        unit, flag = _clean_unit_and_flag("million/uL Normal", "")
        self.assertEqual(unit, "million/uL")
        self.assertEqual(flag.lower(), "normal")

        unit, flag = _clean_unit_and_flag("% High", "")
        self.assertEqual(unit, "%")
        self.assertEqual(flag.lower(), "high")

    def test_extract_test_rows_does_not_leak_flags_into_units(self):
        text = (
            "INVESTIGATION RESULTS\n"
            "Investigation Result Unit Biological Reference Range Flag\n"
            "Hemoglobin 10.8 g/dL 12.0 - 15.5 Low\n"
            "RBC Count 4.18 million/uL 4.0 - 5.0 Normal\n"
            "Hematocrit (PCV) 34.2 % 36 - 46 Low\n"
            "Platelet Count 468000 /uL 150000 - 400000 High\n"
        )
        rows = extract_test_rows(text)
        self.assertEqual(len(rows), 4)

        section = build_extracted_section("Investigation Results", rows)
        tests_by_name = {t.test_name_normalized: t for t in section.tests}

        # Check Hemoglobin
        hb = tests_by_name["Hemoglobin"]
        self.assertEqual(hb.value_numeric, 10.8)
        self.assertEqual(hb.unit_normalized, "g/dL")
        self.assertNotIn("Low", hb.unit_normalized)
        self.assertIn(hb.flag_raw.lower(), ("low", "l"))

        # Check RBC Count
        rbc = tests_by_name["RBC Count"]
        self.assertEqual(rbc.value_numeric, 4.18)
        self.assertEqual(rbc.unit_normalized, "million/µL")
        self.assertNotIn("Normal", rbc.unit_normalized)


if __name__ == "__main__":
    unittest.main()
