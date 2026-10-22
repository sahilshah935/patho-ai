import os
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import io
import unittest

os.environ["TESTING"] = "1"

from fastapi import HTTPException, UploadFile
from pathology_ai.api.main import (
    health_check,
    _validate_safe_id,
    upload_pdf,
    chat_with_report,
    get_report,
    ChatRequest,
)


class TestAPI(unittest.TestCase):

    def test_health_endpoint(self):
        res = health_check()
        self.assertEqual(res, {"status": "ok"})

    def test_safe_id_validation(self):
        valid_id = "12345678-1234-1234-1234-123456789abc"
        self.assertEqual(_validate_safe_id(valid_id), valid_id)

        # Path traversal attempts
        with self.assertRaises(HTTPException) as ctx:
            _validate_safe_id("../../etc/passwd")
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            _validate_safe_id("..\\..\\windows\\system32")
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            _validate_safe_id("invalid!id*chars")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_upload_rejects_non_pdf_file(self):
        fake_file = UploadFile(
            filename="malicious.exe",
            file=io.BytesIO(b"MZ..."),
        )
        with self.assertRaises(HTTPException) as ctx:
            upload_pdf(fake_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Only PDF files are supported", ctx.exception.detail)

    def test_upload_rejects_fake_pdf(self):
        fake_file = UploadFile(
            filename="fake.pdf",
            file=io.BytesIO(b"NOT_A_REAL_PDF_MAGIC_BYTES"),
        )
        with self.assertRaises(HTTPException) as ctx:
            upload_pdf(fake_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("not a valid PDF", ctx.exception.detail)

    def test_chat_rejects_path_traversal(self):
        req = ChatRequest(file_id="../../etc/passwd", question="What is my diagnosis?")
        with self.assertRaises(HTTPException) as ctx:
            chat_with_report(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_get_report_rejects_path_traversal(self):
        with self.assertRaises(HTTPException) as ctx:
            get_report("../../../windows/system32")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_without_file_id_answers_general_question(self):
        req = ChatRequest(question="What is diabetes?")
        res = chat_with_report(req)
        self.assertIn("answer", res)
        self.assertIn("disclaimer", res)
        self.assertIn("Diabetes", res["answer"])

    def test_chat_info_endpoint(self):
        from pathology_ai.api.main import get_chat_info
        info = get_chat_info()
        self.assertIn("provider", info)
        self.assertIn("model", info)


if __name__ == "__main__":
    unittest.main()
