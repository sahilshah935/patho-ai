from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import uuid
import sys
import os
import re

# Add the project root to sys.path and support pathology_ai module aliasing
REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, PARENT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if "pathology_ai" not in sys.modules:
    import types
    _pkg = types.ModuleType("pathology_ai")
    _pkg.__path__ = [str(REPO_ROOT)]
    sys.modules["pathology_ai"] = _pkg

from pathology_ai.configs.paths import PREDICTIONS_DIR, RENDERED_PAGES_DIR, UPLOADS_DIR, OUTPUT_ROOT
from pathology_ai.preprocessing.pdf_parser import PDFParser
from pathology_ai.ocr.engines import create_ocr_engine
from pathology_ai.ocr.pipeline import apply_ocr
from pathology_ai.extraction.pipeline import ExtractionPipeline
from pathology_ai.inference.interpreter import Interpreter
from pathology_ai.inference.explainer import explain_and_summarise

app = FastAPI(title="Pathology Report Intelligence Platform API")

# Allow CORS for the React frontend (which typically runs on port 5173 for Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and output directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_PAGES_DIR.mkdir(parents=True, exist_ok=True)

from typing import Optional, List, Dict
from pydantic import BaseModel
from pathology_ai.inference.chatbot import PathologyChatbot

class ChatRequest(BaseModel):
    file_id: Optional[str] = None
    question: str
    history: Optional[List[Dict[str, str]]] = None

# Cache the pipelines and engines to avoid reloading them for each request
class PipelineState:
    ocr_engine = None
    extraction_pipeline = None
    interpreter = None
    chatbot = None

state = PipelineState()

def get_ocr_engine():
    if state.ocr_engine is None:
        state.ocr_engine = create_ocr_engine(
            name="easyocr",
            languages=["en"],
            gpu=True, # Attempt GPU
        )
    return state.ocr_engine

def get_extraction_pipeline():
    if state.extraction_pipeline is None:
        state.extraction_pipeline = ExtractionPipeline()
    return state.extraction_pipeline

def get_interpreter():
    if state.interpreter is None:
        state.interpreter = Interpreter()
    return state.interpreter

def get_chatbot():
    if state.chatbot is None:
        state.chatbot = PathologyChatbot()
    return state.chatbot


@app.on_event("startup")
async def preload_models():
    """Pre-warm AI models into GPU VRAM on server startup so user uploads respond instantly."""
    if os.environ.get("TESTING") == "1":
        return
    print("Pre-warming AI models on GPU...")
    try:
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(get_extraction_pipeline)
        await run_in_threadpool(get_interpreter)
        print("AI Models pre-warmed successfully!")
    except Exception as e:
        print(f"Warning pre-warming models: {e}")


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit
SAFE_ID_PATTERN = re.compile(r"^[0-9a-fA-F\-]{16,64}$")


def _validate_safe_id(file_id: str) -> str:
    """Validate that file_id is a safe UUID / alphanumeric token to prevent path traversal."""
    if not file_id or not SAFE_ID_PATTERN.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid identifier format.")
    clean_id = os.path.basename(file_id)
    target_path = (PREDICTIONS_DIR / f"{clean_id}.json").resolve()
    if not str(target_path).startswith(str(PREDICTIONS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")
    return clean_id


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Accepts a PDF upload, runs the full pathology AI pipeline in worker threads, and returns structured JSON."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Validate file content and magic bytes
    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25 MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF document.")

    file_id = str(uuid.uuid4())
    pdf_path = UPLOADS_DIR / f"{file_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save file to disk
        pdf_path.write_bytes(content)

        # 2. PDF Parsing & Rendering
        parser = PDFParser(
            output_dir=RENDERED_PAGES_DIR,
            dpi=300,
            render_pages=True,
        )
        document = parser.parse(pdf_path)

        # 3. OCR (Only for scanned PDFs to make digital uploads instant)
        ocr_summary = {"status": "skipped", "reason": "digital PDF"}
        if document.pdf_type == "scanned":
            ocr_engine = get_ocr_engine()
            ocr_summary = apply_ocr(document, ocr_engine, force=False)

        # 4. Extraction
        extraction_pipeline = get_extraction_pipeline()
        extraction_result = extraction_pipeline.extract(document)

        # 5. Interpretation & Explanation
        interpreter = get_interpreter()
        interpretation_result = interpreter.interpret(extraction_result)
        interpretation_result = explain_and_summarise(interpretation_result)

        extraction_dict = interpretation_result.extraction.to_dict()
        interpretation_dict = interpretation_result.to_dict()

        # Build payload
        payload = {
            "file_id": file_id,
            "filename": file.filename,
            "pipeline_stage": "pdf_preprocessing_ocr_extraction_interpretation",
            "disclaimer": interpretation_dict.get("disclaimer"),
            "ocr": ocr_summary,
            "document": document.to_dict(),
            "extraction": extraction_dict,
            "interpretation": interpretation_dict,
        }

        # Save prediction
        output_path = PREDICTIONS_DIR / f"{file_id}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return payload

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_with_report(req: ChatRequest):
    """Accepts an optional report file_id and user question, returning a context-grounded answer."""
    report_data = None
    if req.file_id and req.file_id.strip():
        clean_id = _validate_safe_id(req.file_id)
        output_path = PREDICTIONS_DIR / f"{clean_id}.json"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail=f"Report with ID '{clean_id}' not found.")
        report_data = json.loads(output_path.read_text(encoding="utf-8"))

    try:
        chatbot = get_chatbot()
        res = chatbot.answer_question(report_data, req.question, history=req.history)
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/info")
def get_chat_info():
    """Returns metadata about the active LLM / Clinical Knowledge Engine."""
    chatbot = get_chatbot()
    return chatbot.llm_provider.get_info()


@app.get("/report/{report_id}")
def get_report(report_id: str):
    """Fetches a previously processed report by its sanitized ID."""
    clean_id = _validate_safe_id(report_id)
    output_path = PREDICTIONS_DIR / f"{clean_id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    data = json.loads(output_path.read_text(encoding="utf-8"))
    return data


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Production Single-Service Frontend Serving (React SPA)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static_assets")

    @app.get("/")
    def serve_frontend_root():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        # Avoid intercepting API routes
        if full_path in ("health", "chat", "upload") or full_path.startswith(("report/", "chat/")):
            raise HTTPException(status_code=404, detail="API route not found.")
        target = FRONTEND_DIST / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")
