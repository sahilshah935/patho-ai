# 🧬 PathO — Clinical Pathology Report Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8-646cff.svg)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PathO** is an enterprise-grade multimodal clinical AI platform designed to extract, interpret, and explain routine and complex clinical pathology reports (CBC, CMP, LFT, Lipid Panels, Thyroid, Renal, etc.). It pairs rule-governed clinical reference ranges with state-of-the-art LLM medical reasoning (Google Gemini, Groq, OpenAI, Ollama, and a fully offline clinical fallback engine).

---

## 🌟 Key Features

- **Multimodal Document Processing:** Automatic PDF parsing and OCR text extraction for digitized lab reports.
- **Precision Biomarker Extraction:** Robust clinical schema normalizing over 60+ biomarkers with biological range evaluation (LOW, BORDERLINE, NORMAL, HIGH, CRITICAL).
- **Multi-Engine Medical Intelligence Gateway:**
  - **Google Gemini** (`gemini-flash-latest` / `gemini-1.5-flash`) for clinical case synthesis.
  - **Groq Cloud** (`llama-3.3-70b-versatile`) for ultra-low latency streaming inference.
  - **OpenAI** (`gpt-4o-mini` / `gpt-4o`).
  - **Ollama** for local, private offline clinical workflows.
  - **Offline Medical Knowledge Base:** Built-in monographs, differential diagnoses, biomarker correlations, and dietary guidance that function with zero internet or API keys.
- **Conversational Health Copilot:** Ask questions with an uploaded report or ask general health/medical questions directly.
- **Single-Service Architecture:** FastAPI serves both the high-performance REST API and the bundled React SPA from a single port.

---

## 🚀 Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/sahilshah935/patho-ai.git
cd patho-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (cmd):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your Google Gemini API key (or Groq / OpenAI):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Full Stack Application
```bash
# Start FastAPI (automatically serves API + React frontend)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

---

## 🌐 Cloud Deployment Options

### Option 1: Render.com (Recommended Free / Low Cost)
1. Fork or push this repository to GitHub.
2. Sign up on [Render.com](https://render.com) and click **New + > Web Service**.
3. Connect your `patho-ai` repository.
4. Set the configuration:
   - **Environment:** `Python 3` (or `Docker`)
   - **Build Command:** `pip install --no-cache-dir -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. In **Environment Variables**, add:
   - `GEMINI_API_KEY` = `your_gemini_api_key`
6. Click **Deploy Web Service**!

---

### Option 2: Railway.app
1. Go to [Railway.app](https://railway.app) and select **New Project > Deploy from GitHub repo**.
2. Select `sahilshah935/patho-ai`.
3. In the project settings, add the environment variable `GEMINI_API_KEY`.
4. Railway will automatically detect the `Procfile` / `Dockerfile` and deploy the service.

---

### Option 3: Docker
```bash
# Build the Docker image
docker build -t patho-ai .

# Run container
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key" patho-ai
```
Access at `http://localhost:8000`.

---

## 🔒 Security & Medical Disclaimer

- **Privacy:** Lab reports and API keys are strictly guarded. Uploaded files remain isolated and `.env` files are excluded from git.
- **Clinical Disclaimer:** PathO is an investigational decision-support and educational tool. It does not provide medical diagnoses or replace consultations with licensed physicians. Always consult a qualified medical professional for clinical decisions.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
