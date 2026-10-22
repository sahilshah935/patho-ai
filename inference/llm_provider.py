"""Multi-Provider LLM Gateway for Pathology AI Chatbot.

Supports:
- Google Gemini (Free tier, top-grade clinical reasoning)
- Groq (Free tier, ultra-fast Llama-3 / Mixtral inference)
- OpenAI / OpenAI-compatible (GPT-4o, GPT-4o-mini, OpenRouter, LocalAI)
- Ollama (Local offline models: Llama-3, Mistral, Qwen, Meditron)
- Offline Medical Knowledge Engine (Fallback when no API key or network is available)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


# ---------------------------------------------------------------------------
# Simple zero-dependency .env loader
# ---------------------------------------------------------------------------
def _load_env_files():
    """Load environment variables from .env files if present."""
    possible_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for p in possible_paths:
        if p.exists() and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


_load_env_files()


class LLMProvider:
    """Manages connections to generative LLM backends with graceful fallback."""

    def __init__(self):
        self.provider: str = "offline"
        self.model_name: str = "KnowledgeEngine-v2"
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0
        self._detect_provider()

    def _detect_provider(self):
        """Detect and configure the best available LLM provider."""
        _load_env_files()
        explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()

        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "").strip()

        if explicit == "gemini" and gemini_key:
            self.provider = "gemini"
            self.model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        elif explicit == "groq" and groq_key:
            self.provider = "groq"
            self.model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif explicit == "openai" and openai_key:
            self.provider = "openai"
            self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        elif explicit == "ollama":
            self.provider = "ollama"
            self.model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
        # Auto-detect if not explicitly overridden
        elif gemini_key:
            self.provider = "gemini"
            self.model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        elif groq_key:
            self.provider = "groq"
            self.model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif openai_key:
            self.provider = "openai"
            self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        elif ollama_url:
            self.provider = "ollama"
            self.model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
        else:
            self.provider = "offline"
            self.model_name = "EmbeddedClinicalKnowledge"

    def get_info(self) -> Dict[str, Any]:
        """Return provider metadata for debugging and frontend UI status."""
        self._detect_provider()
        return {
            "provider": self.provider,
            "model": self.model_name,
            "is_cloud_llm": self.provider in ("gemini", "groq", "openai", "ollama"),
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.4,
    ) -> Optional[str]:
        """Generate response via configured LLM provider. Returns None if provider fails."""
        self._detect_provider()

        # Fast circuit breaker: if remote provider failed repeatedly, fall back instantly
        if self._consecutive_failures >= 2 and (time.time() - self._last_failure_time) < 45.0:
            return None

        if self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt, history, max_tokens, temperature)
        elif self.provider == "groq":
            return self._call_groq(prompt, system_prompt, history, max_tokens, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt, history, max_tokens, temperature)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt, history, max_tokens, temperature)

        return None

    # -----------------------------------------------------------------------
    # 1. Google Gemini API
    # -----------------------------------------------------------------------
    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        # Build sanitized contents array adhering strictly to Gemini's alternation rules
        raw_turns = []
        if history:
            for m in history[-6:]:  # Keep last 6 conversational turns
                role = "user" if m.get("sender") in ("user", "human") else "model"
                text = (m.get("text") or "").strip()
                if text:
                    raw_turns.append({"role": role, "parts": [{"text": text}]})

        # Gemini requires the first turn to be 'user'
        while raw_turns and raw_turns[0]["role"] == "model":
            raw_turns.pop(0)

        # Ensure strict alternating roles: user -> model -> user -> model
        contents = []
        for turn in raw_turns:
            if not contents:
                contents.append(turn)
            elif contents[-1]["role"] == turn["role"]:
                contents[-1]["parts"][0]["text"] += "\n\n" + turn["parts"][0]["text"]
            else:
                contents.append(turn)

        # Append current user prompt
        if contents and contents[-1]["role"] == "user":
            contents[-1]["parts"][0]["text"] += "\n\n" + prompt
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        # Candidate models to try in order (fast failover, max 1 fallback)
        candidate_models = [self.model_name]
        for fallback in ["gemini-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        try:
            with httpx.Client(timeout=8.0) as client:
                for model in candidate_models:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    res = client.post(url, json=payload, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                self.model_name = model
                                self._consecutive_failures = 0
                                return parts[0].get("text", "").strip()
                    elif res.status_code == 404:
                        # Model not available on this tier, try next candidate
                        continue
                    else:
                        print(f"[LLMProvider:Gemini] Warning HTTP {res.status_code} for {model}: {res.text[:150]}")
                        break
        except Exception as e:
            print(f"[LLMProvider:Gemini] Request failed: {e}")

        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        return None

    # -----------------------------------------------------------------------
    # 2. Groq Cloud API
    # -----------------------------------------------------------------------
    def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for m in history[-6:]:
                role = "user" if m.get("sender") in ("user", "human") else "assistant"
                text = m.get("text", "")
                if text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LLMProvider:Groq] Warning HTTP {res.status_code}: {res.text[:150]}")
        except Exception as e:
            print(f"[LLMProvider:Groq] Request failed: {e}")

        return None

    # -----------------------------------------------------------------------
    # 3. OpenAI / Compatible API
    # -----------------------------------------------------------------------
    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for m in history[-6:]:
                role = "user" if m.get("sender") in ("user", "human") else "assistant"
                text = m.get("text", "")
                if text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LLMProvider:OpenAI] Warning HTTP {res.status_code}: {res.text[:150]}")
        except Exception as e:
            print(f"[LLMProvider:OpenAI] Request failed: {e}")

        return None

    # -----------------------------------------------------------------------
    # 4. Ollama Local Endpoint
    # -----------------------------------------------------------------------
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for m in history[-6:]:
                role = "user" if m.get("sender") in ("user", "human") else "assistant"
                text = m.get("text", "")
                if text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLMProvider:Ollama] Request failed: {e}")

        return None
