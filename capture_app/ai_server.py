"""
FastAPI Gemini microservice — vision-only ASK (no voice)

Endpoints
---------
GET  /health
POST /ask
    Request JSON:
      {
        "question": "Your concise question or analysis instruction",
        "system_hint": "Optional system guidance string",
        "image_b64_jpg": "<base64-encoded .jpg of the rectified board composite>",
        "max_chars": 300
      }
    Response JSON:
      { "answer": "Model's concise reply (<= max_chars, single line)" }

Environment
-----------
- Set your Google API key in a .env file or the environment:
    GEMINI_API_KEY="AIza..."
- Optional overrides:
    GEMINI_MODEL="gemini-2.5-flash"
    GEMINI_TEMPERATURE="0.2"
    GEMINI_TOP_P="0.95"
    GEMINI_TOP_K="40"

Run
---
pip install fastapi uvicorn[pstandard] pydantic pillow python-dotenv google-generativeai
uvicorn ai_server:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image
from dotenv import load_dotenv

import google.generativeai as genai

# ---------------------------
# Config & Model bootstrap
# ---------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# A fast, cheap, strong vision model. Adjust if needed.
# Default model (vision-capable). Adjust via GEMINI_MODEL env var if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

try:
    TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
except Exception:
    TEMPERATURE = 0.2

try:
    TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
except Exception:
    TOP_P = 0.95

try:
    TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
except Exception:
    TOP_K = 40

genai.configure(api_key=GEMINI_API_KEY)

# Construct once; reuse between requests.
MODEL = genai.GenerativeModel(GEMINI_MODEL)

GENERATION_CONFIG = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "max_output_tokens": 512,   # character clamp is handled separately
}

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="AirNote Gemini Service", version="1.0.0")

# Allow localhost apps to call us without hassle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------
# Schemas
# ---------------------------
class AskRequest(BaseModel):
    question: Optional[str] = Field(default="Analyze the whiteboard area based ONLY on the image.")
    system_hint: Optional[str] = Field(
        default=(
            "You are a vision-first assistant. Infer from the image only. "
            "Extract legible text, math, diagrams, tables, and relationships. "
            "Be concrete and concise."
        )
    )
    image_b64_jpg: str
    max_chars: Optional[int] = Field(default=360, ge=60, le=1200)

    @validator("image_b64_jpg")
    def _validate_b64(cls, v: str) -> str:
        if not v:
            raise ValueError("image_b64_jpg is required")
        if v.startswith("data:image"):
            try:
                v = v.split(",", 1)[1]
            except Exception:
                pass
        try:
            _ = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"image_b64_jpg must be valid base64: {e}")
        return v


class AskResponse(BaseModel):
    answer: str

# ---------------------------
# Utilities
# ---------------------------
def _decode_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    img = Image.open(buf).convert("RGB")
    return img

def _single_line(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text.strip())
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("```", "")
    return text

def _clamp_chars(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    cut = limit
    window = s[:limit]
    dot = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if dot >= limit - 40:
        cut = dot + 1
    return s[:cut].rstrip()

def _hud_safe(text: str) -> str:
    repl = {
        "•": "-", "·": "-", "‣": "-", "∙": "-",
        "…": "...", "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-",
        "\u00a0": " ",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("ascii", "ignore").decode("ascii")

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health_check():
    """Lightweight liveness / readiness probe (no model call)."""
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        # 1) Decode image → JPEG bytes (stable path for inline_data)
        img = _decode_image(req.image_b64_jpg)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        jpeg_bytes = buf.getvalue()

        # 2) Build the prompt parts (system hint → image → question)
        parts = [
            req.system_hint or "",
            {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes}},
            (req.question or "").strip() or "Be brief.",
        ]

        # 3) Generate (don’t use resp.text)
        resp = MODEL.generate_content(
            contents=[{"role": "user", "parts": parts}],
            generation_config=GENERATION_CONFIG,
            safety_settings=None,
        )

        # 4) Safely extract first candidate’s text (no .text access!)
        def extract_text(r) -> str:
            try:
                cands = getattr(r, "candidates", None) or []
                if not cands:
                    return ""
                c0 = cands[0]
                # Handle safety/finish_reason cases explicitly
                # (SDK types differ, so use getattr defensively)
                finish = getattr(c0, "finish_reason", None)
                if str(finish).lower() in {"safety", "blocked", "recitation", "other"}:
                    return "(AI) Response blocked by safety filters."
                content = getattr(c0, "content", None)
                parts = getattr(content, "parts", None) or []
                # Concatenate all text parts
                out = []
                for p in parts:
                    # p.text exists for text parts; otherwise skip
                    t = getattr(p, "text", None)
                    if t:
                        out.append(str(t))
                return " ".join(out).strip()
            except Exception:
                return ""

        raw = extract_text(resp)
        if not raw:
            # Try to surface safety info if present
            try:
                c0 = resp.candidates[0]
                safety = getattr(c0, "safety_ratings", None)
                if safety:
                    # Compact safety reasons
                    reasons = []
                    for s in safety:
                        cat = getattr(s, "category", "unknown")
                        sev = getattr(s, "probability", getattr(s, "severity", ""))
                        reasons.append(f"{cat}:{sev}")
                    msg = " (AI) No answer (safety). " + ", ".join(reasons)
                    return AskResponse(answer=_hud_safe(_clamp_chars(msg, int(req.max_chars or 360))))
            except Exception:
                pass
            return AskResponse(answer="(AI) No answer.")

        # 5) Clamp & HUD-safe
        text = _single_line(raw)
        text = _clamp_chars(text, int(req.max_chars or 360))
        text = _hud_safe(text)
        return AskResponse(answer=text)

    except Exception as e:
        # Return a friendly one-liner; avoid long tracebacks in HUD
        return AskResponse(answer=f"(AI) {str(e)[:180]}")

# ---------------------------
# Dev server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    try:
        port = int(os.getenv("PORT", "8000"))
    except Exception:
        port = 8000
    uvicorn.run("ai_server:app", host=host, port=port, reload=True)
