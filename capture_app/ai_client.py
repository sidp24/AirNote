"""
ai_client.py — Local AI client for AirNote.

Features:
- Sends (downscaled) board composite + question to a local FastAPI /ask endpoint.
- Robust text sanitization (strip control chars, emojis, smart quotes → ASCII).
- JPEG base64 encoding with size cap to keep requests snappy.
- Requests session with retries and short connect/read timeouts (Windows-friendly).
- AI_OFFLINE mode for UI testing without the server.
- AI_DEBUG logs timing and payload sizes.
- ping_ai_server() helper for /health smoke tests.
"""

import base64
import os
import time
import re
from typing import Optional, Tuple

import cv2
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- Tunables ----
_MAX_EDGE = 1200            # Longest side for the image we send
_JPEG_QUALITY = 80
_CONNECT_TIMEOUT = 3.0      # seconds for TCP connect
_READ_TIMEOUT = 8.0         # seconds for server to respond (short to avoid UI "freezes")
_TOTAL_RETRIES = 2
_BACKOFF_FACTOR = 0.25
_MAX_CHARS_CLAMP = 1000

# ---- Env toggles ----
# AI_OFFLINE=1        -> skip HTTP and return placeholder (for UI testing)
# AI_SERVER_URL=...   -> override http://127.0.0.1:8000/ask
# AI_HEALTH_URL=...   -> override http://127.0.0.1:8000/health
# AI_DEBUG=1          -> print request/response sizes and timing

# --- Text sanitization ---
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_RE_EMOJI   = re.compile(
    "["                       # broad emoji-ish ranges
    "\U0001F300-\U0001FAD6"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F6FF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001FA70-\U0001FAFF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE
)

_SMART_QUOTES = {
    "“":"\"", "”":"\"", "„":"\"", "‟":"\"",
    "‘":"'",  "’":"'",  "‚":"'",  "‛":"'",
    "—":"-",  "–":"-",  "−":"-",  "-":"-",
}

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    s = _RE_CONTROL.sub(" ", s)
    s = _RE_EMOJI.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 800:
        s = s[:800]
    return s

def _fit_within(img_bgr, max_edge=_MAX_EDGE):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return img_bgr
    scale = max_edge / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _to_jpg_base64(img_bgr, quality=_JPEG_QUALITY) -> Tuple[bytes, int]:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    raw = buf.tobytes()
    b64 = base64.b64encode(raw)
    return b64, len(b64)

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=_TOTAL_RETRIES,
        read=_TOTAL_RETRIES,
        connect=_TOTAL_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _endpoint() -> str:
    return os.getenv("AI_SERVER_URL", "http://127.0.0.1:8000/ask")

def _health_endpoint() -> str:
    return os.getenv("AI_HEALTH_URL", "http://127.0.0.1:8000/health")

def _ai_debug() -> bool:
    return os.getenv("AI_DEBUG", "0") != "0"

def ping_ai_server(timeout: float = 2.0) -> bool:
    """
    Lightweight health check. Returns True if /health responds with HTTP 200.
    """
    if os.getenv("AI_OFFLINE", "0") == "1":
        return True
    url = _health_endpoint()
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False

# public API used by main.py
def ask_gemini(
    img_bgr,
    question: Optional[str],
    system_hint: Optional[str] = None,
    max_chars: int = 540
) -> str:
    """
    Send image + question to the local AI server. Returns answer text or a friendly error.

    Arguments:
      img_bgr:      OpenCV BGR image (numpy array), will be resized and JPEG base64'd
      question:     User question text
      system_hint:  Short system hint
      max_chars:    Upper bound on answer length (clamped to [_60, _MAX_CHARS_CLAMP])
    """
    # Offline/testing mode
    if os.getenv("AI_OFFLINE", "0") == "1":
        return "(AI offline) pretend-answer."

    # Sanitize text inputs
    q_norm = _normalize_text(question or "")
    hint_norm = _normalize_text(system_hint or "")

    if not q_norm:
        q_norm = "Summarize the whiteboard content briefly."

    max_chars = max(60, min(int(max_chars or 540), _MAX_CHARS_CLAMP))

    # Validate image
    if img_bgr is None or not hasattr(img_bgr, "shape") or len(img_bgr.shape) != 3:
        return "(AI) No image to send."

    # Prepare image
    img_small = _fit_within(img_bgr, _MAX_EDGE)
    b64, b64_len = _to_jpg_base64(img_small, _JPEG_QUALITY)

    payload = {
        "question": q_norm,
        "system_hint": hint_norm,
        "image_b64_jpg": b64.decode("ascii"),
        "max_chars": max_chars,
    }

    url = _endpoint()
    sess = _make_session()

    t0 = time.time()
    try:
        if _ai_debug():
            print(f"[AI] q='{q_norm[:80]}' hint='{(hint_norm or '')[:60]}'")

        resp = sess.post(
            url,
            json=payload,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),  # (connect, read)
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.time() - t0

        if resp.status_code != 200:
            try:
                j = resp.json()
                msg = j.get("error") or j.get("detail") or j.get("message") or resp.text
            except Exception:
                msg = resp.text
            msg = _normalize_text(str(msg)) or f"HTTP {resp.status_code}"
            if _ai_debug():
                print(f"[AI] HTTP {resp.status_code} in {elapsed:.2f}s | msg={msg}")
            return f"(AI) {msg}"

        j = resp.json()
        ans = _normalize_text(j.get("answer") or "")
        if not ans:
            ans = "(AI) Empty answer."
        if _ai_debug():
            print(f"[AI] POST {url} | img_b64={b64_len}B max_chars={max_chars}")
            print(f"[AI] ok in {elapsed:.2f}s | answer_len={len(ans)}")
        return ans

    except requests.exceptions.ConnectTimeout:
        return "(AI) Server connect timeout."
    except requests.exceptions.ReadTimeout:
        return "(AI) Server read timeout."
    except requests.exceptions.ConnectionError:
        return "(AI) Connection error."
    except Exception as e:
        msg = _normalize_text(str(e))
        if len(msg) > 160:
            msg = msg[:160] + "..."
        return f"(AI) {msg}"

__all__ = ["ask_gemini", "ping_ai_server"]
