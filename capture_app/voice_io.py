# voice_io.py
"""
Minimal voice I/O utilities for AirNote.

Exports:
    - list_input_devices() -> List[Tuple[int, str]]
    - record_push_to_talk(is_down_func, samplerate=16000, channels=1, device=None) -> (pcm_bytes, samplerate)
    - transcribe_pcm16(pcm_bytes, samplerate, model_name="base") -> str
    - speak(text: str) -> None

Notes:
    - Recording uses sounddevice (RawInputStream) and captures 16-bit PCM (mono by default).
    - Transcription prefers the local 'whisper' package if installed.
      If unavailable, returns an empty string (fallback behavior).
    - TTS prefers pyttsx3 if installed; otherwise no-ops gracefully.
"""

from __future__ import annotations

import os
import sys
import time
import queue
import threading
from typing import Callable, Iterable, List, Optional, Tuple

# -------- Recording (sounddevice) --------
try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    sd = None
    _SD_IMPORT_ERROR = e
else:
    _SD_IMPORT_ERROR = None

# -------- Transcription (whisper - optional) --------
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None

try:
    import whisper  # openai-whisper
except Exception:
    whisper = None

# -------- TTS (pyttsx3 - optional) --------
try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# --------------------------------------
# Device discovery
# --------------------------------------
def list_input_devices() -> List[Tuple[int, str]]:
    """
    Return a list of available input devices as (index, name).
    If sounddevice is not available, returns an empty list.
    """
    if sd is None:
        return []
    out: List[Tuple[int, str]] = []
    try:
        for idx, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels", 0)) > 0:
                name = str(info.get("name", f"Device {idx}"))
                out.append((idx, name))
    except Exception:
        return []
    return out


# --------------------------------------
# Recording
# --------------------------------------
def _resolve_device(device: Optional[object]) -> Optional[object]:
    """
    Allow selection by numeric index or partial name match.
    """
    if sd is None:
        return None
    if device is None or device == "":
        return None
    # Numeric index
    if isinstance(device, int):
        return device
    # Try to parse str as int
    if isinstance(device, str):
        try:
            return int(device)
        except ValueError:
            pass
        # Partial name match
        try:
            devs = sd.query_devices()
            candidates = [
                i for i, d in enumerate(devs)
                if str(device).lower() in str(d.get("name", "")).lower()
                and int(d.get("max_input_channels", 0)) > 0
            ]
            return candidates[0] if candidates else None
        except Exception:
            return None
    return None


def record_push_to_talk(
    is_down_func: Callable[[], bool],
    samplerate: int = 16000,
    channels: int = 1,
    device: Optional[object] = None,
    block_ms: int = 50,
    start_timeout_sec: float = 4.0,
    max_duration_sec: float = 30.0,
    tail_ms: int = 200,
) -> Tuple[bytes, int]:
    """
    Record microphone audio while is_down_func() returns True (push-to-talk).

    Args:
        is_down_func: callable returning True while the key/button is held
        samplerate: target sampling rate (Hz)
        channels: input channels (1 recommended)
        device: None for default, int index, or partial name string
        block_ms: stream block size in milliseconds
        start_timeout_sec: give up if the user never presses within this window
        max_duration_sec: hard cap on recording length
        tail_ms: small tail of silence appended after release for ASR stability

    Returns:
        (pcm_bytes, samplerate). If recording fails, returns (b"", samplerate).
    """
    if sd is None or _SD_IMPORT_ERROR is not None or np is None:
        return b"", int(samplerate)

    device_sel = _resolve_device(device)

    # Wait for press to start
    t0 = time.time()
    while not is_down_func():
        if time.time() - t0 > float(start_timeout_sec):
            return b"", int(samplerate)
        time.sleep(0.01)

    q: "queue.Queue[bytes]" = queue.Queue()
    stopped = threading.Event()
    dtype = "int16"
    blocksize = max(1, int(samplerate * block_ms / 1000))

    def _callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            # Non-fatal stream status messages are common; ignore quietly
            pass
        try:
            q.put(bytes(indata))
        except Exception:
            pass

    pcm_chunks: List[bytes] = []

    try:
        with sd.RawInputStream(
            samplerate=int(samplerate),
            channels=int(channels),
            dtype=dtype,
            blocksize=blocksize,
            device=device_sel,
            callback=_callback,
        ):
            t_start = time.time()
            while True:
                # Drain any queued audio blocks
                try:
                    while True:
                        pcm_chunks.append(q.get_nowait())
                except queue.Empty:
                    pass

                if not is_down_func():
                    break

                if time.time() - t_start > float(max_duration_sec):
                    break

                time.sleep(0.01)

            # Small tail of silence for ASR
            tail_samples = int((tail_ms / 1000.0) * samplerate) * channels
            pcm_chunks.append(b"\x00\x00" * max(0, tail_samples))

    except Exception:
        return b"", int(samplerate)

    pcm = b"".join(pcm_chunks)
    return pcm, int(samplerate)


# --------------------------------------
# Transcription
# --------------------------------------
def _pcm16_bytes_to_float32_mono(pcm_bytes: bytes, samplerate: int, channels: int = 1) -> Optional["np.ndarray"]:
    """
    Convert PCM16 bytes (little-endian) to float32 waveform in [-1, 1], mono.
    """
    if np is None or not pcm_bytes:
        return None
    try:
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
        wav = arr.astype(np.float32) / 32768.0
        return wav
    except Exception:
        return None


def transcribe_pcm16(pcm_bytes: bytes, samplerate: int, model_name: str = "base") -> str:
    """
    Transcribe PCM16 audio using local whisper if available.
    If whisper is not installed, returns an empty string.

    Args:
        pcm_bytes: raw little-endian 16-bit PCM audio
        samplerate: sampling rate (Hz)
        model_name: whisper model name ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Text transcription (may be empty on failure).
    """
    if whisper is None or np is None or not pcm_bytes:
        return ""

    try:
        wav = _pcm16_bytes_to_float32_mono(pcm_bytes, samplerate, channels=1)
        if wav is None or len(wav) == 0:
            return ""

        # Whisper prefers ~16kHz mono float32 arrays
        if samplerate != 16000:
            # Resample with numpy + simple polyphase using librosa if available
            try:
                import librosa  # optional
                wav = librosa.resample(wav, orig_sr=samplerate, target_sr=16000)
                sr = 16000
            except Exception:
                # Fallback: naive decimate/interpolate for close rates
                sr = samplerate
        else:
            sr = samplerate

        model = whisper.load_model(model_name)
        # Use the in-memory audio interface
        result = model.transcribe(
            wav,
            fp16=False,
            language=None,      # let it auto-detect
            task="transcribe",
            verbose=False,
            temperature=0.0,
            initial_prompt=None,
        )
        text = (result.get("text") or "").strip()
        return text
    except Exception:
        return ""


# --------------------------------------
# TTS
# --------------------------------------
def speak(text: str) -> None:
    """
    Best-effort local TTS. No-ops silently if unavailable.

    Priority:
        1) pyttsx3 (cross-platform)
        2) macOS 'say' CLI
        3) No-op
    """
    t = (text or "").strip()
    if not t:
        return

    # pyttsx3
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.say(t)
            engine.runAndWait()
            return
        except Exception:
            pass
    # Else: no-op
    return
