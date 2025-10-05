# capture_app/uploader.py
import os
import io
import json
import time
import math
import mimetypes
from typing import Dict, Any, Optional

import numpy as np

# --- Optional Gemini captioner (only used if available and enabled) ---
_AI_AVAILABLE = False
try:
    # Local client you may use elsewhere
    # ask_gemini(composite_bgr, question, system_hint=None, host="127.0.0.1", port=8000, max_chars=140)
    from .ai_client import ask_gemini  # noqa: F401
    _AI_AVAILABLE = True
except Exception:
    _AI_AVAILABLE = False

# --- Optional OpenCV for re-encoding to PNG / reading images ---
_CV2_AVAILABLE = False
try:
    import cv2  # noqa: F401
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

# --- Optional requests for ingest_note POST (multipart) ---
_REQUESTS_AVAILABLE = False
try:
    import requests  # noqa: F401
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False

import firebase_admin
from firebase_admin import credentials, storage, firestore


# -----------------------------
# Helpers to sanitize Firestore
# -----------------------------
def _clean_for_firestore(x: Any):
    if isinstance(x, dict):
        return {str(k): _clean_for_firestore(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean_for_firestore(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean_for_firestore(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _safe_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make meta Firestore-safe:
    - No nested arrays (array of arrays).
    - Replace NaN/Inf with None.
    """
    meta2 = dict(meta) if meta else {}

    # H0: 3x3 -> dict rows {'r0': [...], 'r1': [...], 'r2': [...]}
    H0 = meta2.get("H0", None)
    if H0 is not None:
        H0_arr = np.array(H0, dtype=float)
        try:
            rows = H0_arr.tolist()
        except Exception:
            rows = []
        H0_map = {}
        for i, row in enumerate(rows):
            H0_map[f"r{i}"] = [_safe_float(v) for v in row]
        meta2["H0"] = H0_map

    # curr_quad: [[x,y], ...] -> [{'x':x, 'y':y}, ...]
    cq = meta2.get("curr_quad", None)
    if cq is not None:
        try:
            cq_list = np.array(cq, dtype=float).tolist()
        except Exception:
            cq_list = []
        cq_objs = []
        for p in cq_list:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x = _safe_float(p[0])
                y = _safe_float(p[1])
                cq_objs.append({"x": x, "y": y})
        meta2["curr_quad"] = cq_objs

    # Sanitize lingering numpy/NaN in remaining fields
    meta2 = _clean_for_firestore(meta2)
    return meta2


_DEFAULT_CRED = "serviceAccountKey.json"


def _guess_content_type(path: str) -> str:
    ctype, _ = mimetypes.guess_type(path)
    return ctype or "application/octet-stream"


def _make_requests_session(
    total_retries: int = 2,
    backoff: float = 0.25,
    timeout_connect: float = 3.0,
    timeout_read: float = 30.0
):
    """
    Create a small retrying requests session for robustness.
    """
    if not _REQUESTS_AVAILABLE:
        return None

    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))

    # attach default timeouts via wrapper
    def _request(method, url, **kwargs):
        timeout = kwargs.pop("timeout", (timeout_connect, timeout_read))
        return s.request(method, url, timeout=timeout, **kwargs)

    s.request_with_timeout = _request  # type: ignore[attr-defined]
    return s


# -----------------------------
# Timestamp helpers
# -----------------------------
def _ts_epoch(now: Optional[float] = None) -> int:
    """Unix epoch seconds."""
    return int(now if now is not None else time.time())


def _ts_bucket_yyyymmddhh(now: Optional[float] = None) -> str:
    """Return 'YYYYMMDDHH' to match your curl TS trimming (first 10 chars of WMIC)."""
    return time.strftime("%Y%m%d%H", time.localtime(now if now is not None else time.time()))


class FirebaseUploader:
    """
    Handles:
      1) Upload composite/strokes/meta files to GCS
      2) Write a Firestore doc under: /<fs_root>/<sessionId>/<fs_subcol>/<tsEpoch>
      3) (Optional) POST to local `ingest_note` with multipart form-data including the composite image

    Defaults mirror your working curl:
      - timestamp sent as YYYYMMDDHH
      - only (session_id, timestamp, image)
      - image uploaded as PNG (type=image/png)
    """

    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        cred_path: Optional[str] = None,
        storage_prefix: str = "snapshots",
        fs_root: str = "sessions",
        fs_subcol: str = "saves"
    ):
        if not project_id or not bucket_name:
            raise ValueError("project_id and bucket_name are required")

        cred_path = cred_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or _DEFAULT_CRED
        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                f"Service account JSON not found at: {cred_path}\n"
                "Place your downloaded key there or set GOOGLE_APPLICATION_CREDENTIALS."
            )

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                "projectId": project_id,
                "storageBucket": bucket_name
            })

        self.db = firestore.client()
        self.bucket = storage.bucket(bucket_name)

        self.storage_prefix = storage_prefix.strip("/ ")
        self.fs_root = fs_root.strip("/ ")
        self.fs_subcol = fs_subcol.strip("/ ")

        # requests session for ingest_note (if available)
        self._http = _make_requests_session()

    # --------- Cloud Storage ----------
    def _upload_one(self, local_path: str, dest_path: str, make_public: bool) -> Dict[str, Any]:
        blob = self.bucket.blob(dest_path)
        ctype = _guess_content_type(local_path)
        blob.upload_from_filename(local_path, content_type=ctype)
        url = None
        if make_public:
            blob.make_public()
            url = blob.public_url
        return {"bucket": self.bucket.name, "path": dest_path, "contentType": ctype, "publicUrl": url}

    # --------- Optional AI caption ----------
    def _try_gemini_caption(self, composite_path: str, host: str, port: int, max_chars: int = 240) -> Optional[str]:
        if not (_AI_AVAILABLE and _CV2_AVAILABLE):
            return None
        try:
            import cv2  # local import to be safe in constrained envs
            img = cv2.imread(composite_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            system_hint = (
                "You are a vision assistant. Briefly describe the whiteboard content for search and recall. "
                "Mention visible text, shapes, axes, and high-level topic in one or two sentences."
            )
            q = "Summarize this board so a user can find it later."
            ans = ask_gemini(img, q, system_hint=system_hint, host=host, port=port, max_chars=max_chars)  # type: ignore[name-defined]
            if not isinstance(ans, str):
                return None
            # Strip control chars just in case
            return "".join(ch for ch in ans if ch.isprintable())
        except Exception:
            return None

    # --------- Optional POST to ingest_note ----------
    def _post_to_ingest(
        self,
        ingest_url: str,
        session_id: str,
        timestamp_out: str,
        image_path: str,
        note: Optional[str] = None,
        extra_form: Optional[Dict[str, Any]] = None,
        *,
        force_png: bool = True,
        minimal: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        POST multipart/form-data to your local service.

        When minimal=True, this mirrors your curl exactly:

          curl -v -X POST "http://127.0.0.1:5050/ingest_note" \
            -H "Content-Type: multipart/form-data" \
            -F "session_id=$SESSION" \
            -F "timestamp=$TS" \
            -F "image=@/path/to/file.png;type=image/png";

        Otherwise, optional fields (note, doc_path, storage_json, meta_json) can be sent.
        """
        if not ingest_url:
            return None
        if not _REQUESTS_AVAILABLE or self._http is None:
            print("[Uploader] requests not available; skipping ingest POST.")
            return None
        try:
            # Content type exactly like curl when force_png=True
            ctype = "image/png" if force_png else _guess_content_type(image_path)
            files = {"image": (os.path.basename(image_path), open(image_path, "rb"), ctype)}
            data = {
                "session_id": str(session_id),
                "timestamp": str(timestamp_out),   # <= already pre-formatted
            }

            if not minimal:
                if note:
                    data["note"] = note
                if extra_form:
                    for k, v in extra_form.items():
                        data[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else str(v)

            resp = self._http.request_with_timeout("POST", ingest_url, files=files, data=data)  # type: ignore[attr-defined]
            try:
                return resp.json()
            except Exception:
                return {"status": resp.status_code, "text": resp.text[:500]}
        except Exception as e:
            print(f"[Uploader] ingest POST failed: {e}")
            return None

    # --------- Public API ----------
    def upload_save(
        self,
        session_id: str,
        save_ts: Optional[int],
        files: Dict[str, str],
        meta: Dict[str, Any],
        make_public: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
        *,
        # Local ingest behavior (defaults mirror your curl)
        ingest_url: str = "http://127.0.0.1:5050/ingest_note",
        use_gemini: Optional[bool] = None,
        gemini_host: str = "127.0.0.1",
        gemini_port: int = 8000,
        ingest_minimal: bool = True,
        ingest_force_png: bool = True,
        # Timestamp output mode for POST (curl parity by default)
        ingest_timestamp_mode: str = "bucket10"  # "bucket10" -> YYYYMMDDHH, "epoch" -> seconds
    ) -> Dict[str, Any]:
        """
        Uploads artifacts to GCS, writes a Firestore doc, and (optionally) calls your local
        ingest endpoint. By default this mirrors your manual curl (minimal + PNG + YYYYMMDDHH).
        """
        now_epoch = _ts_epoch()
        ts_epoch = int(save_ts if save_ts is not None else now_epoch)
        ts_bucket = _ts_bucket_yyyymmddhh(ts_epoch)

        # Which timestamp string to send to ingest
        if ingest_timestamp_mode == "bucket10":
            timestamp_out = ts_bucket
        elif ingest_timestamp_mode == "epoch":
            timestamp_out = str(ts_epoch)
        else:
            # Fallback to curl-like bucket
            timestamp_out = ts_bucket

        base_prefix = f"{self.storage_prefix}/{session_id}/{ts_epoch}"

        uploaded: Dict[str, Any] = {}

        # 1) Upload files to GCS
        comp_path_local = None
        if (p := files.get("composite_jpg")) and os.path.exists(p):
            dest = f"{base_prefix}/composite{os.path.splitext(p)[1] or '.jpg'}"
            uploaded["composite"] = self._upload_one(p, dest, make_public)
            comp_path_local = p

        if (p := files.get("strokes_json")) and os.path.exists(p):
            dest = f"{base_prefix}/strokes.json"
            uploaded["strokes"] = self._upload_one(p, dest, make_public)

        if (p := files.get("meta_json")) and os.path.exists(p):
            dest = f"{base_prefix}/meta.json"
            uploaded["meta"] = self._upload_one(p, dest, make_public)

        # 2) Normalize + write Firestore doc
        meta_norm = _normalize_meta(meta)
        meta_clean = _clean_for_firestore(meta_norm)

        doc = {
            "sessionId": str(session_id),
            "saveTsEpoch": ts_epoch,      # numeric epoch (doc id basis)
            "tsBucket": ts_bucket,        # e.g., 2025100511 for easy grouping / querying
            "createdAt": firestore.SERVER_TIMESTAMP,
            "storage": _clean_for_firestore(uploaded),
            "boardMeta": meta_clean,
            "public": bool(make_public)
        }
        if extra_fields:
            doc.update(_clean_for_firestore(extra_fields))

        ref = (
            self.db.collection(self.fs_root)
            .document(session_id)
            .collection(self.fs_subcol)
            .document(str(ts_epoch))
        )
        ref.set(doc)

        result = {"ref": ref.path, "storage": uploaded, "public": make_public}

        # 3) Optional: caption with Gemini + POST to ingest_note
        # Decide if we should run Gemini
        if use_gemini is None:
            use_gemini = _AI_AVAILABLE and _CV2_AVAILABLE

        # Prepare path for POST. If forcing PNG, re-encode to PNG first.
        post_image_path = comp_path_local
        if comp_path_local and ingest_force_png:
            try:
                if _CV2_AVAILABLE:
                    import cv2
                    img = cv2.imread(comp_path_local, cv2.IMREAD_COLOR)
                    if img is not None:
                        tmp_dir = os.path.dirname(comp_path_local)
                        tmp_png = os.path.join(tmp_dir, "composite_for_post.png")
                        cv2.imwrite(tmp_png, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                        post_image_path = tmp_png
                # If OpenCV not available, we still try with original path
            except Exception:
                post_image_path = comp_path_local

        ingest_result = None
        if ingest_url and post_image_path:
            note_text = None
            extras = None
            if not ingest_minimal:
                if use_gemini:
                    # only generate caption when we plan to send it
                    note_text = self._try_gemini_caption(
                        composite_path=post_image_path if ingest_force_png else comp_path_local,
                        host=gemini_host,
                        port=gemini_port,
                        max_chars=260
                    )
                extras = {
                    "doc_path": ref.path,
                    "storage_json": uploaded,
                    "meta_json": meta_clean
                }

            ingest_result = self._post_to_ingest(
                ingest_url=ingest_url,
                session_id=session_id,
                timestamp_out=timestamp_out,   # <= matches curl format by default
                image_path=post_image_path,
                note=note_text if not ingest_minimal else None,
                extra_form=extras if not ingest_minimal else None,
                force_png=ingest_force_png,
                minimal=ingest_minimal
            )

        if ingest_result is not None:
            result["ingest_result"] = ingest_result

        return result
