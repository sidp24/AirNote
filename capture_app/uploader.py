# capture_app/uploader.py
import os, time, mimetypes, math, json
from typing import Dict, Any, Optional
import numpy as np

import firebase_admin
from firebase_admin import credentials, storage, firestore

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

    # H0: 3x3 -> dict of rows {'r0': [...], 'r1': [...], 'r2': [...]}
    H0 = meta2.get("H0", None)
    if H0 is not None:
        H0_arr = np.array(H0, dtype=float)
        rows = []
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

    # Sanitize any lingering numpy/NaN in remaining fields
    meta2 = _clean_for_firestore(meta2)
    return meta2

_DEFAULT_CRED = "serviceAccountKey.json"

def _guess_content_type(path: str) -> str:
    ctype, _ = mimetypes.guess_type(path)
    return ctype or "application/octet-stream"

class FirebaseUploader:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        cred_path: Optional[str] = None,
        storage_prefix: str = "snapshots",
        fs_root: str = "sessions",
        fs_subcol: str = "saves"    # <-- allow subcollection override
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
        self.fs_root  = fs_root.strip("/ ")
        self.fs_subcol = fs_subcol.strip("/ ")

    def _upload_one(self, local_path: str, dest_path: str, make_public: bool) -> Dict[str, Any]:
        blob = self.bucket.blob(dest_path)
        ctype = _guess_content_type(local_path)
        blob.upload_from_filename(local_path, content_type=ctype)
        url = None
        if make_public:
            blob.make_public()
            url = blob.public_url
        return {"bucket": self.bucket.name, "path": dest_path, "contentType": ctype, "publicUrl": url}

    def upload_save(
        self,
        session_id: str,
        save_ts: int,
        files: Dict[str, str],
        meta: Dict[str, Any],
        make_public: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ts = int(save_ts or time.time())
        base_prefix = f"{self.storage_prefix}/{session_id}/{ts}"

        uploaded: Dict[str, Any] = {}

        if (p := files.get("composite_jpg")) and os.path.exists(p):
            dest = f"{base_prefix}/composite{os.path.splitext(p)[1] or '.jpg'}"
            uploaded["composite"] = self._upload_one(p, dest, make_public)

        if (p := files.get("strokes_json")) and os.path.exists(p):
            dest = f"{base_prefix}/strokes.json"
            uploaded["strokes"] = self._upload_one(p, dest, make_public)

        if (p := files.get("meta_json")) and os.path.exists(p):
            dest = f"{base_prefix}/meta.json"
            uploaded["meta"] = self._upload_one(p, dest, make_public)

        # --- Normalize then clean the meta for Firestore ---
        meta_norm  = _normalize_meta(meta)
        meta_clean = _clean_for_firestore(meta_norm)

        doc = {
            "sessionId": str(session_id),
            "saveTs": int(ts),
            "createdAt": firestore.SERVER_TIMESTAMP,
            "storage": _clean_for_firestore(uploaded),
            "boardMeta": meta_clean,
            "public": bool(make_public)
        }
        if extra_fields:
            doc.update(_clean_for_firestore(extra_fields))

        # Debug (optional): print the doc to verify shape
        # print("Firestore doc to write:")
        # print(json.dumps(doc, indent=2))

        ref = (self.db.collection(self.fs_root)
                    .document(session_id)
                    .collection(self.fs_subcol)
                    .document(str(ts)))
        ref.set(doc)
        return {"ref": ref.path, "storage": uploaded, "public": make_public}
