# uploader.py
"""
Firebase uploader for AirNote saves.

Uploads composite images and JSON artifacts to Firebase Storage and writes
a metadata document to Firestore.

Requirements:
    pip install firebase-admin google-cloud-storage

Environment:
    GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service_account.json

Usage:
    from uploader import FirebaseUploader

    up = FirebaseUploader(project_id="your-project-id", bucket_name="your-bucket.appspot.com")
    out = up.upload_save(
        session_id="20250101_ab12cd",
        save_ts=int(time.time()),
        files={
            "composite_jpg": "/abs/path/out/<session>/save_composite_0_1700000000.jpg",
            "strokes_json": "/abs/path/out/<session>/save_strokes_0_1700000000.json",
            "meta_json": "/abs/path/out/<session>/save_meta_1700000000.json",
        },
        meta={
            "W": 1200, "H": 800, "page_idx": 0, "page_count": 3,
            "color_idx": 0, "draw_width": 5, "hotbar_idx": 0,
            "H0": None, "curr_quad": None,
        },
        make_public=True,
        extra_fields={"source": "AirNote", "auto": False, "prefix": "save"},
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, storage, firestore
except Exception as e:
    raise RuntimeError(
        "firebase-admin is required. Install with: pip install firebase-admin google-cloud-storage"
    ) from e


class FirebaseUploader:
    """
    Simple helper for uploading AirNote artifacts to Firebase Storage and recording
    a Firestore document for each save under:
        sessions/<session_id>/saves/<save_ts>
    """

    def __init__(self, project_id: str, bucket_name: str, init_app: bool = True):
        """
        Args:
            project_id: GCP/Firebase project id (e.g., "myapp-prod")
            bucket_name: Firebase Storage bucket (e.g., "myapp-prod.appspot.com")
            init_app: set False only if you initialize firebase elsewhere in your process
        """
        if not project_id or not bucket_name:
            raise ValueError("FirebaseUploader: project_id and bucket_name are required")

        self.project_id = project_id
        self.bucket_name = bucket_name

        if init_app and not firebase_admin._apps:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if not cred_path or not Path(cred_path).exists():
                raise RuntimeError(
                    "GOOGLE_APPLICATION_CREDENTIALS not set or file not found. "
                    "Set it to a service account JSON for your Firebase project."
                )
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id, "storageBucket": bucket_name})

        # Clients
        self.bucket = storage.bucket(bucket_name)
        self.db = firestore.client(project=project_id)

    def _blob_path(self, session_id: str, filename: str) -> str:
        """
        Sessions are grouped under sessions/<session_id>/.
        """
        session_id = str(session_id).strip()
        filename = str(filename).strip()
        return f"sessions/{session_id}/{filename}"

    def _upload_file(self, local_path: str, dest_path: str, make_public: bool = True) -> str:
        """
        Upload a single file to Storage.

        Returns:
            Public URL if make_public, else a gs:// URL.
        """
        p = Path(local_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(str(p))

        if make_public:
            blob.make_public()
            return blob.public_url

        return f"gs://{self.bucket_name}/{dest_path}"

    def upload_save(
        self,
        session_id: str,
        save_ts: int,
        files: Dict[str, str],
        meta: Dict[str, Any],
        make_public: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upload artifacts and write a Firestore document describing the save.

        Args:
            session_id: logical session identifier
            save_ts: unix timestamp (seconds)
            files: mapping of short key -> absolute local path
                   expected keys typically include "composite_jpg", "strokes_json", "meta_json"
            meta: metadata dictionary for the canvas/page/homography, etc.
            make_public: if True, make uploaded blobs world-readable and return https URLs
            extra_fields: additional fields to merge into the Firestore document root

        Returns:
            dict containing:
                {
                    "firestore_path": "sessions/<session_id>/saves/<save_ts>",
                    "files": { short_key: url, ... }
                }
        """
        if not session_id:
            raise ValueError("upload_save: session_id is required")
        if not isinstance(save_ts, int) or save_ts <= 0:
            raise ValueError("upload_save: save_ts must be a positive integer")

        # 1) Upload artifacts to Storage
        urls: Dict[str, str] = {}
        for short, local in files.items():
            filename = Path(local).name
            dest = self._blob_path(session_id, filename)
            url = self._upload_file(local, dest, make_public=make_public)
            urls[short] = url

        # 2) Build document
        doc = {
            "session_id": session_id,
            "save_ts": int(save_ts),
            "save_iso": datetime.utcfromtimestamp(int(save_ts)).isoformat() + "Z",
            "files": urls,
            "meta": dict(meta or {}),
        }
        if extra_fields:
            doc.update(dict(extra_fields))

        # 3) Write to Firestore
        doc_ref = (
            self.db.collection("sessions")
            .document(session_id)
            .collection("saves")
            .document(str(int(save_ts)))
        )
        doc_ref.set(doc)

        return {
            "firestore_path": f"sessions/{session_id}/saves/{int(save_ts)}",
            "files": urls,
        }
