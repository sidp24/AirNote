"""
Firebase uploader for AirNote saves.

Requirements:
  pip install firebase-admin google-cloud-storage

Environment:
  GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service_account.json

Usage:
  from uploader import FirebaseUploader
  up = FirebaseUploader(project_id="your-project-id", bucket_name="your-bucket.appspot.com")
  up.upload_save(session_id, save_ts, files_dict, meta_dict)
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, storage, firestore

class FirebaseUploader:
    def __init__(self, project_id: str, bucket_name: str, init_app: bool = True):
        """
        project_id: GCP/Firebase project id (e.g., "myapp-prod")
        bucket_name: Firebase Storage bucket (e.g., "myapp-prod.appspot.com")
        init_app: set False only if you init firebase elsewhere
        """
        self.project_id = project_id
        self.bucket_name = bucket_name

        if init_app and not firebase_admin._apps:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not cred_path or not Path(cred_path).exists():
                raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set or file not found.")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                "projectId": project_id,
                "storageBucket": bucket_name
            })

        self.bucket = storage.bucket(bucket_name)
        self.db = firestore.client(project=project_id)

    def _blob_path(self, session_id: str, filename: str) -> str:
        return f"sessions/{session_id}/{filename}"

    def _upload_file(self, local_path: str, dest_path: str, make_public: bool = True) -> str:
        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(local_path)
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
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Uploads files and writes a Firestore document.

        files: mapping of short keys -> local paths, e.g.:
          {
            "board_png": "/abs/out/<session>/board_0_<ts>.png",
            "board_preview_png": "...",
            "strokes_json": "...",
            "meta_json": "..."
          }

        meta: metadata dict to include in Firestore (e.g., page_idx, H0, curr_quad, width, etc.)
        extra_fields: optional extra data to merge into Firestore doc

        Returns: dict with uploaded URLs and Firestore doc path.
        """
        # 1) Upload to Storage
        urls = {}
        for short, local in files.items():
            fname = os.path.basename(local)
            dest = self._blob_path(session_id, fname)
            url = self._upload_file(local, dest, make_public=make_public)
            urls[short] = url

        # 2) Build Firestore document
        doc = {
            "session_id": session_id,
            "save_ts": save_ts,
            "save_iso": datetime.utcfromtimestamp(save_ts).isoformat() + "Z",
            "files": urls,
            "meta": meta,
        }
        if extra_fields:
            doc.update(extra_fields)

        # 3) Write: sessions/<session_id>/saves/<save_ts>
        doc_ref = self.db.collection("sessions").document(session_id)\
                         .collection("saves").document(str(save_ts))
        doc_ref.set(doc)

        return {
            "firestore_path": f"sessions/{session_id}/saves/{save_ts}",
            "files": urls
        }
