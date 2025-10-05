# ai_service/firebase_admin_client.py
import os, json
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Resolve service account path
HERE = os.path.dirname(__file__)
SA_PATH = os.path.join(HERE, "serviceAccount.json")
if not os.path.exists(SA_PATH):
    raise RuntimeError(f"Missing serviceAccount.json at: {SA_PATH}")

# Resolve bucket name
bucket_env = os.environ.get("FIREBASE_STORAGE_BUCKET", "").strip()

# Fallback: derive "<project>.appspot.com" from service account if env not set
if not bucket_env:
    with open(SA_PATH, "r", encoding="utf-8") as f:
        proj_id = json.load(f).get("project_id")
    if not proj_id:
        raise RuntimeError("Could not derive project_id from serviceAccount.json")
    bucket_env = f"{proj_id}.appspot.com"

# Initialize exactly once
if not firebase_admin._apps:
    cred = credentials.Certificate(SA_PATH)
    firebase_admin.initialize_app(cred, {"storageBucket": bucket_env})

# Expose Firestore and Storage handles
db = firestore.client()
bucket = storage.bucket()
