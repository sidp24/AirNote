import os, firebase_admin
from firebase_admin import credentials, firestore, storage

SA_PATH = os.path.join(os.path.dirname(__file__), "serviceAccount.json")
BUCKET  = os.environ.get("FIREBASE_STORAGE_BUCKET")

if not BUCKET:
    raise RuntimeError("FIREBASE_STORAGE_BUCKET env var not set (e.g., '<project>.appspot.com')")

if not firebase_admin._apps:
    cred = credentials.Certificate(SA_PATH)
    firebase_admin.initialize_app(cred, {"storageBucket": BUCKET})

db = firestore.client()
bucket = storage.bucket()