import firebase_admin
from firebase_admin import credentials, storage, firestore
import uuid
import datetime

# Path to service account JSON
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "YOUR_PROJECT_ID.appspot.com"
})

db = firestore.client()
bucket = storage.bucket()
