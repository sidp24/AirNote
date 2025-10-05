# ai_service/main.py
import os, io, json, time, re
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai

from firebase_admin_client import db, bucket
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

import firebase_admin
from firebase_admin import credentials, storage
import os

from fastapi import Body
import requests
from io import BytesIO
from typing import List, Dict

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccount.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.environ.get("FIREBASE_STORAGE_BUCKET")
    })

# --- Gemini setup ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var is not set")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for hackathon, tighten later
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
def health():
    return {"ok": True, "model": MODEL_NAME}

# ---------- Labeling ----------
LABEL_PROMPT = (
    "You are labeling a single screenshot from an AR whiteboard/ROI. "
    "Return JSON with keys: label (2-3 words), summary (<=25 words). "
    "Focus on the main subject or concept visible."
)

@app.post("/label_note")
async def label_note(image: UploadFile = File(...)):
    try:
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        resp = model.generate_content(
            [LABEL_PROMPT, img],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 30},
        )
        try:
            js = json.loads(resp.text)
        except Exception:
            # Fallback: try to parse keys from freeform text
            txt = resp.text or ""
            m1 = re.search(r'"label"\s*:\s*"([^"]+)"', txt)
            m2 = re.search(r'"summary"\s*:\s*"([^"]+)"', txt)
            js = {
                "label": m1.group(1) if m1 else "Note",
                "summary": m2.group(1) if m2 else txt.strip()[:120],
            }
        return js
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/label_by_url")
def label_by_url(payload: dict = Body(...)):
    url = payload.get("imageURL") or payload.get("url")
    if not url:
        return JSONResponse({"error":"imageURL required"}, status_code=400)
    img_bytes = requests.get(url, timeout=30).content
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    resp = model.generate_content([LABEL_PROMPT, img],
      generation_config={"response_mime_type": "application/json"},
      request_options={"timeout": 30},
    )
    try:
      return json.loads(resp.text)
    except Exception:
      txt = resp.text or ""
      m1 = re.search(r'"label"\s*:\s*"([^"]+)"', txt)
      m2 = re.search(r'"summary"\s*:\s*"([^"]+)"', txt)
      return { "label": m1.group(1) if m1 else "Note", "summary": m2.group(1) if m2 else txt.strip()[:120] }

# ---------- Ingest (image + sidecars) ----------
@app.post("/ingest_note")
async def ingest_note(
    session_id: str = Form(...),
    timestamp: float = Form(...),
    image: UploadFile = File(...),
    meta: UploadFile | None = File(None),
    strokes: UploadFile | None = File(None),
):
    try:
        ts = int(timestamp)
        base = f"notes/{session_id}/{ts}"

        # 1) Upload composite image
        img_blob = bucket.blob(f"{base}/composite.jpg")
        img_blob.upload_from_file(image.file, content_type="image/jpeg")
        img_blob.make_public()
        image_url = img_blob.public_url

        # 2) Optional sidecars
        meta_path = None
        strokes_path = None
        if meta:
            m_blob = bucket.blob(f"{base}/meta.json")
            m_blob.upload_from_file(meta.file, content_type="application/json")
            meta_path = m_blob.name
        if strokes:
            s_blob = bucket.blob(f"{base}/strokes.json")
            s_blob.upload_from_file(strokes.file, content_type="application/json")
            strokes_path = s_blob.name

        # 3) Write Firestore doc
        doc_id = f"{session_id}_{ts}"
        db.collection("notes").document(doc_id).set(
            {
                "sessionId": session_id,
                "timestamp": ts,
                "title": f"Note {ts}",
                "imagePath": img_blob.name,
                "imageURL": image_url,
                "metaPath": meta_path,
                "strokesPath": strokes_path,
                "createdAt": SERVER_TIMESTAMP,
            },
            merge=True,
        )

        return {"ok": True, "id": doc_id, "imageURL": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Build a simple similarity graph (Obsidian-style) ----------
@app.post("/rebuild_graph")
def rebuild_graph():
    """
    Build a lightweight note graph from Firestore and store it at graphs/latest.
    Similarity is computed from AI tags + label/type if available.
    """
    # 1) Pull all notes
    notes_ref = db.collection("notes")
    docs = list(notes_ref.stream())

    notes: List[Dict] = []
    for d in docs:
        data = d.to_dict() or {}
        ai = data.get("ai") or {}
        tags = ai.get("tags") or []
        note = {
            "id": d.id,
            "title": data.get("title") or f"Note {d.id}",
            "label": data.get("label") or None,        # optional plain label
            "type": ai.get("type") or None,            # ai-assigned type
            "tags": tags,                               # ai-assigned tags (array)
            "timestamp": data.get("timestamp") or 0,
            "sessionId": data.get("sessionId") or None # used for graph similarity
        }
        notes.append(note)

    # 2) Nodes (clusterId falls back to label, then type, then 'unlabeled')
    nodes = []
    for n in notes:
        cluster = n.get("label") or n.get("type") or "unlabeled"
        nodes.append({
            "id": n["id"],
            "label": n.get("title") or n["id"],
            "clusterId": cluster,
        })

    # 3) Edges: tag overlap + same label/type
    def sim(a: Dict, b: Dict) -> float:
        at = set(a.get("tags") or [])
        bt = set(b.get("tags") or [])
        tag_overlap = len(at & bt)
        same_label = 1.0 if a.get("label") and a.get("label") == b.get("label") else 0.0
        same_type  = 0.5 if a.get("type")  and a.get("type")  == b.get("type")  else 0.0
        same_session = 0.25 if a.get("sessionId") and a.get("sessionId") == b.get("sessionId") else 0.0
        return float(tag_overlap + same_label + same_type + same_session)

    edges = []
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            w = sim(notes[i], notes[j])
            if w > 0:
                edges.append({"s": notes[i]["id"], "t": notes[j]["id"], "w": w})

    # 3b) Time adjacency: connect consecutive notes taken within 5 minutes
    notes_sorted = sorted(notes, key=lambda n: n.get("timestamp") or 0)
    for i in range(len(notes_sorted) - 1):
        a, b = notes_sorted[i], notes_sorted[i + 1]
        ta = a.get("timestamp") or 0
        tb = b.get("timestamp") or 0
        if abs(tb - ta) <= 5 * 60:
            edges.append({"s": a["id"], "t": b["id"], "w": 0.2})

    graph_doc = {
        "updatedAt": int(time.time()),
        "params": {"k": 0, "simThreshold": 0.0},
        "nodes": nodes,
        "edges": edges,
    }

    db.collection("graphs").document("latest").set(graph_doc)
    return {"nodes": len(nodes), "edges": len(edges)}

# ---------- Backfill AI labels for existing notes by imageURL ----------
@app.post("/backfill_analyze")
def backfill_analyze():
    """
    For notes missing 'ai' field, call Gemini on imageURL and store ai.summary/type/tags/entities.
    Useful for older notes created before AI-labeling was added.
    """
    missing = []
    notes_ref = db.collection("notes")
    docs = list(notes_ref.stream())

    updated = 0
    for d in docs:
        data = d.to_dict() or {}
        if data.get("ai"):
            continue
        url = data.get("imageURL") or data.get("imageUrl")
        if not url:
            continue

        try:
            img_bytes = requests.get(url, timeout=30).content
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            resp = model.generate_content(
                [LABEL_PROMPT, img],
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 30},
            )
            try:
                js = json.loads(resp.text)
            except Exception:
                txt = resp.text or ""
                m1 = re.search(r'"label"\s*:\s*"([^"]+)"', txt)
                m2 = re.search(r'"summary"\s*:\s*"([^"]+)"', txt)
                js = {
                    "label": m1.group(1) if m1 else "Note",
                    "summary": m2.group(1) if m2 else txt.strip()[:120],
                }

            ai_payload = {
                "summary": js.get("summary") or "",
                "type": js.get("type") or "other",
                "tags": js.get("tags") or [],
                "entities": js.get("entities") or [],
            }

            payload = {
                "ai": ai_payload,
                "tags": ai_payload["tags"],              
                "content": ai_payload["summary"],      
                "updatedAt": int(time.time()),          
            }

            db.collection("notes").document(d.id).set(payload, merge=True)
            updated += 1
        except Exception as exc:
            missing.append({"id": d.id, "error": str(exc)})

    return {"updated": updated, "errors": missing}
# ---------- Debugging endpoint ----------
@app.get("/_fb_debug")
def fb_debug():
    import json, os
    from firebase_admin_client import bucket as _bucket
    SA_PATH = os.path.join(os.path.dirname(__file__), "serviceAccount.json")
    return {
        "env_BUCKET": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "resolved_bucket": _bucket.name,
        "project_from_sa": json.load(open(SA_PATH))["project_id"],
    }