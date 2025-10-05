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


# --- Gemini setup ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var is not set")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Text embedding model (for graph similarity)
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "models/text-embedding-004")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for hackathon, tighten later
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---- Embedding helpers (32-dim vector from text) ----
def embed_text_32(text: str) -> list[float]:
    """
    Create a compact 32-dim vector from a text embedding (Gemini text-embedding-004).
    We truncate to 32 dims to keep Firestore payloads small.
    """
    if not text or not text.strip():
        return []
    try:
        resp = genai.embed_content(model=TEXT_EMBED_MODEL, content=text)
        vec = resp.get("embedding") or resp.get("data", {}).get("embedding")
        if not vec:
            return []
        return [float(x) for x in vec[:32]]
    except Exception as e:
        print("[embed_text_32] error:", e)
        return []

def ensure_text_embedding_for_note(note: dict) -> list[float]:
    """
    Returns a 32-dim vector for a note if possible.
    Prefers existing embedding.pca32; otherwise embeds title+summary text.
    """
    emb = (note.get("embedding") or {})
    if isinstance(emb.get("pca32"), list) and emb["pca32"]:
        return [float(x) for x in emb["pca32"]]

    ai = note.get("ai") or {}
    title = (note.get("title") or "").strip()
    summary = (ai.get("summary") or "").strip()
    text = (f"{title}\n{summary}".strip() or title or summary)
    return embed_text_32(text)

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))

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

@app.post("/rebuild_graph")
def rebuild_graph(
    sim: float = 0.65,
    k: int = 4,
    mutual: bool = True,
    add_time_edges: bool = False,
    time_window_sec: int = 120,
):
    """
    Build a note graph and store it at graphs/latest.

    Query params:
      - sim: similarity threshold (0..1) for edge creation (default 0.65)
      - k:   max neighbors per node based on similarity ranking (default 4)
      - mutual: if True, keep an edge only when A is in B's top-k AND B is in A's top-k
      - add_time_edges: if True, also connect temporally adjacent notes within time_window_sec
      - time_window_sec: window (seconds) for time adjacency (default 120)
    """
    notes_ref = db.collection("notes")
    docs = list(notes_ref.stream())

    notes: List[Dict] = []
    for d in docs:
        data = d.to_dict() or {}
        ai = data.get("ai") or {}
        tags = ai.get("tags") or data.get("tags") or []
        notes.append({
            "id": d.id,
            "title": data.get("title") or f"Note {d.id}",
            "label": data.get("label") or ai.get("label") or None,
            "type": ai.get("type") or None,
            "tags": tags,
            "timestamp": data.get("timestamp") or 0,
            "sessionId": data.get("sessionId") or None,
            "embedding": data.get("embedding") or {},
            "ai": ai,
        })

    # Build nodes (cluster by label/type)
    nodes = []
    for n in notes:
        cluster = n.get("label") or n.get("type") or "unlabeled"
        nodes.append({
            "id": n["id"],
            "label": n.get("title") or n["id"],
            "clusterId": cluster,
        })

    # Prepare vectors (prefer persisted embeddings; otherwise synthesize)
    vecs: dict[str, list[float]] = {}
    for n in notes:
        emb = (n.get("embedding") or {})
        v = emb.get("pca32")
        if isinstance(v, list) and v:
            vecs[n["id"]] = [float(x) for x in v]
        else:
            vecs[n["id"]] = ensure_text_embedding_for_note(n)

    # Heuristic fallback when vectors are missing/weak
    def heuristic(a: Dict, b: Dict) -> float:
        at = set(a.get("tags") or [])
        bt = set(b.get("tags") or [])
        tag_overlap = len(at & bt)
        same_label = 1.0 if a.get("label") and a.get("label") == b.get("label") else 0.0
        same_type  = 0.5 if a.get("type")  and a.get("type")  == b.get("type")  else 0.0
        same_session = 0.25 if a.get("sessionId") and a.get("sessionId") == b.get("sessionId") else 0.0
        return float(tag_overlap + same_label + same_type + same_session)

    # Compute top-k neighbor lists
    neighbors: dict[str, list[tuple[str, float]]] = {n["id"]: [] for n in notes}
    N = len(notes)
    for i in range(N):
        a = notes[i]
        va = vecs.get(a["id"]) or []
        row = []
        for j in range(N):
            if i == j:
                continue
            b = notes[j]
            vb = vecs.get(b["id"]) or []
            s = cosine(va, vb) if (va and vb) else heuristic(a, b)
            if s >= sim:
                row.append((b["id"], s))
        row.sort(key=lambda t: t[1], reverse=True)
        neighbors[a["id"]] = row[:max(0, int(k))]

    # Build edges (optionally mutual)
    edges_set = set()
    for a_id, nbrs in neighbors.items():
        for b_id, score in nbrs:
            if mutual:
                if any(x_id == a_id for x_id, _ in neighbors.get(b_id, [])):
                    key = tuple(sorted((a_id, b_id)))
                    edges_set.add((key[0], key[1], float(round(score, 4))))
            else:
                key = tuple(sorted((a_id, b_id)))
                edges_set.add((key[0], key[1], float(round(score, 4))))

    edges = [{"s": s, "t": t, "w": w} for (s, t, w) in edges_set]

    # Optional: time adjacency edges
    if add_time_edges and time_window_sec > 0:
        notes_sorted = sorted(notes, key=lambda n: n.get("timestamp") or 0)
        for i in range(len(notes_sorted) - 1):
            a, b = notes_sorted[i], notes_sorted[i + 1]
            ta = a.get("timestamp") or 0
            tb = b.get("timestamp") or 0
            if abs(tb - ta) <= int(time_window_sec):
                key = tuple(sorted((a["id"], b["id"])))
                if not any(e["s"] == key[0] and e["t"] == key[1] for e in edges):
                    edges.append({"s": key[0], "t": key[1], "w": 0.2})

    graph_doc = {
        "updatedAt": int(time.time()),
        "params": {
            "k": int(k),
            "simThreshold": float(sim),
            "mutual": bool(mutual),
            "add_time_edges": bool(add_time_edges),
            "time_window_sec": int(time_window_sec),
        },
        "nodes": nodes,
        "edges": edges,
    }
    db.collection("graphs").document("latest").set(graph_doc)
    return {"nodes": len(nodes), "edges": len(edges), "params": graph_doc["params"]}

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

@app.post("/backfill_summaries_embeddings")
def backfill_summaries_embeddings():
    """
    Ensure notes have AI summaries (if missing) and a text embedding at embedding.pca32.
    """
    notes_ref = db.collection("notes")
    docs = list(notes_ref.stream())

    updated = 0
    errors = []
    batch = db.batch()

    for d in docs:
        try:
            n = d.to_dict() or {}
            n["id"] = d.id

            ai = n.get("ai") or {}
            needs_ai = not bool(ai.get("summary"))

            # If summary missing, try to label from image URL
            if needs_ai:
                url = n.get("imageURL") or n.get("imageUrl")
                if url:
                    try:
                        img_bytes = requests.get(url, timeout=20).content
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
                            js = {"label": m1.group(1) if m1 else "Note",
                                  "summary": m2.group(1) if m2 else txt.strip()[:120]}
                        ai = {
                            "summary": js.get("summary") or "",
                            "type": js.get("type") or "other",
                            "tags": js.get("tags") or [],
                            "entities": js.get("entities") or [],
                        }
                    except Exception as le:
                        errors.append({"id": d.id, "error": f"label_fail: {le}"})
                        ai = ai or {}

            # Ensure embedding.pca32 from text (title + summary)
            vec32 = ensure_text_embedding_for_note({"title": n.get("title"), "ai": ai, "embedding": n.get("embedding")})

            update_payload = {}
            if ai:
                update_payload["ai"] = ai
                # also mirror tags/summary for convenience
                update_payload["tags"] = ai.get("tags") or []
                update_payload["content"] = ai.get("summary") or ""

            if vec32:
                update_payload["embedding"] = {"pca32": vec32}

            if update_payload:
                batch.update(notes_ref.document(d.id), update_payload)
                updated += 1

            # Commit in chunks to avoid exceeding limits
            if updated % 400 == 0:
                batch.commit()
                batch = db.batch()

        except Exception as e:
            errors.append({"id": d.id, "error": str(e)})

    if updated % 400 != 0:
        batch.commit()

    return {"updated": updated, "errors": errors}
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