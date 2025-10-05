import {
  collection,
  doc,
  getDoc,
  getDocs,
  limit,
  onSnapshot,
  orderBy,
  query,
  where,
  setDoc,
  deleteDoc,
} from "firebase/firestore";
import type { DocumentData } from "firebase/firestore";
import { db } from "../firebase";
import { storage } from "../firebase";
import { ref as storageRef, deleteObject } from "firebase/storage";
import type { NoteDoc, VaultGraph } from "../types";

// Normalize Firestore note documents into the NoteDoc shape the UI expects.
function toNoteDoc(id: string, data: any): NoteDoc {
  const normalized: any = {
    id,
    ...data,
  };
  // Legacy image field normalization
  if (data?.imageUrl && !data?.imageURL) normalized.imageURL = data.imageUrl;
  // Normalize created time (fallback to legacy timestamp)
  if (normalized.createdAt === undefined && data?.timestamp !== undefined) {
    normalized.createdAt = data.timestamp;
  }
  // Surface AI summary as content fallback so Dashboard/Note always show it
  normalized.content = normalized.content ?? data?.ai?.summary ?? "";
  // Tags fallback from AI
  normalized.tags = normalized.tags ?? data?.ai?.tags ?? [];
  return normalized as NoteDoc;
}

export async function fetchNotes(limitCount = 100): Promise<NoteDoc[]> {
  const q = query(collection(db, "notes"), limit(limitCount));
  const snap = await getDocs(q);
  const items = snap.docs.map(d => toNoteDoc(d.id, d.data()));
  items.sort((a: any, b: any) => ((b.createdAt ?? b.timestamp ?? 0) - (a.createdAt ?? a.timestamp ?? 0)));
  return items;
}

export async function fetchNotesByCluster(clusterId: string): Promise<NoteDoc[]> {
  const q = query(
    collection(db, "notes"),
    where("clusterId", "==", clusterId),
    limit(100)
  );
  const snap = await getDocs(q);
  const items = snap.docs.map(d => toNoteDoc(d.id, d.data()));
  items.sort((a: any, b: any) => ((b.createdAt ?? b.timestamp ?? 0) - (a.createdAt ?? a.timestamp ?? 0)));
  return items;
}

export function subscribeNotes(
  cb: (notes: NoteDoc[]) => void,
  limitCount = 200
) {
  const q = query(
    collection(db, "notes"),
    orderBy("timestamp", "desc"),
    limit(limitCount)
  );
  return onSnapshot(q, (snap) => {
    const items: NoteDoc[] = snap.docs.map(d => toNoteDoc(d.id, d.data()));
    items.sort((a: any, b: any) => ((b.createdAt ?? b.timestamp ?? 0) - (a.createdAt ?? a.timestamp ?? 0)));
    cb(items);
  });
}

export async function fetchGraph(): Promise<VaultGraph | null> {
  const ref = doc(db, "graphs", "latest");
  const s = await getDoc(ref);
  return s.exists() ? (s.data() as VaultGraph) : null;
}

/**
 * Delete a note and attempt to delete its image from Firebase Storage.
 * Works with gs://, https download URLs, or storage paths.
 */
export async function deleteNote(id: string, imageUrl?: string) {
  if (imageUrl) {
    try {
      const r = storageRef(storage, imageUrl);
      await deleteObject(r);
    } catch (err) {
      console.warn("Skipping image delete:", err);
    }
  }
  await deleteDoc(doc(db, "notes", id));
}

export async function saveLabel(id: string, label: string): Promise<void> {
  const ref = doc(db, "notes", id);
  await setDoc(ref, { label }, { merge: true });
}

export async function saveLabelAndSummary(id: string, label: string, summary?: string): Promise<void> {
  const ref = doc(db, "notes", id);
  const payload: Record<string, any> = { label };

  if (summary && summary.trim()) {
    payload.ai = { summary: summary.trim() }; 
  }

  await setDoc(ref, payload, { merge: true });

  const docSnap = await getDoc(ref);
  if (docSnap.exists()) {
    const updatedData = docSnap.data() as NoteDoc;
    console.log("Updated note with summary:", updatedData);
  }
}