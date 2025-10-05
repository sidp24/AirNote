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
} from "firebase/firestore";
import type { DocumentData } from "firebase/firestore";
import { db } from "../firebase";
import type { NoteDoc, VaultGraph } from "../types";

export async function fetchNotes(limitCount = 100): Promise<NoteDoc[]> {
  // Fetch latest N (no Firestore ordering assumption), then sort locally
  const q = query(collection(db, "notes"), limit(limitCount));
  const snap = await getDocs(q);
  const items = snap.docs.map(d => {
    const data = d.data() as any;
    const normalized: any = { id: d.id, ...data };
    if (data.imageUrl && !data.imageURL) normalized.imageURL = data.imageUrl;
    if (data.createdAt === undefined && data.timestamp !== undefined) {
      normalized.createdAt = data.timestamp;
    }
    return normalized as NoteDoc;
  });
  // Sort by createdAt (new) falling back to legacy timestamp
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
  const items = snap.docs.map(d => {
    const data = d.data() as any;
    const normalized: any = { id: d.id, ...data };
    if (data.imageUrl && !data.imageURL) normalized.imageURL = data.imageUrl;
    if (data.createdAt === undefined && data.timestamp !== undefined) {
      normalized.createdAt = data.timestamp;
    }
    return normalized as NoteDoc;
  });
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
    const items: NoteDoc[] = [];
    snap.forEach((d) => {
      const data = d.data() as DocumentData;
      // Normalize a couple of legacy fields so the UI doesn't care about casing
      const normalized: any = {
        id: d.id,
        ...data,
      };
      if (data.imageUrl && !data.imageURL) normalized.imageURL = data.imageUrl;
      if (data.createdAt === undefined && data.timestamp !== undefined) {
        normalized.createdAt = data.timestamp;
      }
      items.push(normalized as NoteDoc);
    });
    // Keep newest first if Firestore ordering ever changes
    items.sort((a: any, b: any) => ((b.createdAt ?? b.timestamp ?? 0) - (a.createdAt ?? a.timestamp ?? 0)));
    cb(items);
  });
}

export async function fetchGraph(): Promise<VaultGraph | null> {
  const ref = doc(db, "graphs", "latest");
  const s = await getDoc(ref);
  return s.exists() ? (s.data() as VaultGraph) : null;
}

export async function saveLabel(id: string, label: string): Promise<void> {
  const ref = doc(db, "notes", id);
  await setDoc(ref, { label }, { merge: true });
}