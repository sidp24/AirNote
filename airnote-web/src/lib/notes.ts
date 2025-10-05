import { collection, getDocs, doc, updateDoc, orderBy, query } from "firebase/firestore";
import { db } from "../firebase";
import type { Note } from "../types";

export async function fetchNotes(): Promise<Note[]> {
  const qy = query(collection(db, "notes"), orderBy("timestamp", "desc"));
  const snap = await getDocs(qy);
  return snap.docs.map(d => ({ id: d.id, ...(d.data() as any) }));
}

export async function saveLabel(id: string, label: string, summary: string) {
  await updateDoc(doc(db, "notes", id), { label, summary });
}