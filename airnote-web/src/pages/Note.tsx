import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { db } from "../firebase";
import { doc, getDoc } from "firebase/firestore";
import type { Note } from "../types";
import { motion } from "framer-motion";

export default function NotePage() {
  const { id } = useParams();
  const [note, setNote] = useState<Note | null>(null);

  useEffect(() => {
    (async () => {
      const s = await getDoc(doc(db, "notes", id!));
      if (s.exists()) setNote({ id: s.id, ...(s.data() as any) });
    })();
  }, [id]);

  if (!note) return <div className="text-[var(--muted)]">Loading…</div>;

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mx-auto max-w-7xl">
      <div className="mb-4">
        <Link to="/" className="btn">← Back</Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass overflow-hidden">
          <img src={note.imageURL} className="w-full h-auto object-contain" />
        </div>

        <div className="glass p-4 space-y-2">
          <div><b>Label:</b> <span className="text-white/90">{note.label || "—"}</span></div>
          <div><b>Summary:</b> <div className="text-white/80">{note.summary || "—"}</div></div>
          <div><b>Session:</b> {note.sessionId}</div>
          <div><b>When:</b> {new Date(note.timestamp * 1000).toLocaleString()}</div>
        </div>
      </div>
    </motion.div>
  );
}