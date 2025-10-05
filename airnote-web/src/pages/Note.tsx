import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { db } from "../firebase";
import { doc, onSnapshot } from "firebase/firestore";
import { motion } from "framer-motion";

type NoteView = {
  id: string;
  title?: string;
  content?: string;
  imageUrl?: string;
  imageURL?: string; // legacy
  label?: string;
  sessionId?: string;
  createdAt?: number;
  updatedAt?: number;
  timestamp?: number; // legacy
  tags?: string[];
  ai?: { summary?: string; type?: string; tags?: string[] };
};

function displayLabel(n: NoteView): string {
  const l = n.label?.trim();
  if (l) return l;
  if (n.ai?.type) return n.ai.type;
  if (n.tags && n.tags.length) return n.tags[0]!;
  return "unlabeled";
}

export default function NotePage() {
  const { id } = useParams();
  const [note, setNote] = useState<NoteView | null>(null);

  useEffect(() => {
    if (!id) return;
    const ref = doc(db, "notes", id);
    const unsub = onSnapshot(ref, (snap) => {
      if (!snap.exists()) return setNote(null);
      const d = snap.data() as any;
      setNote({
        id: snap.id,
        title: d.title ?? snap.id,
        content: d.content ?? d.ai?.summary ?? "",
        imageUrl: d.imageUrl ?? d.imageURL,
        imageURL: d.imageURL, // keep legacy field just in case other UI references it
        label: d.label,
        sessionId: d.sessionId,
        createdAt: d.createdAt ?? d.timestamp,
        updatedAt: d.updatedAt,
        timestamp: d.timestamp,
        tags: d.tags ?? d.ai?.tags ?? [],
        ai: d.ai,
      });
    });
    return () => unsub();
  }, [id]);

  if (!note) return <div className="text-[var(--muted)]">Loading…</div>;

  const img = note.imageUrl || note.imageURL;
  const when = note.updatedAt ?? note.createdAt ?? note.timestamp;
  const lbl = displayLabel(note);
  const summary = note.ai?.summary || note.content || "";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-7xl"
    >
      <div className="mb-4 flex items-center gap-2">
        <Link to="/" className="btn">← Back</Link>
        <div className="ml-auto text-xs text-[var(--muted)]">
          {when ? new Date(when * 1000).toLocaleString() : ""}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass overflow-hidden rounded-xl border border-[var(--border)]">
          {img ? (
            <img src={img} className="w-full h-auto object-contain" />
          ) : (
            <div className="aspect-video grid place-items-center text-[var(--muted)]">
              No preview available
            </div>
          )}
        </div>

        <div className="glass p-4 space-y-3 rounded-xl border border-[var(--border)]">
          <div className="flex items-center gap-2">
            <div className="font-semibold truncate">{note.title || note.id}</div>
            <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs bg-[var(--accent)]/18 text-[#cfe0ff] border border-[var(--accent)]/30">
              {lbl}
            </span>
          </div>

          {/* AI Summary */}
          {summary ? (
            <div>
              <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">AI Summary</div>
              <p className="leading-relaxed text-white/90 whitespace-pre-wrap">
                {summary}
              </p>
            </div>
          ) : (
            <div className="text-[var(--muted)] text-sm">No summary</div>
          )}

          {/* Extra meta */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><span className="text-[var(--muted)]">Session:</span> {note.sessionId || "—"}</div>
            <div><span className="text-[var(--muted)]">Tags:</span> {(note.tags || []).join(", ") || "—"}</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}