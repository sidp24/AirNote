import { useEffect, useState } from "react";
import { doc, onSnapshot } from "firebase/firestore";
import { db } from "../firebase";
import type { NoteDoc } from "../types";

export default function NoteDrawer({
  noteId, onClose
}: { noteId: string|null; onClose: () => void }) {
  const [note, setNote] = useState<NoteDoc | null>(null);

  useEffect(() => {
    if (!noteId) return;
    const unsub = onSnapshot(doc(db, "notes", noteId), (snap) => {
      if (snap.exists()) setNote({ id: snap.id, ...(snap.data() as any) });
    });
    return () => unsub?.();
  }, [noteId]);

  return (
    <div className={`fixed top-0 right-0 h-screen w-[420px] bg-[rgba(11,16,32,.9)] border-l border-[var(--border)] backdrop-blur-md transition-transform ${noteId ? "translate-x-0" : "translate-x-full"}`}>
      <div className="p-3.5 flex items-center gap-3 border-b border-[var(--border)]">
        <div className="font-semibold">Note</div>
        <button className="ml-auto btn" onClick={onClose}>Close</button>
      </div>

      {!note ? (
        <div className="p-6 text-[var(--muted)]">Select a node…</div>
      ) : (
        <div className="p-4 space-y-3 overflow-y-auto">
          <div className="text-lg font-semibold">{note.title || note.id}</div>
          <div className="text-sm text-white/80 whitespace-pre-wrap">{(note.content || "").slice(0, 600)}</div>
          <div className="flex flex-wrap gap-2">
            {(note.tags || []).map((t) => (
              <span key={t} className="px-2 py-0.5 rounded-lg border border-[var(--border)] text-xs bg-white/5">#{t}</span>
            ))}
          </div>
          <div className="text-xs text-[var(--muted)]">
            created {new Date((note.createdAt||0) * 1000).toLocaleString()}
            {note.updatedAt ? ` · updated ${new Date(note.updatedAt * 1000).toLocaleString()}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}