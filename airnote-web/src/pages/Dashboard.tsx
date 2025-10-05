import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { subscribeNotes } from "../lib/notes";
import type { NoteDoc } from "../types";
import { Search } from "lucide-react";

export default function Dashboard() {
  const [notes, setNotes] = useState<NoteDoc[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    const unsub = subscribeNotes((ns) => setNotes(ns as NoteDoc[]));
    return () => unsub();
  }, []);

  const filtered = useMemo(() => {
    const t = q.toLowerCase();
    return notes.filter(n =>
      (n.title || "").toLowerCase().includes(t) ||
      (n.tags?.join(", ") || "").toLowerCase().includes(t) ||
      (n.content || "").toLowerCase().includes(t)
    );
  }, [notes, q]);

  const displayLabel = (n: NoteDoc) => {
    const label = (n as any).label as string | undefined;
    const aiType = (n as any).ai?.type as string | undefined;
    if (label && label.trim()) return label;
    if (aiType && aiType.trim()) return aiType;
    if (n.tags && n.tags.length > 0) return n.tags[0]!;
    return "unlabeled";
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex items-end justify-between">
        <h1 className="text-[22px] font-semibold">AirNote — Notes</h1>
      </div>

      <div className="relative max-w-xl">
        <Search className="absolute left-3 top-3.5 size-4 text-[var(--muted)]" />
        <input
          className="input pl-9"
          placeholder="Search title, tags, content…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((n, idx) => (
          <motion.div
            key={n.id}
            initial={{ y: 18, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: idx * 0.04, type: "spring", stiffness: 120, damping: 18 }}
            className="animate-[fadeInUp_.55s_ease_forwards] opacity-0 translate-y-3"
          >
            <Link to={`/note/${n.id}`} className="card block group hover:shadow-lg transition-shadow">
              <div className="aspect-[16/9] overflow-hidden rounded-t-xl bg-black/20">
                {(() => {
                  const img: string | undefined =
                    (n as any).imageURL || (n as any).imageUrl || undefined;
                  return img ? (
                    <img
                      src={img}
                      alt={n.title || n.id}
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                      loading="lazy"
                    />
                  ) : (
                    <div className="grid place-items-center w-full h-full text-[var(--muted)] text-sm">
                      No preview
                    </div>
                  );
                })()}
              </div>
              <div className="p-3.5 space-y-1.5">
                <div className="flex items-center gap-2">
                  <div className="font-semibold truncate">{n.title || n.id}</div>
                  {(() => {
                    const lbl = displayLabel(n);
                    return lbl !== "unlabeled" ? (
                      <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs bg-[var(--accent)]/18 text-[#cfe0ff] border border-[var(--accent)]/30">
                        {lbl}
                      </span>
                    ) : (
                      <span className="text-[var(--muted)] text-xs">unlabeled</span>
                    );
                  })()}
                </div>
                {n.content && <div className="text-sm text-white/80 line-clamp-2">{n.content}</div>}
                <div className="text-xs text-[var(--muted)]">
                  Created: {n.createdAt ? new Date(n.createdAt * 1000).toLocaleString() : "N/A"}<br />
                  Updated: {n.updatedAt ? new Date(n.updatedAt * 1000).toLocaleString() : "—"}
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-[var(--muted)]">No notes yet. Ingest from your capture app and refresh.</div>
      )}
    </div>
  );
}