import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { fetchNotes } from "../lib/notes";
import type { Note } from "../types";
import { Search } from "lucide-react";

export default function Dashboard() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => { fetchNotes().then(setNotes); }, []);

  const filtered = useMemo(() => {
    const t = q.toLowerCase();
    return notes.filter(n =>
      (n.title || "").toLowerCase().includes(t) ||
      (n.label || "").toLowerCase().includes(t) ||
      (n.summary || "").toLowerCase().includes(t)
    );
  }, [notes, q]);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex items-end justify-between">
        <h1 className="text-[22px] font-semibold">AirNote — Notes</h1>
        <Link to="/alive" className="btn-accent">Alive</Link>
      </div>

      <div className="relative max-w-xl">
        <Search className="absolute left-3 top-3.5 size-4 text-[var(--muted)]" />
        <input
          className="input pl-9"
          placeholder="Search title, label, summary…"
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
          >
            <Link to={`/note/${n.id}`} className="card block">
              <div className="aspect-video overflow-hidden">
                <img src={n.imageURL} className="w-full h-full object-cover" loading="lazy" />
              </div>
              <div className="p-3.5 space-y-1.5">
                <div className="flex items-center gap-2">
                  <div className="font-semibold truncate">{n.title || n.id}</div>
                  {n.label ? (
                    <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs bg-[var(--accent)]/18 text-[#cfe0ff] border border-[var(--accent)]/30">
                      {n.label}
                    </span>
                  ) : (
                    <span className="text-[var(--muted)] text-xs">unlabeled</span>
                  )}
                </div>
                {n.summary && <div className="text-sm text-white/80 line-clamp-2">{n.summary}</div>}
                <div className="text-xs text-[var(--muted)]">
                  {new Date(n.timestamp * 1000).toLocaleString()}
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