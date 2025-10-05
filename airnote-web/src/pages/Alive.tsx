import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
const BE = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5050";

type NoteItem = {
  id: string;
  title?: string;
  imageURL?: string;
  label?: string;
  summary?: string;
};

import { subscribeNotes, saveLabel } from "../lib/notes";

async function labelByUrl(imageURL: string) {
  const r = await fetch(`${BE}/label_by_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageURL }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ label: string; summary: string }>;
}

export default function Alive() {
  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    const unsub = subscribeNotes((ns) => setNotes(ns as unknown as NoteItem[]));
    return () => unsub();
  }, []);

  const groups = useMemo(() => {
    const m = new Map<string, NoteItem[]>();
    for (const n of notes) m.set(n.label || "Unlabeled", [...(m.get(n.label || "Unlabeled") || []), n]);
    return Array.from(m.entries());
  }, [notes]);

  async function doLabel(n: NoteItem) {
    try {
      setBusy(n.id);
      const { label, summary } = await labelByUrl(n.imageURL || "");
      await saveLabel(n.id, label);
      setNotes(prev => prev.map(p => p.id === n.id ? { ...p, label, summary } : p));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="mx-auto max-w-7xl space-y-7">
      <div className="flex items-end justify-between">
        <h1 className="text-[22px] font-semibold">Alive Notes — Auto Cluster</h1>
      </div>

      {groups.map(([label, list], i) => (
        <motion.section key={label} initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:i*0.05}} className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">{label}</h2>
            <span className="text-[var(--muted)] text-sm">{list.length} notes</span>
          </div>
          <div className="flex flex-wrap gap-4">
            {list.map(n => (
              <div key={n.id} className="glass p-2 pr-3 flex items-center gap-3">
                <img src={n.imageURL} className="w-16 h-11 object-cover rounded-lg border border-[var(--border)]" />
                <div className="w-56">
                  <div className="text-sm font-semibold truncate">{n.title || n.id}</div>
                  <div className="text-xs text-[var(--muted)] truncate">{n.summary || "—"}</div>
                </div>
                {!n.label && (
                  <button className="btn-accent ml-auto disabled:opacity-60" disabled={!!busy} onClick={() => doLabel(n)}>
                    {busy === n.id ? "Labeling…" : "Label"}
                  </button>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      ))}
    </div>
  );
}