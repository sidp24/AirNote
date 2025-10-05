import React, { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d"; 
import { doc, onSnapshot } from "firebase/firestore";
import { db } from "../firebase";
import NoteDrawer from "../components/NoteDrawer";
import { toForceGraphData, type VaultGraph } from "../lib/graph";

export default function Graph() {
  const [graph, setGraph] = useState<VaultGraph | null>(null as VaultGraph | null);
  const [query, setQuery] = useState("");
  const fgRef = useRef<any>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hoverNode, setHoverNode] = useState<any | null>(null);
  const [mouse, setMouse] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const clusters = useMemo(() => {
    if (!graph) return ["all"];
    const set = new Set((graph.nodes || []).map((n: any) => n.clusterId).filter(Boolean));
    return ["all", ...Array.from(set) as string[]];
  }, [graph]);
  const [clusterFilter, setClusterFilter] = useState<string>("all");

  useEffect(() => {
    const unsub = onSnapshot(doc(db, "graphs", "latest"), (snap) => {
      if (snap.exists()) setGraph(snap.data() as any);
    });
    return () => unsub();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "f" || e.key === "F") fgRef.current?.zoomToFit(400, 60);
      if (e.key === "Escape") {
        setActiveId(null);
        setHoverNode(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const data = useMemo(() => {
    if (!graph) return { nodes: [], links: [] };
    return toForceGraphData(graph);
  }, [graph]);

  const filtered = useMemo(() => {
    // 1) Search filter
    let base = data;
    if (query.trim()) {
      const q = query.toLowerCase();
      const keep = new Set(
        data.nodes
          .filter((n: any) => (n.label || "").toLowerCase().includes(q) || (n.id || "").toLowerCase().includes(q))
          .map((n: any) => n.id)
      );
      const nodes = data.nodes.filter((n: any) => keep.has(n.id));
      const links = data.links.filter((l: any) => {
        const s = (l.source as any)?.id ?? l.source;
        const t = (l.target as any)?.id ?? l.target;
        return keep.has(s) && keep.has(t);
      });
      base = { nodes, links };
    }

    // 2) Cluster filter
    if (clusterFilter !== "all") {
      const keep = new Set(base.nodes.filter((n: any) => n.clusterId === clusterFilter).map((n: any) => n.id));
      const nodes = base.nodes.filter((n: any) => keep.has(n.id));
      const links = base.links.filter((l: any) => {
        const s = (l.source as any)?.id ?? l.source;
        const t = (l.target as any)?.id ?? l.target;
        return keep.has(s) && keep.has(t);
      });
      return { nodes, links };
    }

    return base;
  }, [data, query, clusterFilter]);

  const neighborIds = useMemo(() => {
    if (!filtered.links || !activeId) return new Set<string>();
    const s = new Set<string>([activeId]);
    for (const l of filtered.links as any[]) {
      const sid = (l.source as any)?.id ?? l.source;
      const tid = (l.target as any)?.id ?? l.target;
      if (sid === activeId) s.add(tid as string);
      if (tid === activeId) s.add(sid as string);
    }
    return s;
  }, [filtered, activeId]);

  async function rebuild() {
    try {
      await fetch("http://127.0.0.1:5050/rebuild_graph", { method: "POST" });
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div className="min-h-[calc(100vh-120px)]">
      <div className="flex items-center gap-3 mb-3">
        <div className="font-medium">Vault Graph</div>

        <div className="ml-auto flex items-center gap-2">
          <input
            className="input max-w-md"
            placeholder="Search nodes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select
            className="input w-[180px]"
            value={clusterFilter}
            onChange={(e) => setClusterFilter(e.target.value)}
          >
            {clusters.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <button className="btn" onClick={() => fgRef.current?.zoomToFit(400, 60)}>Fit</button>
          <button className="btn" onClick={rebuild}>Rebuild</button>
        </div>
      </div>

      <div
        style={{ height: "calc(100vh - 160px)" }}
        className="relative"
        onMouseMove={(e) => {
          const anyEvt = e as any;
          setMouse({ x: anyEvt.nativeEvent.offsetX, y: anyEvt.nativeEvent.offsetY });
        }}
      >
        <div className="flex items-center gap-4 text-xs text-[var(--muted)] mb-2">
          <div>nodes: {data.nodes.length}</div>
          <div>edges: {data.links.length}</div>
          <div>updated: {graph ? new Date(graph.updatedAt * 1000).toLocaleTimeString() : "—"}</div>
        </div>
        <ForceGraph2D
          ref={fgRef as any}
          graphData={filtered}
          nodeRelSize={6}
          linkDirectionalParticles={1}
          linkDirectionalParticleSpeed={(d: any) => 0.003 + (d.weight || 0) * 0.01}
          linkColor={(d: any) => {
            const sid = (d.source as any)?.id ?? d.source;
            const tid = (d.target as any)?.id ?? d.target;
            const focus = !activeId || (neighborIds.has(sid) && neighborIds.has(tid));
            return focus ? "rgba(148,163,184,.55)" : "rgba(148,163,184,.15)";
          }}
          linkWidth={(d: any) => {
            const sid = (d.source as any)?.id ?? d.source;
            const tid = (d.target as any)?.id ?? d.target;
            const focus = !activeId || (neighborIds.has(sid) && neighborIds.has(tid));
            const base = Math.max(0.5, (d.weight || 0) * 2);
            return focus ? base : 0.5;
          }}
          nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, scale: number) => {
            const isFocus = !activeId || neighborIds.has(node.id);
            const size = 6 + (node.degree || 0) * 0.2;
            ctx.save();
            ctx.globalAlpha = isFocus ? 1 : 0.18;
            ctx.fillStyle = node._color || "#60a5fa";
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
            ctx.fill();
            const label = node.label || node.id;
            ctx.font = `${12 / scale}px ui-sans-serif`;
            ctx.fillStyle = isFocus ? "rgba(226,232,240,.95)" : "rgba(226,232,240,.3)";
            ctx.fillText(label, node.x + size + 2, node.y + 3);
            ctx.restore();
          }}
          onNodeClick={(n: any) => {
            fgRef.current?.centerAt(n.x, n.y, 700);
            fgRef.current?.zoom(3, 700);
            setActiveId(n.id);
          }}
          cooldownTicks={200}
          onEngineStop={() => {
            try {
              const g = (fgRef.current as any)?.graphData?.();
              if (g && g.nodes && g.nodes.length) {
                fgRef.current?.zoomToFit(400, 60);
              }
            } catch {}
          }}
          onNodeHover={(n: any) => setHoverNode(n || null)}
          onBackgroundClick={() => { setHoverNode(null); setActiveId(null); }}
          onZoomEnd={() => setHoverNode(null)}
        />
        {hoverNode && (
          <div
            className="absolute pointer-events-none text-xs bg-black/70 text-white px-2 py-1 rounded border border-white/10"
            style={{ left: mouse.x + 12, top: mouse.y + 12 }}
          >
            <div className="font-medium">{hoverNode.label || hoverNode.id}</div>
            {hoverNode.clusterId && <div className="text-[var(--muted)]">cluster: {hoverNode.clusterId}</div>}
            {typeof hoverNode.degree === "number" && <div className="text-[var(--muted)]">degree: {hoverNode.degree}</div>}
          </div>
        )}
        {!graph && (
          <div className="grid place-items-center text-[var(--muted)] h-full">
            Building graph…
          </div>
        )}
      </div>
      <NoteDrawer noteId={activeId} onClose={() => setActiveId(null)} />
    </div>
  );
}
