export type GraphNode = {
  id: string;
  label?: string;
  clusterId?: string;
  x?: number;
  y?: number;
  degree?: number;
  _color?: string; // computed client-side
};

export type GraphEdge = { s: string; t: string; w: number };

export type VaultGraph = {
  updatedAt: number;
  params: { k: number; simThreshold: number };
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export function toForceGraphData(g: VaultGraph) {
  const links = g.edges.map(e => ({ source: e.s, target: e.t, weight: e.w }));
  const nodes = g.nodes.map(n => ({ ...n }));

  // color by cluster
  const palette = ["#7dd3fc","#a78bfa","#fca5a5","#86efac","#fde68a","#93c5fd","#f9a8d4"];
  const cmap = new Map<string,string>();
  let i = 0;
  for (const n of nodes) {
    if (!n.clusterId) continue;
    if (!cmap.has(n.clusterId)) cmap.set(n.clusterId, palette[i++ % palette.length]);
    n._color = cmap.get(n.clusterId)!;
  }
  return { nodes, links };
}