export type NoteAI = {
  id: string;
  content: string;
  createdAt: number;
  updatedAt?: number;
};

export type NoteDoc = {
  id: string;
  title: string;
  content: string;
  createdAt: number;
  updatedAt?: number;
  tags?: string[];
};

export type GraphNode = {
  id: string;
  label: string;
  data?: any;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label?: string;
};

export type VaultGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};