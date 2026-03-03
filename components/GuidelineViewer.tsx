"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { NICEGuideline, NICEGraphNode, NICEGraphEdge } from "@/lib/types";
import { Button } from "./ui/button";
import { X, FileJson, GitBranch, Copy, Check, Settings, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface GuidelineViewerProps {
  guideline: NICEGuideline;
  onClose: () => void;
}

// ─── Decision Tree Layout ───────────────────────────────────────────

interface LayoutNode {
  id: string;
  type: "condition" | "action";
  text: string;
  x: number;
  y: number;
}

interface LayoutEdge {
  from: string;
  to: string;
  label: string;
}

function computeTreeLayout(nodes: NICEGraphNode[], edges: NICEGraphEdge[]) {
  const children: Record<string, { to: string; label: string }[]> = {};
  const inDegree: Record<string, number> = {};
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  for (const n of nodes) {
    children[n.id] = [];
    inDegree[n.id] = 0;
  }
  for (const e of edges) {
    children[e.from]?.push({ to: e.to, label: e.label || "next" });
    inDegree[e.to] = (inDegree[e.to] || 0) + 1;
  }

  // Find root (in-degree 0)
  let root = nodes[0]?.id || "";
  for (const n of nodes) {
    if ((inDegree[n.id] || 0) === 0) { root = n.id; break; }
  }

  // Assign layers using longest-path from root (ensures proper depth)
  const nodeLayer: Record<string, number> = {};
  const visited = new Set<string>();

  function assignLayer(nid: string, layer: number) {
    // Always take the MAX layer (deepest path wins)
    if (nodeLayer[nid] !== undefined && nodeLayer[nid] >= layer) return;
    nodeLayer[nid] = layer;
    if (visited.has(nid)) return; // Prevent infinite loops on back-edges
    visited.add(nid);
    for (const child of children[nid] || []) {
      assignLayer(child.to, layer + 1);
    }
    visited.delete(nid); // Allow revisiting through different paths
  }
  assignLayer(root, 0);

  // Handle disconnected nodes — assign them to a layer based on their edges
  for (const n of nodes) {
    if (nodeLayer[n.id] === undefined) {
      // Try to place after any parent that points to it
      let maxParentLayer = -1;
      for (const e of edges) {
        if (e.to === n.id && nodeLayer[e.from] !== undefined) {
          maxParentLayer = Math.max(maxParentLayer, nodeLayer[e.from]);
        }
      }
      nodeLayer[n.id] = maxParentLayer >= 0 ? maxParentLayer + 1 : 0;
    }
  }

  // Group nodes by layer
  const layerMap: Record<number, string[]> = {};
  for (const [nid, layer] of Object.entries(nodeLayer)) {
    if (!layerMap[layer]) layerMap[layer] = [];
    layerMap[layer].push(nid);
  }
  const maxLayer = Math.max(...Object.keys(layerMap).map(Number), 0);
  const layers: string[][] = [];
  for (let i = 0; i <= maxLayer; i++) {
    layers.push(layerMap[i] || []);
  }

  const NODE_WIDTH = 210;
  const NODE_HEIGHT = 64;
  const H_GAP = 36;
  const V_GAP = 80;

  let maxWidth = 0;
  for (const layer of layers) {
    const w = layer.length * NODE_WIDTH + (layer.length - 1) * H_GAP;
    if (w > maxWidth) maxWidth = w;
  }

  const layoutNodes: LayoutNode[] = [];
  for (let li = 0; li < layers.length; li++) {
    const layer = layers[li];
    const totalW = layer.length * NODE_WIDTH + (layer.length - 1) * H_GAP;
    const startX = (maxWidth - totalW) / 2;
    for (let ci = 0; ci < layer.length; ci++) {
      const node = nodeMap[layer[ci]];
      if (!node) continue;
      layoutNodes.push({
        id: node.id,
        type: node.type,
        text: node.text,
        x: startX + ci * (NODE_WIDTH + H_GAP),
        y: li * (NODE_HEIGHT + V_GAP),
      });
    }
  }

  // Include ALL edges (including cross-layer)
  const allNodeIds = new Set(nodes.map(n => n.id));
  const layoutEdges: LayoutEdge[] = edges
    .filter(e => allNodeIds.has(e.from) && allNodeIds.has(e.to))
    .map(e => ({ from: e.from, to: e.to, label: e.label || "next" }));

  return { nodes: layoutNodes, edges: layoutEdges, width: maxWidth + NODE_WIDTH, height: layers.length * (NODE_HEIGHT + V_GAP), nodeWidth: NODE_WIDTH, nodeHeight: NODE_HEIGHT };
}

// ─── Visual Decision Tree ────────────────────────────────────────────

function DecisionTreeView({ guideline }: { guideline: NICEGuideline }) {
  const layout = useMemo(
    () => computeTreeLayout(guideline.nodes, guideline.edges),
    [guideline.nodes, guideline.edges]
  );

  const nodePos = useMemo(() => {
    const map: Record<string, LayoutNode> = {};
    for (const n of layout.nodes) map[n.id] = n;
    return map;
  }, [layout.nodes]);

  const PAD = 24;
  const svgW = layout.width + PAD * 2;
  const svgH = layout.height + PAD * 2;
  const halfW = layout.nodeWidth / 2;

  // Zoom & pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(z => Math.min(3, Math.max(0.2, z + delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging) return;
    setPan({
      x: dragStart.current.panX + (e.clientX - dragStart.current.x),
      y: dragStart.current.panY + (e.clientY - dragStart.current.y),
    });
  }, [dragging]);

  const handleMouseUp = useCallback(() => setDragging(false), []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Fit to view
  const fitToView = useCallback(() => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const scaleX = rect.width / svgW;
    const scaleY = rect.height / svgH;
    const newZoom = Math.min(scaleX, scaleY, 1) * 0.95;
    setZoom(newZoom);
    setPan({ x: (rect.width - svgW * newZoom) / 2, y: 0 });
  }, [svgW, svgH]);

  useEffect(() => { fitToView(); }, [fitToView]);

  return (
    <div className="relative border border-gray-200 rounded-lg bg-white" style={{ height: "60vh" }}>
      {/* Zoom controls */}
      <div className="absolute top-2 right-2 z-10 flex items-center gap-1 bg-white/90 border border-gray-200 rounded-md px-1 py-0.5 shadow-sm">
        <button onClick={() => setZoom(z => Math.min(3, z + 0.2))} className="p-1 hover:bg-gray-100 rounded" title="Zoom in">
          <ZoomIn className="w-3.5 h-3.5 text-gray-600" />
        </button>
        <span className="text-[10px] text-gray-500 font-mono w-10 text-center">{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom(z => Math.max(0.2, z - 0.2))} className="p-1 hover:bg-gray-100 rounded" title="Zoom out">
          <ZoomOut className="w-3.5 h-3.5 text-gray-600" />
        </button>
        <div className="w-px h-4 bg-gray-200 mx-0.5" />
        <button onClick={fitToView} className="p-1 hover:bg-gray-100 rounded" title="Fit to view">
          <Maximize2 className="w-3.5 h-3.5 text-gray-600" />
        </button>
      </div>

      {/* Pan/zoom canvas */}
      <div
        ref={containerRef}
        className="w-full h-full overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          width={svgW}
          height={svgH}
          viewBox={`0 0 ${svgW} ${svgH}`}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
            transition: dragging ? "none" : "transform 0.15s ease",
          }}
        >
          <defs>
            <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#94a3b8" />
            </marker>
            <marker id="arr-y" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#3b82f6" />
            </marker>
            <marker id="arr-n" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#f97316" />
            </marker>
          </defs>

          {/* Edges */}
          {layout.edges.map((e, i) => {
            const f = nodePos[e.from];
            const t = nodePos[e.to];
            if (!f || !t) return null;

            const x1 = PAD + f.x + halfW;
            const y1 = PAD + f.y + layout.nodeHeight;
            const x2 = PAD + t.x + halfW;
            const y2 = PAD + t.y;

            const isY = e.label === "yes";
            const isN = e.label === "no";
            const color = isY ? "#3b82f6" : isN ? "#f97316" : "#94a3b8";

            // Handle upward edges (back-edges to earlier layers)
            const goesUp = y2 <= y1;
            let path: string;
            if (goesUp) {
              // Route around the side for back-edges
              const offset = 20;
              const sideX = Math.max(PAD + f.x + layout.nodeWidth + offset, PAD + t.x + layout.nodeWidth + offset);
              path = `M ${x1} ${y1} L ${x1} ${y1 + 15} L ${sideX} ${y1 + 15} L ${sideX} ${y2 - 15} L ${x2} ${y2 - 15} L ${x2} ${y2}`;
            } else {
              const mid = (y1 + y2) / 2;
              path = `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
            }

            const labelX = goesUp ? (x1 + x2) / 2 + 40 : (x1 + x2) / 2;
            const labelY = goesUp ? (y1 + y2) / 2 : (y1 + y2) / 2;

            return (
              <g key={`e-${i}`}>
                <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray={goesUp ? "4 2" : "none"} markerEnd={`url(#${isY ? "arr-y" : isN ? "arr-n" : "arr"})`} />
                {e.label !== "next" && (
                  <>
                    <rect x={labelX - 14} y={labelY - 9} width={28} height={16} rx={4} fill={isY ? "#dbeafe" : isN ? "#ffedd5" : "#f1f5f9"} stroke={color} strokeWidth={0.5} />
                    <text x={labelX} y={labelY + 3} textAnchor="middle" fontSize={9} fontWeight="bold" fill={color}>{e.label.toUpperCase()}</text>
                  </>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {layout.nodes.map((node) => {
            const x = PAD + node.x;
            const y = PAD + node.y;
            const isCond = node.type === "condition";

            return (
              <g key={node.id}>
                <rect
                  x={x} y={y}
                  width={layout.nodeWidth} height={layout.nodeHeight}
                  rx={8}
                  fill={isCond ? "#eff6ff" : "#f0fdf4"}
                  stroke={isCond ? "#3b82f6" : "#22c55e"}
                  strokeWidth={1.5}
                />
                <rect x={x + 4} y={y + 4} width={26} height={14} rx={3} fill={isCond ? "#3b82f6" : "#22c55e"} />
                <text x={x + 17} y={y + 14} textAnchor="middle" fill="white" fontSize={8} fontWeight="bold">{node.id}</text>
                <foreignObject x={x + 4} y={y + 22} width={layout.nodeWidth - 8} height={layout.nodeHeight - 26}>
                  <div style={{ fontSize: "10px", lineHeight: "1.3", color: "#1f2937", padding: "0 2px", overflow: "hidden" }}>
                    {node.text}
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}


// ─── Main GuidelineViewer ────────────────────────────────────────────

export default function GuidelineViewer({ guideline, onClose }: GuidelineViewerProps) {
  const [activeTab, setActiveTab] = useState<"tree" | "json" | "evaluators">("tree");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    let content = "";
    if (activeTab === "tree") {
      content = guideline.rules.join("\n\n");
    } else if (activeTab === "json") {
      content = JSON.stringify(guideline, null, 2);
    } else if (activeTab === "evaluators" && guideline.condition_evaluators) {
      content = JSON.stringify(guideline.condition_evaluators, null, 2);
    }
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{guideline.name}</h2>
            <p className="text-sm text-gray-500">{guideline.version}</p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b">
          {(["tree", "json", "evaluators"] as const).map((tab) => {
            const icons = { tree: GitBranch, json: FileJson, evaluators: Settings };
            const labels = { tree: "Decision Tree", json: "JSON Structure", evaluators: "Evaluators" };
            const Icon = icons[tab];
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                <Icon className="w-4 h-4" />
                {labels[tab]}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === "tree" ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-4">
                  <h3 className="text-sm font-semibold text-gray-700">Clinical Decision Tree</h3>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-blue-100 border border-blue-500 inline-block" />
                      Condition ({guideline.nodes.filter(n => n.type === "condition").length})
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-green-100 border border-green-500 inline-block" />
                      Action ({guideline.nodes.filter(n => n.type === "action").length})
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-1 bg-blue-500 inline-block rounded" /> Yes
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-1 bg-orange-500 inline-block rounded" /> No
                    </span>
                  </div>
                </div>
                <Button onClick={handleCopy} variant="outline" size="sm" className="gap-2">
                  {copied ? <><Check className="w-4 h-4" />Copied!</> : <><Copy className="w-4 h-4" />Copy Rules</>}
                </Button>
              </div>
              <DecisionTreeView guideline={guideline} />
            </div>
          ) : activeTab === "json" ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">Complete JSON Structure</h3>
                <Button onClick={handleCopy} variant="outline" size="sm" className="gap-2">
                  {copied ? <><Check className="w-4 h-4" />Copied!</> : <><Copy className="w-4 h-4" />Copy JSON</>}
                </Button>
              </div>
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-auto text-xs">
                {JSON.stringify(guideline, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  Condition Evaluators
                  {guideline.condition_evaluators && ` (${Object.keys(guideline.condition_evaluators).length})`}
                </h3>
                {guideline.condition_evaluators && (
                  <Button onClick={handleCopy} variant="outline" size="sm" className="gap-2">
                    {copied ? <><Check className="w-4 h-4" />Copied!</> : <><Copy className="w-4 h-4" />Copy JSON</>}
                  </Button>
                )}
              </div>
              {guideline.condition_evaluators ? (
                <div className="space-y-3">
                  {Object.entries(guideline.condition_evaluators).map(([nodeId, evaluator]) => {
                    const node = guideline.nodes.find(n => n.id === nodeId);
                    const evalType = "type" in evaluator ? evaluator.type : "boolean";
                    const badgeColors: Record<string, string> = {
                      bp_compare: "bg-red-100 text-red-700 border-red-200",
                      bp_range: "bg-pink-100 text-pink-700 border-pink-200",
                      age_compare: "bg-purple-100 text-purple-700 border-purple-200",
                      numeric_compare: "bg-blue-100 text-blue-700 border-blue-200",
                      or: "bg-orange-100 text-orange-700 border-orange-200",
                      and: "bg-green-100 text-green-700 border-green-200",
                    };
                    return (
                      <div key={nodeId} className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-xs font-mono font-semibold text-gray-900 bg-gray-100 px-2 py-1 rounded">{nodeId}</span>
                          <span className={`text-xs font-medium px-2 py-1 rounded border ${badgeColors[evalType] || "bg-gray-100 text-gray-700 border-gray-200"}`}>
                            {evalType.replace("_", " ").toUpperCase()}
                          </span>
                        </div>
                        {node && <p className="text-sm text-gray-600 mb-3 italic">&ldquo;{node.text}&rdquo;</p>}
                        <div className="bg-gray-50 rounded p-3">
                          <pre className="text-xs text-gray-800 font-mono">{JSON.stringify(evaluator, null, 2)}</pre>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Settings className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                  <h4 className="text-sm font-medium text-gray-900 mb-1">No Evaluators Available</h4>
                  <p className="text-sm text-gray-500">
                    This guideline was created before evaluator generation was implemented.<br />
                    Try uploading a new PDF to generate evaluators automatically.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t p-4 bg-gray-50">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <div>
              <span className="font-medium">Source:</span>{" "}
              <a href={guideline.citation_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                {guideline.citation}
              </a>
            </div>
            <Button onClick={onClose} variant="outline" size="sm">Close</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
