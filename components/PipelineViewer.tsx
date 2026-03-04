"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

/**
 * The 7 LangGraph pipeline nodes in execution order.
 * Node names must match the keys in graph.py.
 */
const PIPELINE_NODES = [
    { id: "load_patient", label: "Load Patient", description: "Fetch patient record from DB" },
    { id: "triage", label: "Triage", description: "LLM urgency assessment" },
    { id: "select_guideline", label: "Select Guideline", description: "Choose NICE guideline" },
    { id: "clarify", label: "Clarify", description: "Ask guideline-specific questions" },
    { id: "extract_variables", label: "Extract Variables", description: "LLM + regex variable extraction" },
    { id: "walk_graph", label: "Walk Graph", description: "BFS traversal of decision tree" },
    { id: "format_output", label: "Format Output", description: "Template-based recommendation" },
];

/** Map short guideline IDs to human-readable names. */
const GUIDELINE_NAMES: Record<string, string> = {
    NG84: "Sore Throat",
    NG91: "Otitis Media",
    NG112: "Urinary Tract Infection",
    NG133: "Hypertension in Pregnancy",
    NG136: "Hypertension",
    NG184: "Animal & Human Bites",
    NG222: "Depression",
    NG232: "Head Injury",
    NG81_GLAUCOMA: "Chronic Glaucoma",
    NG81_HYPERTENSION: "Ocular Hypertension",
};

function getGuidelineDisplayName(id: string): string {
    const upper = id.toUpperCase();
    for (const [key, name] of Object.entries(GUIDELINE_NAMES)) {
        if (upper.includes(key)) return `${key} (${name})`;
    }
    return id;
}

interface PipelineMeta {
    selectedGuideline?: string;
    urgency?: string;
    extractedVarCount?: number;
}

interface PipelineViewerProps {
    nodesVisited: string[];
    isProcessing?: boolean;
    meta?: PipelineMeta;
}

/** Build a dynamic label for a node based on pipeline metadata. */
function getDynamicLabel(nodeId: string, defaultLabel: string, isVisited: boolean, meta: PipelineMeta): { label: string; detail?: string } {
    if (!isVisited) return { label: defaultLabel };

    switch (nodeId) {
        case "select_guideline":
            if (meta.selectedGuideline) {
                return { label: "Guideline", detail: getGuidelineDisplayName(meta.selectedGuideline) };
            }
            return { label: defaultLabel };
        case "extract_variables":
            if (meta.extractedVarCount && meta.extractedVarCount > 0) {
                return { label: "Variables", detail: `${meta.extractedVarCount} extracted` };
            }
            return { label: defaultLabel };
        default:
            return { label: defaultLabel };
    }
}

export default function PipelineViewer({ nodesVisited, isProcessing, meta = {} }: PipelineViewerProps) {
    const [expanded, setExpanded] = useState(true);

    const visitedSet = new Set(nodesVisited);
    // Determine the currently active node (last visited while still processing)
    const activeNode = isProcessing && nodesVisited.length > 0 ? nodesVisited[nodesVisited.length - 1] : null;

    return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-700 hover:bg-gray-50 rounded-t-lg transition-colors"
            >
                <span className="flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5 text-blue-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    Pipeline
                    {nodesVisited.length > 0 && (
                        <span className="px-1.5 py-0.5 text-[10px] font-bold bg-blue-100 text-blue-700 rounded">
                            {visitedSet.size}/{PIPELINE_NODES.length}
                        </span>
                    )}
                </span>
                {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {expanded && (
                <div className="px-3 pb-3 pt-1">
                    <div className="space-y-0">
                        {PIPELINE_NODES.map((node, i) => {
                            const isVisited = visitedSet.has(node.id);
                            const isActive = node.id === activeNode;
                            const isLast = i === PIPELINE_NODES.length - 1;
                            const { label, detail } = getDynamicLabel(node.id, node.label, isVisited, meta);

                            let dotColor = "bg-gray-300";
                            let textColor = "text-gray-400";
                            let lineColor = "bg-gray-200";

                            if (isActive) {
                                dotColor = "bg-blue-500 animate-pulse";
                                textColor = "text-blue-700 font-semibold";
                                lineColor = "bg-blue-300";
                            } else if (isVisited) {
                                dotColor = "bg-green-500";
                                textColor = "text-gray-700";
                                lineColor = "bg-green-300";
                            }

                            return (
                                <div key={node.id} className="flex items-start gap-2">
                                    {/* Vertical timeline */}
                                    <div className="flex flex-col items-center w-3 flex-shrink-0 pt-[5px]">
                                        <div className={`w-2 h-2 rounded-full ${dotColor} flex-shrink-0 transition-colors`} />
                                        {!isLast && <div className={`w-px flex-1 min-h-[14px] ${lineColor} transition-colors`} />}
                                    </div>
                                    {/* Label */}
                                    <div className="pb-1 min-w-0 flex items-baseline gap-1">
                                        <span className={`text-[11px] leading-tight ${textColor} transition-colors`}>
                                            {label}
                                        </span>
                                        {detail && (
                                            <span className="text-[10px] text-emerald-600 font-medium truncate">
                                                {detail}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
