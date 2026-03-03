"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatMessage, AnyGuideline, NICEGuideline } from "@/lib/types";
import { PatientRecord } from "@/components/PatientInfoPanel";
import {
    Conversation,
    ConversationContent,
    ConversationScrollButton,
} from "@/components/ui/shadcn-io/ai/conversation";
import {
    Message,
    MessageAvatar,
    MessageContent,
} from "@/components/ui/shadcn-io/ai/message";
import {
    PromptInput,
    PromptInputTextarea,
    PromptInputToolbar,
    PromptInputTools,
    PromptInputSubmit,
    PromptInputButton,
} from "@/components/ui/shadcn-io/ai/prompt-input";
import { Response } from "@/components/ui/shadcn-io/ai/response";
import { Loader } from "@/components/ui/shadcn-io/ai/loader";
import { PlusIcon, Info, ChevronDown, ChevronUp, GitBranch } from "lucide-react";
import RuleExplanationModal from "./RuleExplanationModal";
import PipelineViewer from "./PipelineViewer";

const BACKEND_WS_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL || "ws://localhost:8000";
const BACKEND_HTTP_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Type guard to check if guideline is NICE format
function isNICEGuideline(guideline: AnyGuideline): guideline is NICEGuideline {
    return 'rules' in guideline && 'edges' in guideline;
}

interface ChatPanelProps {
    guideline: AnyGuideline | null;
    allGuidelines?: AnyGuideline[];
    selectedPatient?: PatientRecord | null;
    onDiagnosisComplete?: () => void;
}

// Parse backend pathway steps like "n1(action)" into human-readable format
function parsePathwayStep(step: string): { nodeId: string; decision: string; isAction: boolean } {
    const match = step.match(/^(n\d+)\((.+)\)$/);
    if (!match) return { nodeId: step, decision: "", isAction: false };
    return {
        nodeId: match[1],
        decision: match[2],
        isAction: match[2] === "action",
    };
}

function PathwayViewer({ pathway, guidelineId, guideline }: { pathway: string[]; guidelineId?: string; guideline?: AnyGuideline | null }) {
    const [expanded, setExpanded] = useState(false);
    const steps = pathway.map(parsePathwayStep);

    // Build node text lookup from guideline data
    const nodeTextMap: Record<string, string> = {};
    if (guideline && isNICEGuideline(guideline) && guideline.nodes) {
        for (const node of guideline.nodes) {
            nodeTextMap[node.id] = node.text;
        }
    }

    return (
        <div className="mt-3">
            <button
                onClick={() => setExpanded(!expanded)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors shadow-sm"
            >
                <GitBranch className="w-3.5 h-3.5" />
                {guidelineId ? `${guidelineId} Decision Path` : "View Decision Path"}
                {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {expanded && (
                <div className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs">
                    <div className="space-y-0.5">
                        {steps.map((step, i) => {
                            const isLast = i === steps.length - 1;
                            const nodeText = nodeTextMap[step.nodeId] || "";

                            let badgeColor = "bg-gray-100 text-gray-600";
                            let lineColor = "bg-slate-300";
                            let dotColor = "bg-gray-400";

                            if (step.isAction) {
                                badgeColor = "bg-green-100 text-green-700";
                                dotColor = "bg-green-500";
                            } else if (step.decision === "yes") {
                                badgeColor = "bg-blue-100 text-blue-700";
                                dotColor = "bg-blue-500";
                                lineColor = "bg-blue-300";
                            } else if (step.decision === "no") {
                                badgeColor = "bg-orange-100 text-orange-700";
                                dotColor = "bg-orange-400";
                                lineColor = "bg-orange-300";
                            } else if (step.decision === "missing_variable") {
                                badgeColor = "bg-amber-100 text-amber-700";
                                dotColor = "bg-amber-500";
                            }

                            return (
                                <div key={i} className="flex items-start gap-3">
                                    {/* Vertical timeline */}
                                    <div className="flex flex-col items-center w-3 flex-shrink-0 pt-1">
                                        <div className={`w-2.5 h-2.5 rounded-full ${dotColor} flex-shrink-0`} />
                                        {!isLast && <div className={`w-px flex-1 min-h-[16px] ${lineColor}`} />}
                                    </div>
                                    {/* Content */}
                                    <div className="flex flex-col gap-0.5 pb-2 min-w-0">
                                        <div className="flex items-center gap-2">
                                            <span className="text-slate-400 font-mono text-[10px]">{step.nodeId}</span>
                                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${badgeColor}`}>
                                                {step.isAction ? "action" : step.decision === "missing_variable" ? "needs data" : step.decision}
                                            </span>
                                        </div>
                                        {nodeText && (
                                            <span className="text-slate-700 text-xs leading-snug">{nodeText}</span>
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

export default function ChatPanel({ guideline, allGuidelines, selectedPatient, onDiagnosisComplete }: ChatPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [streamingMessage, setStreamingMessage] = useState("");
    const [sessionId, setSessionId] = useState(() => Date.now());
    const [explanationPath, setExplanationPath] = useState<string[]>([]);
    const [isExplanationOpen, setIsExplanationOpen] = useState(false);
    const [wsConnected, setWsConnected] = useState(false);
    const [backendPatientId, setBackendPatientId] = useState<string | null>(null);
    const [nodesVisited, setNodesVisited] = useState<string[]>([]);
    const [pipelineMeta, setPipelineMeta] = useState<{
        selectedGuideline?: string;
        urgency?: string;
        extractedVarCount?: number;
    }>({});
    const wsRef = useRef<WebSocket | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const initializedRef = useRef(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Resolve backend patient UUID from frontend patient record
    useEffect(() => {
        if (!selectedPatient) {
            setBackendPatientId(null);
            return;
        }

        // If the patient already has a backend UUID, use it directly
        if (selectedPatient.backendId) {
            setBackendPatientId(selectedPatient.backendId);
            return;
        }

        // Fallback: resolve by matching name against backend patients
        async function resolvePatient() {
            try {
                const res = await fetch(`${BACKEND_HTTP_URL}/patients`);
                if (!res.ok) return;
                const patients = await res.json();
                const nameParts = selectedPatient!.name.split(" ");
                const match = patients.find((p: { first_name: string; last_name: string }) =>
                    p.first_name === nameParts[0] && p.last_name === nameParts.slice(1).join(" ")
                );
                if (match) {
                    setBackendPatientId(match.id);
                } else {
                    console.warn("No backend match found for patient:", selectedPatient!.name);
                    setBackendPatientId(null);
                }
            } catch (err) {
                console.warn("Failed to resolve backend patient:", err);
            }
        }
        resolvePatient();
    }, [selectedPatient]);

    // WebSocket connection management
    const connectWebSocket = useCallback((patientId: string) => {
        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        const ws = new WebSocket(`${BACKEND_WS_URL}/ws/chat/${patientId}`);

        ws.onopen = () => {
            console.log("WebSocket connected to backend for patient", patientId);
            setWsConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === "error") {
                    const errorMsg: ChatMessage = {
                        role: "assistant",
                        content: `Error: ${data.detail || "Unknown error from backend"}`,
                    };
                    setMessages((prev) => [...prev, errorMsg]);
                    setIsLoading(false);
                    return;
                }

                // Skip user message echoes (backend broadcasts user msgs back)
                if (data.type === "message" && data.message?.role === "user") {
                    return;
                }

                // Acknowledge new conversation created
                if (data.type === "new_conversation_ack") {
                    console.log("New conversation created:", data.conversation_id);
                    return;
                }

                // Handle assistant responses (clarification_question or final)
                if (data.message && data.message.role === "assistant") {
                    const content = data.message.content || "";
                    const payload = data.payload || {};

                    // Update pipeline visualization from nodes_visited
                    if (payload.nodes_visited && Array.isArray(payload.nodes_visited)) {
                        setNodesVisited(payload.nodes_visited);
                    }

                    // Update pipeline metadata for dynamic labels
                    setPipelineMeta(prev => ({
                        ...prev,
                        ...(payload.selected_guideline && { selectedGuideline: payload.selected_guideline }),
                        ...(payload.urgent_escalation !== undefined && { urgency: payload.urgent_escalation ? "emergency" : (payload.triage_urgency || prev.urgency) }),
                        ...(payload.extracted_variables && { extractedVarCount: Object.keys(payload.extracted_variables).length }),
                    }));

                    let displayContent = content;
                    const isUrgent = payload.urgent_escalation === true;

                    // For final recommendations, include guideline info
                    if (payload.type === "final" && !isUrgent) {
                        const guidelineId = payload.selected_guideline || "";
                        const citation = payload.meta?.citation || "";
                        // Avoid "NICE NICE" duplication
                        const cleanCitation = citation.replace(/^NICE\s+/i, "");
                        const cleanGuideline = guidelineId.replace(/^NICE\s+/i, "");
                        if (cleanCitation && !displayContent.includes(cleanCitation)) {
                            displayContent += `\n\nThis recommendation is based on NICE ${cleanCitation}.`;
                        } else if (cleanGuideline && !displayContent.includes(cleanGuideline)) {
                            displayContent += `\n\nBased on NICE ${cleanGuideline}.`;
                        }
                    }

                    // For clarification questions, show what we know so far
                    if (payload.type === "clarification_question" && payload.selected_guideline) {
                        const vars = payload.extracted_variables || {};
                        const varCount = Object.keys(vars).length;
                        if (varCount > 0) {
                            displayContent += `\n\n*[Guideline: ${payload.selected_guideline} | Variables collected: ${varCount}]*`;
                        }
                    }

                    const assistantMsg: ChatMessage = {
                        role: "assistant",
                        content: displayContent,
                        ...(isUrgent && { isUrgent: true }),
                        ...(payload.type === "final" && payload.pathway_walked?.length > 0 && {
                            pathwayWalked: payload.pathway_walked,
                            selectedGuideline: payload.selected_guideline,
                        }),
                    };
                    setMessages((prev) => [...prev, assistantMsg]);
                    setIsLoading(false);
                    setStreamingMessage("");

                    // Notify parent that a diagnosis completed so patient records refresh
                    if (payload.type === "final" || isUrgent) {
                        onDiagnosisComplete?.();
                    }
                }
            } catch (err) {
                console.error("Failed to parse WebSocket message:", err);
            }
        };

        // Only update wsConnected if this is still the active WebSocket
        // Prevents race condition where old WS onclose fires after new WS onopen
        ws.onclose = () => {
            if (wsRef.current === ws) {
                console.log("WebSocket disconnected");
                setWsConnected(false);
            }
        };

        ws.onerror = (err) => {
            if (wsRef.current === ws) {
                console.error("WebSocket error:", err);
                setWsConnected(false);
            }
        };

        wsRef.current = ws;
    }, []);

    // Connect/reconnect WebSocket when backend patient ID changes
    useEffect(() => {
        if (backendPatientId) {
            connectWebSocket(backendPatientId);
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [backendPatientId, sessionId, connectWebSocket]);

    // Reset initialized ref when guideline or patient changes
    useEffect(() => {
        initializedRef.current = false;
    }, [guideline, backendPatientId]);

    // Show greeting when patient is connected (guideline auto-selected by triage)
    useEffect(() => {
        if (initializedRef.current) return;

        if (backendPatientId && wsConnected) {
            initializedRef.current = true;
            const greeting: ChatMessage = {
                role: "assistant",
                content: `Hello! I'm your clinical decision support assistant powered by NICE guidelines. I have **${selectedPatient?.name}**'s records loaded.\n\nPlease describe the patient's symptoms and I'll run triage, select the appropriate guideline, and walk the decision tree to provide a recommendation.`,
            };
            setMessages([greeting]);
        } else if (!backendPatientId && !initializedRef.current) {
            // No patient selected — show a generic welcome
            initializedRef.current = true;
            const welcome: ChatMessage = {
                role: "assistant",
                content: "Hello! I'm your clinical decision support assistant. Please select a patient from the panel on the right to get started.",
            };
            setMessages([welcome]);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [guideline, sessionId, backendPatientId, wsConnected]);

    // Fallback: direct OpenAI chat (when no patient selected / backend not available)
    const sendInitialGreetingFallback = async () => {
        if (!guideline) return;

        setIsLoading(true);
        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [{ role: "user", content: "START_CONVERSATION" }],
                    guideline,
                    decision: null,

                    patientContext: selectedPatient ? {
                        name: selectedPatient.name,
                        age: selectedPatient.age,
                        id: selectedPatient.id,
                        primaryConcern: selectedPatient.primaryConcern,
                        status: selectedPatient.status,
                        clinician: selectedPatient.clinician,
                        notes: selectedPatient.notes,
                    } : null,
                }),
                signal: abortController.signal,
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error("No response body");

            let accumulatedMessage = "";
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                for (const line of chunk.split("\n")) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        if (data === "[DONE]") continue;
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.content) {
                                accumulatedMessage += parsed.content;
                                setStreamingMessage(accumulatedMessage);
                            }
                        } catch { /* skip */ }
                    }
                }
            }

            setMessages([{ role: "assistant", content: accumulatedMessage }]);
            setStreamingMessage("");
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") return;
            setMessages([{
                role: "assistant",
                content: `Hello! I'm your clinical decision support assistant${guideline?.name ? ` for **${guideline.name}**` : ""}. How can I help you today?`,
            }]);
            setStreamingMessage("");
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    };

    const handleNewConversation = useCallback(() => {
        // Tell backend to close current conversation so a fresh one is created
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "new_conversation" }));
        }
        setMessages([]);
        setInput("");
        setStreamingMessage("");
        setNodesVisited([]);
        setPipelineMeta({});
        initializedRef.current = false;
        // Small delay so the backend processes the close before we reconnect
        setTimeout(() => setSessionId(Date.now()), 150);
    }, []);

    // Send via WebSocket (backend LangGraph pipeline)
    const sendViaWebSocket = (text: string) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error("WebSocket not connected");
            return;
        }

        wsRef.current.send(JSON.stringify({
            role: "user",
            content: text,
        }));
    };

    // Send via fallback direct OpenAI API
    const sendViaFallback = async (userMessage: ChatMessage) => {
        if (!guideline) return;

        abortControllerRef.current = new AbortController();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [...messages, userMessage],
                    guideline,
                    decision: null,

                    patientContext: selectedPatient ? {
                        name: selectedPatient.name,
                        age: selectedPatient.age,
                        id: selectedPatient.id,
                        primaryConcern: selectedPatient.primaryConcern,
                        status: selectedPatient.status,
                        clinician: selectedPatient.clinician,
                        notes: selectedPatient.notes,
                    } : null,
                }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error("No response body");

            let accumulatedMessage = "";
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                for (const line of chunk.split("\n")) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        if (data === "[DONE]") continue;
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.content) {
                                accumulatedMessage += parsed.content;
                                setStreamingMessage(accumulatedMessage);
                            }
                        } catch { /* skip */ }
                    }
                }
            }

            setMessages((prev) => [...prev, { role: "assistant", content: accumulatedMessage }]);
            setStreamingMessage("");
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") return;
            setMessages((prev) => [...prev, {
                role: "assistant",
                content: `Error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again.`,
            }]);
            setStreamingMessage("");
        } finally {
            abortControllerRef.current = null;
        }
    };

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        const useBackend = backendPatientId && wsConnected && wsRef.current?.readyState === WebSocket.OPEN;
        if (!input.trim() || (!guideline && !useBackend)) return;

        const userMessage: ChatMessage = { role: "user", content: input };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);
        setStreamingMessage("");

        if (useBackend) {
            // Send through backend LangGraph pipeline
            sendViaWebSocket(input);
        } else {
            // Fallback: direct OpenAI via Next.js API route
            await sendViaFallback(userMessage);
            setIsLoading(false);
        }
    };

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, streamingMessage]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    // Connection status indicator
    const connectionStatus = backendPatientId
        ? wsConnected ? "pipeline" : "connecting"
        : "direct";

    return (
        <div className="flex flex-col h-full bg-gray-50">
            <Conversation className="flex-1">
                <ConversationContent
                    className={`py-6 px-4 sm:px-6 md:px-8 h-full ${messages.length === 0
                            ? "flex items-center justify-center"
                            : ""
                        }`}
                >
                    {messages.length === 0 ? (
                        <div className="flex flex-col items-center justify-center text-center text-gray-500 max-w-2xl w-full">
                            <Loader />
                            <p className="text-sm font-medium text-gray-700 mb-1 mt-4">
                                Initializing Clinical Assistant
                            </p>
                            <p className="text-xs text-gray-600">
                                Loading {guideline?.name || "Clinical Assistant"}...
                            </p>
                        </div>
                    ) : (
                        <div className="max-w-5xl w-full">
                            <div className="space-y-4">
                                {messages.map((message, idx) => {
                                    const pathMatch = message.content.match(/\[\[PATH: (.*?)\]\]/);
                                    let displayContent = message.content;
                                    let path: string[] | null = null;

                                    if (pathMatch) {
                                        displayContent = message.content.replace(pathMatch[0], "").trim();
                                        path = pathMatch[1].split(",").map((s) => s.trim());
                                    }

                                    return (
                                        <Message key={idx} from={message.role}>
                                            <MessageAvatar
                                                src={message.role === "user" ? "" : ""}
                                                name={message.role === "user" ? "You" : "AI"}
                                            />
                                            <MessageContent>
                                                {message.isUrgent ? (
                                                    <div className="rounded-lg border-2 border-red-500 bg-red-50 p-4 shadow-md">
                                                        <div className="flex items-center gap-2 mb-3">
                                                            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-red-600 text-white font-bold text-lg animate-pulse">!</span>
                                                            <span className="text-red-800 font-bold text-base uppercase tracking-wide">Urgent Clinical Alert</span>
                                                        </div>
                                                        <div className="text-red-900 text-sm font-medium leading-relaxed">
                                                            <Response>{displayContent}</Response>
                                                        </div>
                                                        <div className="mt-3 pt-3 border-t border-red-200 flex items-center gap-2">
                                                            <span className="inline-block w-2 h-2 rounded-full bg-red-600 animate-pulse" />
                                                            <span className="text-xs font-semibold text-red-700 uppercase tracking-wider">Immediate action required</span>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <Response>{displayContent}</Response>
                                                )}
                                                {/* Backend pipeline pathway viewer */}
                                                {message.pathwayWalked && message.pathwayWalked.length > 0 && (
                                                    <PathwayViewer
                                                        pathway={message.pathwayWalked}
                                                        guidelineId={message.selectedGuideline}
                                                        guideline={
                                                            allGuidelines?.find(g => {
                                                                const gid = g.guideline_id.toLowerCase();
                                                                const sel = (message.selectedGuideline || '').toLowerCase();
                                                                return gid === sel || gid.includes(sel) || sel.includes(gid);
                                                            }) || guideline
                                                        }
                                                    />
                                                )}
                                                {/* Fallback direct-mode path viewer */}
                                                {path && guideline && isNICEGuideline(guideline) && (
                                                    <button
                                                        onClick={() => {
                                                            setExplanationPath(path!);
                                                            setIsExplanationOpen(true);
                                                        }}
                                                        className="mt-2 flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline transition-colors"
                                                    >
                                                        <Info className="w-3.5 h-3.5" />
                                                        View Decision Path
                                                    </button>
                                                )}
                                            </MessageContent>
                                        </Message>
                                    );
                                })}
                                {streamingMessage && (
                                    <Message from="assistant">
                                        <MessageAvatar src="" name="AI" />
                                        <MessageContent>
                                            <Response>{streamingMessage.replace(/\[\[PATH: .*?\]\]/, "")}</Response>
                                        </MessageContent>
                                    </Message>
                                )}
                                {isLoading && !streamingMessage && (
                                    <Message from="assistant">
                                        <MessageAvatar src="" name="AI" />
                                        <MessageContent>
                                            <Loader />
                                        </MessageContent>
                                    </Message>
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                        </div>
                    )}
                </ConversationContent>
                <ConversationScrollButton />
            </Conversation>

            <div className="border-t border-gray-200 px-4 sm:px-6 md:px-8 pt-6 pb-8 bg-white shadow-lg">
                <div className="w-full mx-auto">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2">
                                <span className={`inline-block w-2 h-2 rounded-full ${
                                    connectionStatus === "pipeline" ? "bg-green-500" :
                                    connectionStatus === "connecting" ? "bg-yellow-500 animate-pulse" :
                                    "bg-gray-400"
                                }`} />
                                <span className="text-xs text-gray-500">
                                    {connectionStatus === "pipeline" ? "LangGraph Pipeline" :
                                     connectionStatus === "connecting" ? "Connecting..." :
                                     "Direct Mode"}
                                </span>
                            </div>
                            {connectionStatus === "pipeline" && (
                                <div className="w-48">
                                    <PipelineViewer nodesVisited={nodesVisited} isProcessing={isLoading} meta={pipelineMeta} />
                                </div>
                            )}
                        </div>
                        <PromptInputButton
                            onClick={handleNewConversation}
                            disabled={isLoading}
                        >
                            <PlusIcon className="w-4 h-4" />
                            New Conversation
                        </PromptInputButton>
                    </div>
                    <PromptInput onSubmit={handleSend}>
                        <PromptInputTextarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={isLoading}
                            placeholder={selectedPatient
                                ? `Describe ${selectedPatient.name}'s symptoms...`
                                : "Select a patient, then describe their symptoms..."
                            }
                        />
                        <PromptInputToolbar>
                            <PromptInputTools>
                                {/* Add any additional tools here */}
                            </PromptInputTools>
                            <PromptInputSubmit
                                disabled={!input.trim() || isLoading}
                                status={isLoading ? "streaming" : undefined}
                            />
                        </PromptInputToolbar>
                    </PromptInput>
                </div>
            </div>
            {guideline && isNICEGuideline(guideline) && (
                <RuleExplanationModal
                    isOpen={isExplanationOpen}
                    onClose={() => setIsExplanationOpen(false)}
                    path={explanationPath}
                    guideline={guideline}
                />
            )}
        </div>
    );
}
