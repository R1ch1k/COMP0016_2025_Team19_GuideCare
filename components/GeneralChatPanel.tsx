"use client";

import { useState, useRef, useEffect } from "react";
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
} from "@/components/ui/shadcn-io/ai/prompt-input";
import { Response } from "@/components/ui/shadcn-io/ai/response";
import { Loader } from "@/components/ui/shadcn-io/ai/loader";
import { Trash2, Sparkles } from "lucide-react";

interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    isUiOnly?: boolean;
}

const SUGGESTED_QUESTIONS = [
    "What are the NICE criteria for starting antihypertensive treatment?",
    "Explain the FeverPAIN score and how to interpret it",
    "What are the red flag symptoms for headache that require urgent referral?",
    "When should I suspect secondary hypertension?",
    "What is the first-line antibiotic for community-acquired pneumonia?",
    "How do I interpret a raised HbA1c in a non-diabetic patient?",
];

const WELCOME_MESSAGE: ChatMessage = {
    role: "assistant",
    content: "Hello! I'm your general clinical assistant. Ask me anything — guidelines, drug queries, clinical reasoning, or medical concepts. I'm not tied to a specific patient or guideline here.",
    isUiOnly: true,
};

export default function GeneralChatPanel() {
    const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [streamingMessage, setStreamingMessage] = useState("");
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, streamingMessage]);

    useEffect(() => {
        return () => {
            abortControllerRef.current?.abort();
        };
    }, []);

    const sendMessage = async (text: string) => {
        if (!text.trim() || isLoading) return;

        const userMessage: ChatMessage = { role: "user", content: text };
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);
        setInput("");
        setIsLoading(true);
        setStreamingMessage("");

        abortControllerRef.current = new AbortController();

        try {
            // Exclude UI-only messages (e.g. the welcome message) from API history
            const apiMessages = updatedMessages
                .filter((m) => !m.isUiOnly)
                .map((m) => ({ role: m.role, content: m.content }));

            const response = await fetch("/api/general-chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages: apiMessages }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({}));
                throw new Error(errorBody.error || `Server error ${response.status}`);
            }

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error("No response body");

            let accumulated = "";
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
                                accumulated += parsed.content;
                                setStreamingMessage(accumulated);
                            }
                        } catch { /* skip */ }
                    }
                }
            }

            setMessages((prev) => [...prev, { role: "assistant", content: accumulated }]);
            setStreamingMessage("");
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") return;
            const detail = error instanceof Error ? error.message : "Unknown error";
            console.error("GeneralChat error:", detail);
            setMessages((prev) => [
                ...prev,
                {
                    role: "assistant",
                    content: "Sorry, I couldn't process that request. Please try again later.",
                },
            ]);
            setStreamingMessage("");
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        sendMessage(input);
    };

    const handleClear = () => {
        abortControllerRef.current?.abort();
        setMessages([WELCOME_MESSAGE]);
        setInput("");
        setStreamingMessage("");
        setIsLoading(false);
    };

    const showSuggestions = messages.length === 1; // Only the welcome message

    return (
        <div className="flex flex-col h-full bg-gray-50">
            <Conversation className="flex-1">
                <ConversationContent
                    className={`py-6 px-4 sm:px-6 md:px-8 h-full ${
                        showSuggestions ? "" : ""
                    }`}
                >
                    <div className="max-w-3xl w-full mx-auto">
                        <div className="space-y-4">
                            {messages.map((message, idx) => (
                                <Message key={idx} from={message.role}>
                                    <MessageAvatar src="" name={message.role === "user" ? "You" : "AI"} />
                                    <MessageContent>
                                        <Response>{message.content}</Response>
                                    </MessageContent>
                                </Message>
                            ))}

                            {/* Suggested questions — shown only on fresh chat */}
                            {showSuggestions && (
                                <div className="mt-2 ml-10">
                                    <p className="text-xs text-gray-500 mb-2 flex items-center gap-1">
                                        <Sparkles className="w-3 h-3" />
                                        Suggested questions
                                    </p>
                                    <div className="flex flex-col gap-1.5">
                                        {SUGGESTED_QUESTIONS.map((q, i) => (
                                            <button
                                                key={i}
                                                onClick={() => sendMessage(q)}
                                                className="text-left text-xs px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-700 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700 transition-colors shadow-sm"
                                            >
                                                {q}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {streamingMessage && (
                                <Message from="assistant">
                                    <MessageAvatar src="" name="AI" />
                                    <MessageContent>
                                        <Response>{streamingMessage}</Response>
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
                </ConversationContent>
                <ConversationScrollButton />
            </Conversation>

            <div className="border-t border-gray-200 px-4 sm:px-6 md:px-8 pt-6 pb-8 bg-white shadow-lg">
                <div className="max-w-3xl mx-auto">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-purple-500" />
                            <span className="text-xs text-gray-500">General Assistant</span>
                        </div>
                        <button
                            onClick={handleClear}
                            disabled={isLoading || messages.length <= 1}
                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg border border-gray-200 hover:border-red-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <Trash2 className="w-3.5 h-3.5" />
                            Clear chat
                        </button>
                    </div>
                    <PromptInput onSubmit={handleSubmit}>
                        <PromptInputTextarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={isLoading}
                            placeholder="Ask a clinical question..."
                        />
                        <PromptInputToolbar>
                            <PromptInputTools />
                            <PromptInputSubmit
                                disabled={!input.trim() || isLoading}
                                status={isLoading ? "streaming" : undefined}
                            />
                        </PromptInputToolbar>
                    </PromptInput>
                </div>
            </div>
        </div>
    );
}
