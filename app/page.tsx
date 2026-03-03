"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ChatPanel from "@/components/ChatPanel";
import { AnyGuideline, NICEGuideline } from "@/lib/types";
import PatientInfoPanel, {
    PatientRecord,
} from "@/components/PatientInfoPanel";
import AddPatientModal from "@/components/AddPatientModal";
import ResizablePanel from "@/components/ResizablePanel";
import GuidelineViewer from "@/components/GuidelineViewer";
import { niceHypertensionGuideline } from "@/lib/guidelines/nice-hypertension";
import { Eye } from "lucide-react";
import SampleInputModal from "@/components/SampleInputModal";
import ConnectDataModal from "@/components/ConnectDataModal";

const BACKEND_HTTP_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// All NICE guideline JSON files in public/guidelines/
const GUIDELINE_FILES = [
    "ng84.json",
    "ng91.json",
    "ng112.json",
    "ng133.json",
    "ng136.json",
    "ng184.json",
    "ng222.json",
    "ng232.json",
    "ng81_chronic_glaucoma.json",
    "ng81_ocular_hypertension.json",
];

// Type guard to check if guideline is NICE format
function isNICEGuideline(guideline: AnyGuideline): guideline is NICEGuideline {
    return 'rules' in guideline && 'edges' in guideline;
}

export default function Home() {
    const [guidelines, setGuidelines] = useState<AnyGuideline[]>([]);
    const [activeGuideline, setActiveGuideline] = useState<AnyGuideline | null>(
        null
    );
    const [showGuidelineSelector, setShowGuidelineSelector] = useState(false); // Closed by default
    const [sessionKey, setSessionKey] = useState(0); // Used to reset chat when guideline changes
    const [isUploadingPdf, setIsUploadingPdf] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [showPatientPanel, setShowPatientPanel] = useState(true);
    const [isAddPatientOpen, setIsAddPatientOpen] = useState(false);
    const [isSampleInputOpen, setIsSampleInputOpen] = useState(false);
    const [isConnectDataOpen, setIsConnectDataOpen] = useState(false);
    const [viewerGuideline, setViewerGuideline] = useState<NICEGuideline | null>(null);
    const [selectedPatient, setSelectedPatient] = useState<PatientRecord | null>(null);

    // Load all NICE guidelines from public/guidelines/ on mount
    useEffect(() => {
        async function loadGuidelines() {
            const loaded: NICEGuideline[] = [];
            for (const file of GUIDELINE_FILES) {
                try {
                    const res = await fetch(`/guidelines/${file}`);
                    if (res.ok) {
                        const data = await res.json();
                        loaded.push(data as NICEGuideline);
                    }
                } catch {
                    console.warn(`Failed to load guideline: ${file}`);
                }
            }
            if (loaded.length > 0) {
                setGuidelines(loaded);
                // No guideline selected by default — user picks after selecting a patient
            } else {
                // Fallback to hardcoded hypertension guideline (available but not selected)
                setGuidelines([niceHypertensionGuideline]);
            }
        }
        loadGuidelines();
    }, []);
    const [patientRecords, setPatientRecords] = useState<PatientRecord[]>([]);
    const [patientsLoaded, setPatientsLoaded] = useState(false);

    // Load patients from backend on mount for persistence
    const loadPatientsFromBackend = useCallback(async () => {
        try {
            const res = await fetch(`${BACKEND_HTTP_URL}/patients`);
            if (!res.ok) return;
            const patients = await res.json();
            const records: PatientRecord[] = patients.map((p: {
                id: string;
                first_name: string;
                last_name: string;
                age: number;
                gender?: string;
                date_of_birth?: string;
                nhs_number?: string;
                conditions?: string[];
                medications?: Array<{ name: string; dose?: string }>;
                allergies?: string[];
                clinical_notes?: Array<{ date: string; guideline: string; recommendation: string; urgency?: string; note?: string }>;
                recent_vitals?: Record<string, string | number>;
                updated_at?: string;
            }) => ({
                id: p.id,
                backendId: p.id,
                name: `${p.first_name} ${p.last_name}`,
                age: p.age,
                gender: p.gender,
                dateOfBirth: p.date_of_birth,
                nhsNumber: p.nhs_number,
                primaryConcern: (p.conditions && p.conditions[0]) || "General",
                status: "Needs Attention" as const,
                lastUpdated: p.updated_at
                    ? new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" }).format(new Date(p.updated_at))
                    : formatLastUpdated(),
                clinician: "Assigned clinician",
                notes: p.clinical_notes?.[0]?.note || (p.clinical_notes?.[0]?.recommendation ? p.clinical_notes[0].recommendation.slice(0, 100) + "..." : undefined),
                conditions: p.conditions,
                medications: p.medications,
                allergies: p.allergies,
                recentVitals: p.recent_vitals,
                clinicalNotes: p.clinical_notes,
            }));
            setPatientRecords(records);
        } catch (err) {
            console.warn("Failed to load patients from backend:", err);
        } finally {
            setPatientsLoaded(true);
        }
    }, []);

    useEffect(() => {
        loadPatientsFromBackend();
    }, [loadPatientsFromBackend]);

    const formatLastUpdated = () => {
        return new Intl.DateTimeFormat("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric",
        })
            .format(new Date())
            .replace(/\s/g, " ")
            .replace(",", "");
    };

    const handlePatientSave = (record: Omit<PatientRecord, "lastUpdated">) => {
        const lastUpdated = formatLastUpdated();

        setPatientRecords((prev) => {
            const filtered = prev.filter((existing) => existing.id !== record.id);
            return [
                {
                    ...record,
                    lastUpdated,
                },
                ...filtered,
            ];
        });

        // Auto-select the newly created patient
        setSelectedPatient({
            ...record,
            lastUpdated,
        });
        setSessionKey((prev) => prev + 1);
    };

    const handleGuidelineSelect = (guideline: AnyGuideline) => {
        setActiveGuideline(guideline);
        setShowGuidelineSelector(false);
        setSessionKey((prev) => prev + 1); // Force chat to reset
    };

    const handleGuidelineDelete = (guidelineId: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent selecting the guideline when clicking delete

        if (guidelines.length === 1) {
            alert(
                "You cannot delete the last guideline. At least one guideline must be available."
            );
            return;
        }

        if (!confirm("Are you sure you want to delete this guideline?")) {
            return;
        }

        setGuidelines((prev) => prev.filter((g) => g.guideline_id !== guidelineId));

        // If the deleted guideline was active, switch to the first remaining guideline
        if (activeGuideline?.guideline_id === guidelineId) {
            const remainingGuidelines = guidelines.filter(
                (g) => g.guideline_id !== guidelineId
            );
            if (remainingGuidelines.length > 0) {
                setActiveGuideline(remainingGuidelines[0]);
                setSessionKey((prev) => prev + 1);
            }
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploadingPdf(true);
        setUploadError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/parse-pdf", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (data.error) {
                setUploadError(data.error);
                setIsUploadingPdf(false);
                return;
            }

            const newGuideline: AnyGuideline = data.guideline;

            // Check if guideline already exists
            const exists = guidelines.some(
                (g) => g.guideline_id === newGuideline.guideline_id
            );

            if (!exists) {
                setGuidelines((prev) => [...prev, newGuideline]);
            }

            setActiveGuideline(newGuideline);
            setShowGuidelineSelector(false);
            setSessionKey((prev) => prev + 1); // Force chat to reset
            setIsUploadingPdf(false);
        } catch (error) {
            setUploadError(
                `Failed to upload guideline: ${error instanceof Error ? error.message : "Unknown error"
                }`
            );
            setIsUploadingPdf(false);
        }

        // Reset file input
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b border-gray-200 px-4 sm:px-6 md:px-8 py-4 shadow-sm">
                <div className="max-w-7xl mx-auto flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div className="flex items-start gap-4 min-w-0 flex-1">
                        <div className="min-w-0">
                            <h1 className="text-lg font-bold text-gray-900">
                                Clinical Decision Support
                            </h1>
                            <p className="text-sm text-gray-600 mt-0.5 truncate">
                                {activeGuideline?.name || "Select a guideline"}
                            </p>
                        </div>
                    </div>
                    <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-end">
                        <button
                            onClick={() => setShowGuidelineSelector(!showGuidelineSelector)}
                            className="w-full sm:w-auto px-5 py-2.5 text-sm font-medium rounded-lg bg-white text-gray-700 hover:bg-gray-100 border-2 border-gray-300 transition-all shadow-sm hover:shadow max-w-full sm:max-w-xs truncate"
                            title={
                                activeGuideline
                                    ? activeGuideline.guideline_id
                                    : "Select Guideline"
                            }
                        >
                            <span className="flex items-center justify-center gap-2">
                                <span>📚</span>
                                <span className="truncate">
                                    {activeGuideline
                                        ? activeGuideline.guideline_id
                                        : "Select Guideline"}
                                </span>
                            </span>
                        </button>
                        <div
                            className="hidden sm:block h-6 w-px bg-gray-300"
                            aria-hidden="true"
                        ></div>
                        <button
                            onClick={() => setIsSampleInputOpen(true)}
                            className="px-5 py-2.5 text-sm font-medium rounded-lg bg-white text-gray-700 hover:bg-gray-100 border-2 border-gray-300 transition-all shadow-sm hover:shadow"
                        >
                            💡 Sample Input
                        </button>
                        <button
                            onClick={() => setShowPatientPanel((prev) => !prev)}
                            className="w-full sm:w-auto px-5 py-2.5 text-sm font-semibold rounded-lg bg-white text-gray-700 hover:bg-gray-100 border-2 border-gray-300 transition-all shadow-sm hover:shadow flex items-center justify-center gap-2"
                            aria-pressed={showPatientPanel}
                        >
                            <span>🩺</span>
                            <span className="truncate">
                                {showPatientPanel ? "Hide Patient Panel" : "Show Patient Panel"}
                            </span>
                        </button>
                        <button
                            onClick={async () => {
                                await fetch("/api/auth/logout", { method: "POST" });
                                window.location.href = "/login";
                            }}
                            className="w-full sm:w-auto px-5 py-2.5 text-sm font-semibold rounded-lg bg-red-50 text-red-700 hover:bg-red-100 border-2 border-red-200 transition-all shadow-sm hover:shadow flex items-center justify-center gap-2"
                            title="Logout"
                        >
                            <span>🚪</span>
                            <span>Logout</span>
                        </button>
                    </div>
                </div>
            </header>

            {/* Guideline Selector Dropdown */}
            {showGuidelineSelector && (
                <div className="bg-white border-b border-gray-200 px-4 sm:px-6 md:px-8 py-6 shadow-md">
                    <div className="max-w-7xl mx-auto">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-semibold text-gray-900">
                                Select or Upload a Guideline
                            </h3>
                            <button
                                onClick={() => setShowGuidelineSelector(false)}
                                className="text-gray-500 hover:text-gray-700"
                            >
                                <svg
                                    className="w-5 h-5"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M6 18L18 6M6 6l12 12"
                                    />
                                </svg>
                            </button>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            {guidelines.map((guideline) => (
                                <div
                                    key={guideline.guideline_id}
                                    className={`relative group rounded-lg border-2 transition-all ${activeGuideline?.guideline_id === guideline.guideline_id
                                            ? "border-blue-500 bg-blue-50"
                                            : "border-gray-200 hover:border-gray-300 bg-white"
                                        }`}
                                >
                                    <button
                                        onClick={() => handleGuidelineSelect(guideline)}
                                        className="w-full text-left p-3"
                                    >
                                        <p className="text-xs font-semibold text-gray-900 pr-8">
                                            {guideline.name}
                                        </p>
                                        <p className="text-xs text-gray-600 mt-1">
                                            {guideline.guideline_id} • {guideline.version}
                                        </p>
                                    </button>
                                    {isNICEGuideline(guideline) && (
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setViewerGuideline(guideline);
                                            }}
                                            className="absolute bottom-2 right-2 flex items-center gap-1 px-2 py-1 text-xs font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded transition-colors"
                                        >
                                            <Eye className="w-3 h-3" />
                                            View
                                        </button>
                                    )}
                                    {guidelines.length > 1 && (
                                        <button
                                            onClick={(e) =>
                                                handleGuidelineDelete(guideline.guideline_id, e)
                                            }
                                            className="absolute top-2 right-2 p-1 rounded-md bg-red-100 text-red-600 opacity-0 group-hover:opacity-100 hover:bg-red-200 transition-opacity"
                                            title="Delete guideline"
                                        >
                                            <svg
                                                className="w-4 h-4"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                                                />
                                            </svg>
                                        </button>
                                    )}
                                </div>
                            ))}
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isUploadingPdf}
                                className={`text-left p-4 rounded-lg border-2 border-dashed transition-all relative overflow-hidden ${isUploadingPdf
                                        ? "border-blue-400 bg-blue-100 cursor-not-allowed"
                                        : activeGuideline
                                            ? "border-gray-300 hover:border-blue-400 hover:bg-blue-50"
                                            : "border-blue-400 bg-blue-50 hover:border-blue-500 hover:bg-blue-100 animate-pulse"
                                    }`}
                            >
                                {isUploadingPdf ? (
                                    <>
                                        <div className="absolute inset-0 bg-blue-200 opacity-50 animate-pulse"></div>
                                        <div className="relative z-10">
                                            <div className="flex items-center gap-2 mb-2">
                                                <svg
                                                    className="animate-spin h-5 w-5 text-blue-600"
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    fill="none"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <circle
                                                        className="opacity-25"
                                                        cx="12"
                                                        cy="12"
                                                        r="10"
                                                        stroke="currentColor"
                                                        strokeWidth="4"
                                                    ></circle>
                                                    <path
                                                        className="opacity-75"
                                                        fill="currentColor"
                                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                                    ></path>
                                                </svg>
                                                <p className="text-xs font-semibold text-blue-900">
                                                    Processing PDF...
                                                </p>
                                            </div>
                                            <p className="text-xs text-blue-700">
                                                Analyzing guideline structure and decision logic...
                                            </p>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="flex items-center gap-2">
                                            <svg
                                                className="w-5 h-5 text-blue-600"
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                                                />
                                            </svg>
                                            <p className="text-xs font-semibold text-gray-900">
                                                {activeGuideline
                                                    ? "Upload New Guideline"
                                                    : "Upload Guideline PDF"}
                                            </p>
                                        </div>
                                        <p className="text-xs text-gray-600 mt-1.5">
                                            {activeGuideline
                                                ? "Start a new session with a different guideline"
                                                : "Click to select a PDF file from your computer"}
                                        </p>
                                    </>
                                )}
                            </button>
                        </div>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            onChange={handleFileUpload}
                            className="hidden"
                        />
                        {uploadError && (
                            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                                <p className="text-xs text-red-800">
                                    <span className="font-semibold">Upload failed:</span>{" "}
                                    {uploadError}
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Main Chat Interface */}
            <div className="flex-1 overflow-hidden">
                <ResizablePanel
                    leftPanel={
                        <ChatPanel
                            key={sessionKey}
                            guideline={activeGuideline}
                            allGuidelines={guidelines}
                            selectedPatient={selectedPatient}
                            onDiagnosisComplete={loadPatientsFromBackend}
                        />
                    }
                    rightPanel={
                        <PatientInfoPanel
                            records={patientRecords}
                            onAddPatient={() => setIsAddPatientOpen(true)}
                            onConnect={() => setIsConnectDataOpen(true)}
                            selectedPatientId={selectedPatient?.id}
                            allGuidelines={guidelines}
                            onSelectPatient={(patient) => {
                                setSelectedPatient(patient);
                                setSessionKey((prev) => prev + 1);
                            }}
                            className="bg-white h-full w-full"
                        />
                    }
                    showRightPanel={showPatientPanel}
                    defaultRightWidth={600}
                    minRightWidth={400}
                    maxRightWidth={900}
                />
            </div>
            {isAddPatientOpen && (
                <AddPatientModal
                    onClose={() => setIsAddPatientOpen(false)}
                    onSave={(patient) => {
                        handlePatientSave(patient);
                    }}
                />
            )}
            {viewerGuideline && (
                <GuidelineViewer
                    guideline={viewerGuideline}
                    onClose={() => setViewerGuideline(null)}
                />
            )}
            <SampleInputModal
                isOpen={isSampleInputOpen}
                onClose={() => setIsSampleInputOpen(false)}
            />
            <ConnectDataModal
                isOpen={isConnectDataOpen}
                onClose={() => setIsConnectDataOpen(false)}
                onImportComplete={() => loadPatientsFromBackend()}
            />
        </div>
    );
}
