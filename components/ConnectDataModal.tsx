"use client";

import { useRef, useState } from "react";
// @ts-ignore - Custom dialog component
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/shadcn-io/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ConnectDataModalProps {
    isOpen: boolean;
    onClose: () => void;
    onImportComplete?: (count: number) => void;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const SAMPLE_CSV = `nhs_number,first_name,last_name,date_of_birth,gender,conditions,medications,allergies
123-456-7890,Jane,Smith,1985-03-15,Female,"[""Asthma"",""Anxiety""]","[{""name"":""Salbutamol"",""dose"":""100mcg""}]","[""Penicillin""]"
234-567-8901,Robert,Brown,1972-11-22,Male,"[""Type 2 Diabetes"",""Hypertension""]","[{""name"":""Metformin"",""dose"":""500mg""}]","[]"`;

const SAMPLE_SQL = `CREATE TABLE patients (
    nhs_number   VARCHAR(64) PRIMARY KEY,
    first_name   VARCHAR(128) NOT NULL,
    last_name    VARCHAR(128) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender       VARCHAR(32),
    conditions   JSONB DEFAULT '[]',
    medications  JSONB DEFAULT '[]',
    allergies    JSONB DEFAULT '[]'
);`;

export default function ConnectDataModal({ isOpen, onClose, onImportComplete }: ConnectDataModalProps) {
    const [activeTab, setActiveTab] = useState<"upload" | "database">("upload");
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState<{ imported: number; errors: string[] } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleUpload = async (file: File) => {
        setUploading(true);
        setError(null);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch(`${BACKEND_URL}/patients/import`, {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const detail = await res.json().catch(() => ({ detail: "Upload failed" }));
                throw new Error(detail.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            setResult({ imported: data.imported, errors: data.errors || [] });
            onImportComplete?.(data.imported);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Upload failed");
        } finally {
            setUploading(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleUpload(file);
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleUpload(file);
    };

    return (
        <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
            <DialogContent className="max-w-2xl overflow-hidden">
                <DialogHeader>
                    <DialogTitle>Connect Data Source</DialogTitle>
                </DialogHeader>

                {/* Tabs */}
                <div className="flex border-b border-gray-200 mb-4">
                    <button
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                            activeTab === "upload"
                                ? "border-blue-600 text-blue-600"
                                : "border-transparent text-gray-500 hover:text-gray-700"
                        }`}
                        onClick={() => setActiveTab("upload")}
                    >
                        Upload File
                    </button>
                    <button
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                            activeTab === "database"
                                ? "border-blue-600 text-blue-600"
                                : "border-transparent text-gray-500 hover:text-gray-700"
                        }`}
                        onClick={() => setActiveTab("database")}
                    >
                        Connect Database
                        <span className="ml-1.5 text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">
                            Coming Soon
                        </span>
                    </button>
                </div>

                <ScrollArea className="max-h-[60vh]">
                    {activeTab === "upload" ? (
                        <div className="space-y-4">
                            {/* Drop zone */}
                            <div
                                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                                onDragLeave={() => setIsDragging(false)}
                                onDrop={handleDrop}
                                onClick={() => fileInputRef.current?.click()}
                                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                                    isDragging
                                        ? "border-blue-400 bg-blue-50"
                                        : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                                }`}
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".csv,.xlsx"
                                    onChange={handleFileSelect}
                                    className="hidden"
                                />
                                <div className="text-gray-500">
                                    <p className="text-sm font-medium">
                                        {uploading ? "Uploading..." : "Drop a .csv or .xlsx file here, or click to browse"}
                                    </p>
                                    <p className="text-xs mt-1 text-gray-400">
                                        Patient data will be imported into the system
                                    </p>
                                </div>
                            </div>

                            {/* Result / Error */}
                            {result && (
                                <div className={`rounded-lg p-3 text-sm ${result.errors.length > 0 ? "bg-yellow-50 border border-yellow-200" : "bg-green-50 border border-green-200"}`}>
                                    <p className="font-medium text-green-800">
                                        Successfully imported {result.imported} patient(s).
                                    </p>
                                    {result.errors.length > 0 && (
                                        <div className="mt-2 text-yellow-800">
                                            <p className="font-medium">Errors ({result.errors.length}):</p>
                                            <ul className="list-disc pl-5 mt-1 text-xs space-y-0.5">
                                                {result.errors.map((err, i) => (
                                                    <li key={i}>{err}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            )}
                            {error && (
                                <div className="rounded-lg p-3 text-sm bg-red-50 border border-red-200 text-red-800">
                                    {error}
                                </div>
                            )}

                            {/* Expected format */}
                            <div>
                                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                                    Expected CSV Columns
                                </h4>
                                <div className="border border-gray-200 rounded-lg overflow-hidden">
                                    <table className="w-full text-xs">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="px-3 py-1.5 text-left font-semibold text-gray-600">Column</th>
                                                <th className="px-3 py-1.5 text-left font-semibold text-gray-600">Required</th>
                                                <th className="px-3 py-1.5 text-left font-semibold text-gray-600">Example</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {[
                                                ["nhs_number", "Yes", "123-456-7890"],
                                                ["first_name", "Yes", "Jane"],
                                                ["last_name", "Yes", "Smith"],
                                                ["date_of_birth", "Yes", "1985-03-15"],
                                                ["gender", "No", "Female"],
                                                ["conditions", "No", "Asthma, Anxiety"],
                                                ["medications", "No", "Salbutamol 100mcg"],
                                                ["allergies", "No", "Penicillin"],
                                            ].map(([col, req, ex]) => (
                                                <tr key={col}>
                                                    <td className="px-3 py-1 font-mono text-gray-700">{col}</td>
                                                    <td className="px-3 py-1 text-gray-500">{req}</td>
                                                    <td className="px-3 py-1 text-gray-400">{ex}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <p className="text-xs text-gray-400 mt-1">
                                    Conditions, medications, and allergies can be comma-separated or JSON arrays.
                                </p>
                            </div>

                            {/* Sample CSV download */}
                            <button
                                onClick={() => {
                                    const blob = new Blob([SAMPLE_CSV], { type: "text/csv" });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;
                                    a.download = "sample_patients.csv";
                                    a.click();
                                    URL.revokeObjectURL(url);
                                }}
                                className="text-xs text-blue-600 hover:text-blue-800 underline"
                            >
                                Download sample CSV template
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="rounded-lg bg-gray-50 border border-gray-200 p-4">
                                <p className="text-sm text-gray-600 mb-3">
                                    Connect directly to a SQL database containing patient records. The database should have a table matching this schema:
                                </p>
                                <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded-md overflow-x-auto font-mono whitespace-pre-wrap break-words">
{SAMPLE_SQL}
                                </pre>
                            </div>

                            {/* Connection string input (disabled) */}
                            <div className="opacity-50 pointer-events-none">
                                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                                    Connection String
                                </label>
                                <input
                                    type="text"
                                    placeholder="postgresql://user:pass@host:5432/dbname"
                                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                                    disabled
                                />
                                <div className="flex gap-2 mt-3">
                                    <button
                                        disabled
                                        className="px-4 py-2 text-sm font-medium bg-gray-200 text-gray-500 rounded-lg"
                                    >
                                        Test Connection
                                    </button>
                                    <button
                                        disabled
                                        className="px-4 py-2 text-sm font-medium bg-gray-200 text-gray-500 rounded-lg"
                                    >
                                        Import Patients
                                    </button>
                                </div>
                            </div>

                            <p className="text-xs text-gray-400 italic">
                                Direct database connections will be available in a future update. For now, export your data as CSV and use the Upload File tab.
                            </p>
                        </div>
                    )}
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
