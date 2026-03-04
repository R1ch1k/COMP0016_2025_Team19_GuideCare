"use client";

import { useState, useEffect } from "react";
import clsx from "clsx";
import { NICEGuideline, AnyGuideline } from "@/lib/types";

const BACKEND_HTTP_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function isNICEGuideline(g: AnyGuideline): g is NICEGuideline {
  return "rules" in g && "edges" in g;
}

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

function getGuidelineDisplayName(id: string | null): string {
  if (!id) return "Unknown guideline";
  const upper = id.toUpperCase();
  for (const [key, name] of Object.entries(GUIDELINE_NAMES)) {
    if (upper.includes(key)) return `${name}`;
  }
  return id;
}

function parsePathwayStep(step: string): { nodeId: string; decision: string; isAction: boolean } {
  const match = step.match(/^(n\d+)\((.+)\)$/);
  if (!match) return { nodeId: step, decision: "", isAction: false };
  return { nodeId: match[1], decision: match[2], isAction: match[2] === "action" };
}

export const PATIENT_STATUS_OPTIONS = [
  "Stable",
  "Needs Attention",
  "Active",
  "Critical",
] as const;

export type PatientStatus = (typeof PATIENT_STATUS_OPTIONS)[number];

export interface PatientRecord {
  id: string;
  backendId?: string;  // Backend UUID — used for WebSocket + API calls
  name: string;
  age: number;
  gender?: string;
  dateOfBirth?: string;  // ISO date string (YYYY-MM-DD)
  nhsNumber?: string;
  primaryConcern: string;
  status: PatientStatus;
  lastUpdated: string;
  clinician: string;
  notes?: string;
  conditions?: string[];
  medications?: Array<{ name: string; dose?: string }>;
  allergies?: string[];
  recentVitals?: Record<string, string | number>;
  clinicalNotes?: Array<{
    date: string;
    guideline: string;
    recommendation: string;
    urgency?: string;
  }>;
}

interface DiagnosisRecord {
  id: string;
  selected_guideline: string | null;
  final_recommendation: string | null;
  urgency: string | null;
  pathway_walked: string[];
  diagnosed_at: string | null;
}

interface PatientInfoPanelProps {
  records: PatientRecord[];
  onAddPatient: () => void;
  onConnect?: () => void;
  selectedPatientId?: string | null;
  onSelectPatient?: (patient: PatientRecord) => void;
  onUpdatePatient?: (patient: PatientRecord) => void;
  allGuidelines?: AnyGuideline[];
  className?: string;
}

const statusStyles: Record<PatientStatus, string> = {
  Stable:
    "bg-green-50 text-green-700 border border-green-200",
  "Needs Attention":
    "bg-yellow-50 text-yellow-700 border border-yellow-200",
  Active:
    "bg-blue-50 text-blue-700 border border-blue-200",
  Critical:
    "bg-red-50 text-red-700 border border-red-200",
};

const urgencyStyles: Record<string, string> = {
  urgent: "bg-red-100 text-red-800 border-red-300",
  high: "bg-orange-100 text-orange-800 border-orange-300",
  moderate: "bg-yellow-100 text-yellow-800 border-yellow-300",
  low: "bg-green-100 text-green-800 border-green-300",
};

function formatDate(iso: string | null): string {
  if (!iso) return "N/A";
  try {
    return new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export default function PatientInfoPanel({
  records,
  onAddPatient,
  onConnect,
  selectedPatientId,
  onSelectPatient,
  onUpdatePatient,
  allGuidelines,
  className,
}: PatientInfoPanelProps) {
  const [diagnoses, setDiagnoses] = useState<DiagnosisRecord[]>([]);
  const [loadingDiagnoses, setLoadingDiagnoses] = useState(false);
  const [expandedDiagId, setExpandedDiagId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editConditions, setEditConditions] = useState("");
  const [editMedications, setEditMedications] = useState("");
  const [editAllergies, setEditAllergies] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  const selectedPatient = records.find((r) => r.id === selectedPatientId);

  // Fetch diagnosis history when selected patient changes
  useEffect(() => {
    if (!selectedPatientId) {
      setDiagnoses([]);
      return;
    }
    let cancelled = false;
    async function fetchDiagnoses() {
      setLoadingDiagnoses(true);
      try {
        const res = await fetch(`${BACKEND_HTTP_URL}/patients/${selectedPatientId}/diagnoses`);
        if (res.ok && !cancelled) {
          setDiagnoses(await res.json());
        }
      } catch {
        // Silently fail — diagnoses panel just stays empty
      } finally {
        if (!cancelled) setLoadingDiagnoses(false);
      }
    }
    fetchDiagnoses();
    return () => { cancelled = true; };
  }, [selectedPatientId, records]); // re-fetch when records change (after diagnosis update)

  function startEditing() {
    if (!selectedPatient) return;
    setEditConditions((selectedPatient.conditions || []).join(", "));
    setEditMedications(
      (selectedPatient.medications || []).map((m) => (m.dose ? `${m.name} (${m.dose})` : m.name)).join(", ")
    );
    setEditAllergies((selectedPatient.allergies || []).join(", "));
    setIsEditing(true);
  }

  async function saveEdits() {
    if (!selectedPatient) return;
    setIsSaving(true);
    try {
      const conditions = editConditions.split(",").map((s) => s.trim()).filter(Boolean);
      const medications = editMedications.split(",").map((s) => {
        const m = s.trim();
        const match = m.match(/^(.+?)\s*\((.+?)\)$/);
        return match ? { name: match[1].trim(), dose: match[2].trim() } : { name: m, dose: "" };
      }).filter((m) => m.name);
      const allergies = editAllergies.split(",").map((s) => s.trim()).filter(Boolean);

      const patientId = selectedPatient.backendId || selectedPatient.id;
      const res = await fetch(`${BACKEND_HTTP_URL}/patients/${patientId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conditions, medications, allergies }),
      });

      if (res.ok) {
        const updated: PatientRecord = {
          ...selectedPatient,
          conditions,
          medications,
          allergies,
          primaryConcern: conditions[0] || selectedPatient.primaryConcern,
        };
        onUpdatePatient?.(updated);
        setIsEditing(false);
      }
    } catch {
      // Silently fail
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <aside className={clsx("flex flex-col", className)}>
      <div className="px-4 py-3 border-b border-gray-200 bg-white">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-gray-900">
              Patient Records
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Connected to electronic health record (optional integration).
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              onClick={onConnect}
              className="px-3 py-1.5 text-xs font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 transition-colors"
            >
              Connect
            </button>
            <button
              type="button"
              onClick={onAddPatient}
              className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 border border-blue-600 rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 transition-colors"
            >
              Add Patient
            </button>
          </div>
        </div>
      </div>

      {records.length === 0 ? (
        <div className="flex-1 flex items-center justify-center px-6 text-center text-sm text-gray-500">
          No patient records connected yet. Connect your EHR to surface patient
          summaries alongside the chat.
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {/* Patient Table */}
          <div className="overflow-hidden border border-gray-200 rounded-lg bg-white shadow-sm">
            <div className="overflow-x-auto">
              <table className="min-w-full table-fixed divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 w-[36%]">
                      Patient
                    </th>
                    <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 w-[10%]">
                      Age
                    </th>
                    <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 w-[24%]">
                      Primary Concern
                    </th>
                    <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 w-[18%]">
                      Status
                    </th>
                    <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-gray-500 w-[12%]">
                      Last Updated
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {records.map((record) => (
                    <tr
                      key={record.id}
                      onClick={() => onSelectPatient?.(record)}
                      className={clsx(
                        "transition-colors",
                        onSelectPatient && "cursor-pointer",
                        selectedPatientId === record.id
                          ? "bg-blue-100 border-l-4 border-l-blue-500"
                          : "hover:bg-blue-50/70"
                      )}
                    >
                      <td className="px-3 py-3 align-top">
                        <p className={clsx(
                          "text-xs font-semibold break-words",
                          selectedPatientId === record.id ? "text-blue-900" : "text-gray-900"
                        )}>
                          {record.name}
                        </p>
                        <p className="text-[11px] text-gray-500 mt-0.5 break-words">
                          {record.id} {record.clinician && `\u2022 ${record.clinician}`}
                        </p>
                        {record.notes && (
                          <p className="text-[11px] text-gray-600 mt-1 leading-relaxed break-words">
                            {record.notes}
                          </p>
                        )}
                      </td>
                      <td className="px-3 py-3 text-xs text-gray-700 align-top whitespace-nowrap">
                        {record.age}
                      </td>
                      <td className="px-3 py-3 text-xs text-gray-700 align-top break-words">
                        {record.primaryConcern}
                      </td>
                      <td className="px-3 py-3 align-top">
                        <span
                          className={clsx(
                            "inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-semibold",
                            statusStyles[record.status]
                          )}
                        >
                          {record.status}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-xs text-gray-700 align-top whitespace-nowrap">
                        {record.lastUpdated}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Patient Detail + Diagnosis History */}
          {selectedPatient && (
            <div className="border border-gray-200 rounded-lg bg-white shadow-sm overflow-hidden">
              {/* Patient Detail Header */}
              <div className="px-4 py-3 bg-blue-50 border-b border-blue-100 flex items-center justify-between">
                <h3 className="text-xs font-semibold text-blue-900">
                  {selectedPatient.name} — Clinical Summary
                </h3>
                {!isEditing && (
                  <button
                    onClick={startEditing}
                    className="p-1 text-blue-600 hover:text-blue-800 hover:bg-blue-100 rounded transition-colors"
                    title="Edit patient details"
                  >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                    </svg>
                  </button>
                )}
              </div>

              <div className="px-4 py-3 space-y-3">
                {isEditing ? (
                  <div className="space-y-3">
                    <div>
                      <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Conditions</label>
                      <input
                        type="text"
                        value={editConditions}
                        onChange={(e) => setEditConditions(e.target.value)}
                        placeholder="Type 2 Diabetes, Hypertension, ..."
                        className="w-full px-2.5 py-1.5 text-xs border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Medications</label>
                      <input
                        type="text"
                        value={editMedications}
                        onChange={(e) => setEditMedications(e.target.value)}
                        placeholder="Metformin (500mg), Amlodipine (5mg), ..."
                        className="w-full px-2.5 py-1.5 text-xs border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Allergies</label>
                      <input
                        type="text"
                        value={editAllergies}
                        onChange={(e) => setEditAllergies(e.target.value)}
                        placeholder="Penicillin, Aspirin, ..."
                        className="w-full px-2.5 py-1.5 text-xs border border-gray-300 rounded-md focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <p className="text-[10px] text-gray-400">Separate items with commas. For medications with doses, use format: Name (Dose)</p>
                    <div className="flex gap-2">
                      <button
                        onClick={saveEdits}
                        disabled={isSaving}
                        className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
                      >
                        {isSaving ? "Saving..." : "Save"}
                      </button>
                      <button
                        onClick={() => setIsEditing(false)}
                        className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Conditions */}
                    {selectedPatient.conditions && selectedPatient.conditions.length > 0 && (
                      <div>
                        <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Conditions</p>
                        <div className="flex flex-wrap gap-1.5">
                          {selectedPatient.conditions.map((c, i) => (
                            <span key={i} className="inline-flex px-2 py-0.5 text-[11px] font-medium bg-purple-50 text-purple-700 border border-purple-200 rounded-full">
                              {c}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Medications */}
                    {selectedPatient.medications && selectedPatient.medications.length > 0 && (
                      <div>
                        <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Medications</p>
                        <div className="flex flex-wrap gap-1.5">
                          {selectedPatient.medications.map((m, i) => (
                            <span key={i} className="inline-flex px-2 py-0.5 text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200 rounded-full">
                              {m.name}{m.dose ? ` (${m.dose})` : ""}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Allergies */}
                    {selectedPatient.allergies && selectedPatient.allergies.length > 0 && (
                      <div>
                        <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Allergies</p>
                        <div className="flex flex-wrap gap-1.5">
                          {selectedPatient.allergies.map((a, i) => (
                            <span key={i} className="inline-flex px-2 py-0.5 text-[11px] font-medium bg-red-50 text-red-700 border border-red-200 rounded-full">
                              {a}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* Diagnosis History */}
                <div>
                  <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-2">
                    Diagnosis History {diagnoses.length > 0 && `(${diagnoses.length})`}
                  </p>
                  {loadingDiagnoses ? (
                    <p className="text-xs text-gray-400">Loading...</p>
                  ) : diagnoses.length === 0 ? (
                    <p className="text-xs text-gray-400">No previous diagnoses</p>
                  ) : (
                    <div className="space-y-2">
                      {diagnoses.map((d) => (
                        <div
                          key={d.id}
                          className="border border-gray-200 rounded-md overflow-hidden"
                        >
                          <button
                            onClick={() => setExpandedDiagId(expandedDiagId === d.id ? null : d.id)}
                            className="w-full text-left px-3 py-2 flex items-center justify-between hover:bg-gray-50 transition-colors"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              {d.urgency && (
                                <span className={clsx(
                                  "inline-flex px-1.5 py-0.5 text-[10px] font-semibold rounded border",
                                  urgencyStyles[d.urgency] || "bg-gray-100 text-gray-700 border-gray-300"
                                )}>
                                  {d.urgency}
                                </span>
                              )}
                              <span className="text-[11px] font-medium text-gray-900 truncate">
                                {getGuidelineDisplayName(d.selected_guideline)}
                              </span>
                            </div>
                            <span className="text-[10px] text-gray-500 whitespace-nowrap ml-2">
                              {formatDate(d.diagnosed_at)}
                            </span>
                          </button>
                          {expandedDiagId === d.id && (() => {
                            // Find the matching guideline for node text lookup
                            const matchedGuideline = allGuidelines?.find(g => {
                              const gid = g.guideline_id.toLowerCase();
                              const sel = (d.selected_guideline || "").toLowerCase();
                              return gid === sel || gid.includes(sel) || sel.includes(gid);
                            });
                            const nodeTextMap: Record<string, string> = {};
                            if (matchedGuideline && isNICEGuideline(matchedGuideline)) {
                              for (const node of matchedGuideline.nodes) {
                                nodeTextMap[node.id] = node.text;
                              }
                            }

                            return (
                              <div className="px-3 py-2 border-t border-gray-100 bg-gray-50">
                                <p className="text-xs text-gray-700 leading-relaxed whitespace-pre-wrap">
                                  {d.final_recommendation || "No recommendation recorded"}
                                </p>
                                {d.pathway_walked.length > 0 && (
                                  <div className="mt-2 pt-2 border-t border-gray-200">
                                    <p className="text-[10px] font-semibold text-gray-500 mb-1.5">Decision pathway:</p>
                                    <div className="space-y-0.5">
                                      {d.pathway_walked.map((step, i) => {
                                        const parsed = parsePathwayStep(step);
                                        const isLast = i === d.pathway_walked.length - 1;
                                        const nodeText = nodeTextMap[parsed.nodeId] || "";

                                        let badgeColor = "bg-gray-100 text-gray-600";
                                        let dotColor = "bg-gray-400";

                                        if (parsed.isAction) {
                                          badgeColor = "bg-green-100 text-green-700";
                                          dotColor = "bg-green-500";
                                        } else if (parsed.decision === "yes") {
                                          badgeColor = "bg-blue-100 text-blue-700";
                                          dotColor = "bg-blue-500";
                                        } else if (parsed.decision === "no") {
                                          badgeColor = "bg-orange-100 text-orange-700";
                                          dotColor = "bg-orange-400";
                                        } else if (parsed.decision === "missing_variable") {
                                          badgeColor = "bg-amber-100 text-amber-700";
                                          dotColor = "bg-amber-500";
                                        }

                                        return (
                                          <div key={i} className="flex items-start gap-2">
                                            <div className="flex flex-col items-center w-2.5 flex-shrink-0 pt-1">
                                              <div className={`w-2 h-2 rounded-full ${dotColor} flex-shrink-0`} />
                                              {!isLast && <div className="w-px flex-1 min-h-[12px] bg-slate-300" />}
                                            </div>
                                            <div className="flex flex-col gap-0.5 pb-1.5 min-w-0">
                                              <div className="flex items-center gap-1.5">
                                                <span className={`px-1 py-0.5 rounded text-[9px] font-bold uppercase ${badgeColor}`}>
                                                  {parsed.isAction ? "action" : parsed.decision === "missing_variable" ? "needs data" : parsed.decision}
                                                </span>
                                              </div>
                                              {nodeText && (
                                                <span className="text-[11px] text-slate-700 leading-snug">{nodeText}</span>
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
                          })()}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
