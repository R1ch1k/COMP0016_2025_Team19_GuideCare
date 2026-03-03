"use client";

import clsx from "clsx";

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
}

interface PatientInfoPanelProps {
  records: PatientRecord[];
  onAddPatient: () => void;
  onConnect?: () => void;
  selectedPatientId?: string | null;
  onSelectPatient?: (patient: PatientRecord) => void;
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

export default function PatientInfoPanel({
  records,
  onAddPatient,
  onConnect,
  selectedPatientId,
  onSelectPatient,
  className,
}: PatientInfoPanelProps) {
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
        <div className="flex-1 overflow-y-auto px-4 py-4">
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
                          {record.id} • {record.clinician}
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
        </div>
      )}
    </aside>
  );
}
