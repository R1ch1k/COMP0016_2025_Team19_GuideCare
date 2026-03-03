"use client";

import { useState } from "react";
import {
  PATIENT_STATUS_OPTIONS,
  type PatientStatus,
  type PatientRecord,
} from "./PatientInfoPanel";

const BACKEND_HTTP_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface AddPatientModalProps {
  onClose: () => void;
  onSave: (patient: Omit<PatientRecord, "lastUpdated">) => void;
}

const defaultStatus: PatientStatus = "Needs Attention";

interface FormState {
  name: string;
  age: string;
  gender: string;
  dateOfBirth: string;
  nhsNumber: string;
  primaryConcern: string;
  status: PatientStatus;
  clinician: string;
  notes: string;
  conditions: string;
  allergies: string;
}

function estimateDOB(age: number): string {
  const today = new Date();
  const year = today.getFullYear() - age;
  return `${year}-01-01`;
}

const createInitialFormState = (): FormState => ({
  name: "",
  age: "",
  gender: "",
  dateOfBirth: "",
  nhsNumber: `NHS-${Math.floor(1000 + Math.random() * 9000)}`,
  primaryConcern: "",
  status: defaultStatus,
  clinician: "",
  notes: "",
  conditions: "",
  allergies: "",
});

export default function AddPatientModal({
  onClose,
  onSave,
}: AddPatientModalProps) {
  const [formState, setFormState] = useState<FormState>(createInitialFormState);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = event.target;
    setFormState((prev) => {
      if (name === "age") {
        return { ...prev, age: value.replace(/[^0-9]/g, "") };
      }
      if (name === "status") {
        return { ...prev, status: value as PatientStatus };
      }
      return { ...prev, [name]: value };
    });
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (!formState.name || !formState.primaryConcern || !formState.clinician) {
      return;
    }

    const age = Number(formState.age);
    if (!Number.isFinite(age) || age <= 0) {
      return;
    }

    // Split name into first/last
    const nameParts = formState.name.trim().split(/\s+/);
    const firstName = nameParts[0];
    const lastName = nameParts.slice(1).join(" ") || "Unknown";

    // Compute DOB from date field or estimate from age
    const dob = formState.dateOfBirth || estimateDOB(age);

    // Parse conditions and allergies from comma-separated strings
    const conditions = formState.conditions
      ? formState.conditions.split(",").map(s => s.trim()).filter(Boolean)
      : formState.primaryConcern ? [formState.primaryConcern] : [];
    const allergies = formState.allergies
      ? formState.allergies.split(",").map(s => s.trim()).filter(Boolean)
      : [];

    setSaving(true);

    try {
      // POST to backend to persist the patient
      const res = await fetch(`${BACKEND_HTTP_URL}/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nhs_number: formState.nhsNumber || `NHS-${Date.now()}`,
          first_name: firstName,
          last_name: lastName,
          date_of_birth: dob,
          gender: formState.gender || null,
          conditions,
          medications: [],
          allergies,
          recent_vitals: {},
          clinical_notes: formState.notes
            ? [{ note: formState.notes, date: new Date().toISOString().split("T")[0] }]
            : [],
        }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Backend error: ${detail}`);
      }

      const backendPatient = await res.json();

      onSave({
        id: backendPatient.id,  // Use backend UUID as the primary ID
        backendId: backendPatient.id,
        name: formState.name,
        age,
        gender: formState.gender || undefined,
        dateOfBirth: dob,
        nhsNumber: backendPatient.nhs_number,
        primaryConcern: formState.primaryConcern,
        status: formState.status,
        clinician: formState.clinician,
        notes: formState.notes.trim() || undefined,
        conditions,
        allergies,
      });

      setFormState(createInitialFormState());
      onClose();
    } catch (err) {
      console.error("Failed to create patient:", err);
      setError(err instanceof Error ? err.message : "Failed to save patient");
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setFormState(createInitialFormState());
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4">
      <div className="w-full max-w-lg rounded-xl bg-white shadow-xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-start justify-between border-b border-gray-200 px-6 py-4">
          <div>
            <h3 className="text-base font-semibold text-gray-900">Add Patient Record</h3>
            <p className="text-sm text-gray-500">
              Patient will be saved to the database for diagnosis.
            </p>
          </div>
          <button
            type="button"
            onClick={handleCancel}
            className="rounded-md p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100"
            aria-label="Close add patient form"
          >
            <svg
              className="h-5 w-5"
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

        <form onSubmit={handleSubmit} className="px-6 py-5 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-xs text-red-800">{error}</p>
            </div>
          )}

          <label className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
              Patient Name *
            </span>
            <input
              name="name"
              value={formState.name}
              onChange={handleChange}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Alex Morgan"
              required
            />
          </label>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Age *
              </span>
              <input
                name="age"
                value={formState.age}
                onChange={handleChange}
                inputMode="numeric"
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="45"
                required
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Gender
              </span>
              <select
                name="gender"
                value={formState.gender}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              >
                <option value="">Not specified</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Date of Birth
              </span>
              <input
                name="dateOfBirth"
                type="date"
                value={formState.dateOfBirth}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              />
            </label>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                NHS Number
              </span>
              <input
                name="nhsNumber"
                value={formState.nhsNumber}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="NHS-1234"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Primary Concern *
              </span>
              <input
                name="primaryConcern"
                value={formState.primaryConcern}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="Type 2 Diabetes"
                required
              />
            </label>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Conditions
              </span>
              <input
                name="conditions"
                value={formState.conditions}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="Diabetes, Hypertension"
              />
              <span className="text-[10px] text-gray-400">Comma-separated</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Allergies
              </span>
              <input
                name="allergies"
                value={formState.allergies}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="Penicillin, Sulfa"
              />
              <span className="text-[10px] text-gray-400">Comma-separated</span>
            </label>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Assigned Clinician *
              </span>
              <input
                name="clinician"
                value={formState.clinician}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="Dr. Priya Desai"
                required
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
                Status
              </span>
              <select
                name="status"
                value={formState.status}
                onChange={handleChange}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              >
                {PATIENT_STATUS_OPTIONS.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-600">
              Clinical Notes
            </span>
            <textarea
              name="notes"
              value={formState.notes}
              onChange={handleChange}
              rows={2}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Key considerations, follow-up plans..."
            />
          </label>

          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={handleCancel}
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? "Saving..." : "Save Patient"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
