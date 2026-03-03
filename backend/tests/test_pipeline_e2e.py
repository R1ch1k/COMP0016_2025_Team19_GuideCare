"""
End-to-end pipeline test for all 3 patients × 10 guidelines.

Tests each stage: triage, clarification, variable extraction,
graph traversal, and final recommendation formatting.

Run with:
    cd backend && python -m tests.test_pipeline_e2e
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the backend src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from app.guideline_engine import (
    format_recommendation_template,
    get_guideline,
    get_missing_variables_for_next_step,
    load_all_guidelines,
    traverse_guideline_graph,
    parse_bp,
    get_var_description,
)

# ──────────────────────────────────────────────────────────────────
# Test patients (mirrors seed.py)
# ──────────────────────────────────────────────────────────────────

PATIENTS = {
    "alex_morgan": {
        "id": "test-alex",
        "first_name": "Alex",
        "last_name": "Morgan",
        "age": 46,
        "gender": "male",
        "conditions": ["Type 2 diabetes", "Hypertension"],
        "medications": [{"name": "Metformin", "dose": "500mg"}],
        "allergies": ["Penicillin"],
        "recent_vitals": {"last_bp": "145/90", "last_bp_date": "2025-01-10"},
        "clinical_notes": [{"note": "A1C trending upward", "date": "2025-01-09"}],
    },
    "jordan_lee": {
        "id": "test-jordan",
        "first_name": "Jordan",
        "last_name": "Lee",
        "age": 63,
        "gender": "female",
        "conditions": ["Hypertension"],
        "medications": [{"name": "Amlodipine", "dose": "5mg"}],
        "allergies": [],
        "recent_vitals": {"last_bp": "160/100", "last_bp_date": "2025-01-08"},
        "clinical_notes": [{"note": "Medication review scheduled", "date": "2025-01-08"}],
    },
    "samantha_chen": {
        "id": "test-samantha",
        "first_name": "Samantha",
        "last_name": "Chen",
        "age": 32,
        "gender": "female",
        "conditions": ["Asthma"],
        "medications": [{"name": "Salbutamol", "dose": "100mcg"}],
        "allergies": ["Sulfa drugs"],
        "recent_vitals": {"last_bp": "120/78", "last_bp_date": "2025-01-02"},
        "clinical_notes": [{"note": "Follow-up in 6 months", "date": "2025-01-02"}],
    },
}

# ──────────────────────────────────────────────────────────────────
# Test scenarios per guideline
# Each scenario: opening message + simulated clarification answers
# ──────────────────────────────────────────────────────────────────

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "NG136": {
        "patient": "alex_morgan",
        "opening": "Alex has a clinic blood pressure of 155/95 today. He has diabetes and hypertension. Can you help assess his BP?",
        "answers": {
            "abpm": "Yes, ABPM was done and tolerated. Daytime average 150/90.",
            "symptoms": "No retinal haemorrhage, no papilloedema, no life-threatening symptoms.",
        },
        "expected_guideline": "NG136",
        "expected_urgency": "moderate",
        "must_contain": ["NG136"],
        "must_not_contain": [],
    },
    "NG136_jordan": {
        "patient": "jordan_lee",
        "opening": "Jordan has been on Amlodipine 5mg for 6 months but her blood pressure is still elevated at 160/100. What should we do next?",
        "answers": {
            "abpm": "Yes, ABPM confirmed average daytime reading of 155/95. She has been on Amlodipine 5mg for 6 months.",
        },
        "expected_guideline": "NG136",
        "expected_urgency": "moderate",
        "must_contain": ["NG136", "Step 2"],
        "must_not_contain": ["Step 1: Calcium channel blocker"],
    },
    "NG232": {
        "patient": "samantha_chen",
        "opening": "Samantha fell off her bike and hit her head. She's alert but has a mild headache. No vomiting, no loss of consciousness. GCS 15.",
        "answers": {},
        "expected_guideline": "NG232",
        "expected_urgency": "moderate",
        "must_contain": ["NG232"],
        "must_not_contain": [],
    },
    "NG84": {
        "patient": "samantha_chen",
        "opening": "Samantha has a sore throat with fever 38.5°C, purulent tonsils, and tender lymph nodes. FeverPAIN score is 4.",
        "answers": {},
        "expected_guideline": "NG84",
        "expected_urgency": "urgent",
        "must_contain": ["NG84"],
        "must_not_contain": [],
    },
    "NG91": {
        "patient": "samantha_chen",
        "opening": "Samantha's 2-year-old daughter has been pulling at her ear and has a fever of 39°C. She is irritable and not eating well.",
        "answers": {
            "ear": "Both ears are affected. No ear discharge. No signs of systemic illness.",
        },
        "expected_guideline": "NG91",
        "expected_urgency": "urgent",
        "must_contain": ["NG91"],
        "must_not_contain": [],
    },
    "NG112": {
        "patient": "samantha_chen",
        "opening": "Samantha has had 4 UTIs in the past year. She's currently experiencing dysuria and frequency. She is not pregnant.",
        "answers": {
            "uti": "This is her current episode. She is not perimenopausal.",
        },
        "expected_guideline": "NG112",
        "expected_urgency": "moderate",
        "must_contain": ["NG112"],
        "must_not_contain": [],
    },
    "NG133": {
        "patient": "samantha_chen",
        "opening": "A 28-week pregnant patient presents with BP 155/100 and proteinuria on dipstick. First pregnancy. No previous pre-eclampsia.",
        "answers": {
            "risk": "She has 1 high risk factor (first pregnancy + family history). BMI is 28.",
        },
        "expected_guideline": "NG133",
        "expected_urgency": "urgent",
        "must_contain": ["NG133"],
        "must_not_contain": [],
    },
    "NG184": {
        "patient": "samantha_chen",
        "opening": "Samantha was bitten by a cat on her hand. The wound is deep and puncture-like, with broken skin and bleeding. It happened 2 hours ago.",
        "answers": {},
        "expected_guideline": "NG184",
        "expected_urgency": "urgent",
        "must_contain": ["NG184"],
        "must_not_contain": [],
    },
    "NG222": {
        "patient": "samantha_chen",
        "opening": "A 35 year old completed CBT and antidepressant treatment for depression. Full remission achieved. 2 previous episodes. Wants to know about relapse prevention.",
        "answers": {},
        "expected_guideline": "NG222",
        "expected_urgency": "routine",
        "must_contain": ["NG222"],
        "must_not_contain": [],
    },
    "NG81_GLAUCOMA": {
        "patient": "jordan_lee",
        "opening": "Jordan has been diagnosed with chronic open-angle glaucoma. IOP is 28 mmHg. She has visual field defects. Newly diagnosed.",
        "answers": {},
        "expected_guideline": "NG81_GLAUCOMA",
        "expected_urgency": "moderate",
        "must_contain": ["NG81"],
        "must_not_contain": [],
    },
    "NG81_HYPERTENSION": {
        "patient": "jordan_lee",
        "opening": "Jordan has ocular hypertension. IOP is 26 mmHg with family history of glaucoma. Normal optic disc. No visual field defects.",
        "answers": {},
        "expected_guideline": "NG81_HYPERTENSION",
        "expected_urgency": "routine",
        "must_contain": ["NG81"],
        "must_not_contain": [],
    },
}

# ANSWER_BANK: simulated variable values per guideline (from notebook)
ANSWER_BANK = {
    "NG84": {
        "feverpain_score": 4,
        "centor_score": 3,
        "systemically_very_unwell": False,
        "high_risk_of_complications": False,
    },
    "NG232": {
        "age": 28,
        "head_injury_present": True,
        "gcs_score": 15,
        "loss_of_consciousness": False,
        "amnesia_since_injury": False,
        "seizure_present": False,
        "suspected_open_fracture": False,
        "basal_skull_fracture": False,
        "suspicion_c_spine_injury": False,
        "predisposing_condition": False,
        "low_risk_factors_present": True,
        "clotting_disorder_present": False,
        "dangerous_mechanism": False,
        "prolonged_amnesia": False,
    },
    "NG136": {
        "clinic_bp": "155/95",
        "abpm_tolerated": True,
        "abpm_daytime": "150/90",
        "retinal_haemorrhage": False,
        "papilloedema": False,
        "life_threatening_symptoms": False,
        "not_black_african_caribbean": True,
        "cardiovascular_disease": False,
        "diabetes": True,
        "renal_disease": False,
        "target_organ_damage": False,
        "qrisk_10yr": 15,
    },
    "NG222": {
        "treatment_completed": True,
        "remission_achieved": True,
        "higher_risk_of_relapse": True,
        "acute_treatment": "antidepressants",
    },
    "NG112": {
        "age": 32,
        "gender": "female",
        "recurrent_uti": True,
        "resistant_to_first_choice_antibiotics": False,
    },
    "NG133": {
        "gestational_age": 28,
        "clinic_bp": "155/100",
        "high_risk_factors_1_or_more": True,
        "moderate_risk_factors_2_or_more": False,
    },
    "NG184": {
        "signs_serious_illness_or_penetrating_wound": False,
        "wild_or_exotic_animal_bite": False,
        "bite_not_broken_skin": False,
        "human_bite_broken_skin_not_drawn_blood": False,
        "human_bite_broken_skin_drawn_blood": False,
        "cat_bite_broken_skin_not_drawn_blood": False,
        "cat_bite_broken_skin_drawn_blood": True,
        "prophylaxis_required": True,
        "penicillin_allergy": False,
    },
    "NG81_GLAUCOMA": {
        "newly_diagnosed_coag": True,
        "intraocular_pressure": 28,
        "iop": 28,
        "risk_of_visual_impairment": True,
        "slt_not_suitable": False,
    },
    "NG81_HYPERTENSION": {
        "intraocular_pressure": 26,
        "iop": 26,
        "risk_of_visual_impairment": True,
        "slt_not_suitable": False,
        "chooses_not_to_have_slt": True,
        "iop_not_reduced_with_pga": False,
    },
    "NG91": {
        "age": 2,
        "ear_pain": True,
        "fever": True,
        "otorrhoea": False,
        "infection_both_ears": True,
        "systemically_very_unwell": False,
    },
}


# ──────────────────────────────────────────────────────────────────
# Offline pipeline tests (no LLM, no DB — pure engine logic)
# ──────────────────────────────────────────────────────────────────


class PipelineResult:
    def __init__(self, scenario_id: str, guideline_id: str):
        self.scenario_id = scenario_id
        self.guideline_id = guideline_id
        self.patient: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.traversal_path: List[str] = []
        self.reached_actions: List[str] = []
        self.missing_variables: List[str] = []
        self.recommendation: str = ""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"[{status}] {self.scenario_id} ({self.guideline_id})"]
        if self.reached_actions:
            parts.append(f"  Actions: {len(self.reached_actions)}")
        if self.missing_variables:
            parts.append(f"  Missing vars: {self.missing_variables}")
        if self.recommendation:
            parts.append(f"  Recommendation: {self.recommendation[:120]}...")
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARNING: {w}")
        return "\n".join(parts)


def build_variables_for_scenario(
    scenario: Dict[str, Any],
    patient: Dict[str, Any],
    guideline_id: str,
) -> Dict[str, Any]:
    """Simulate variable extraction offline using patient data + ANSWER_BANK."""
    variables: Dict[str, Any] = {}

    # From patient record
    if patient.get("age"):
        variables["age"] = patient["age"]
    if patient.get("gender"):
        variables["gender"] = patient["gender"]
    if patient.get("recent_vitals", {}).get("last_bp"):
        variables["clinic_bp"] = patient["recent_vitals"]["last_bp"]

    # From patient conditions
    conds_lower = [c.lower() for c in (patient.get("conditions") or [])]
    variables["diabetes"] = any("diabetes" in c for c in conds_lower)
    variables["cardiovascular_disease"] = any(
        w in c for c in conds_lower for w in ("cardiovascular", "heart disease", "cvd")
    )
    variables["renal_disease"] = any(
        w in c for c in conds_lower for w in ("renal", "kidney", "ckd")
    )

    # QRISK estimation for hypertension patients
    age = variables.get("age", 0)
    has_hyp = any("hypertension" in c for c in conds_lower)
    if isinstance(age, (int, float)) and age >= 60 and has_hyp:
        variables["qrisk_10yr"] = 15
    elif isinstance(age, (int, float)) and age >= 40 and variables.get("diabetes") and has_hyp:
        variables["qrisk_10yr"] = 12

    # Default safe values for red flag / emergency booleans
    default_false = [
        "emergency_signs", "retinal_haemorrhage", "papilloedema",
        "life_threatening_symptoms", "target_organ_damage",
        "basal_skull_fracture", "suspected_open_fracture", "intubation_needed",
        "suspicion_non_accidental_injury", "suspected_cervical_spine_injury",
        "clotting_disorder_present",
        "systemically_very_unwell", "severe_systemic_infection_or_severe_complications",
        "signs_of_serious_illness_condition",
        "signs_serious_illness_or_penetrating_wound",
        "intrauterine_death", "placental_abruption",
        # NG81 glaucoma/ocular hypertension treatment-path flags
        "prescribed_eye_drops",
        "cannot_tolerate_pharmacological_treatment",
        "allergy_to_preservatives", "significant_ocular_surface_disease",
        "iop_not_reduced_sufficiently_with_current_treatment",
        "iop_not_reduced_sufficiently_with_treatments",
        "non_adherence_or_incorrect_technique",
        "iop_not_reduced_sufficiently",
        "additional_treatment_needed_to_reduce_iop",
        "chooses_not_to_have_slt", "chooses_no_slt",
        "needs_interim_treatment",
        "cannot_tolerate_current_treatment",
        "insufficient_iop_reduction_post_surgery",
        "reduced_effects_of_initial_slt",
        "waiting_for_slt_needs_treatment",
        "iop_not_reduced_with_pga",
        "advanced_coag_no_surgery",
        "needs_additional_iop_reduction",
        # NG91 otitis media
        "high_risk_complications", "serious_illness_condition",
    ]
    for var in default_false:
        variables.setdefault(var, False)

    # Default ethnicity inference
    variables.setdefault("not_black_african_caribbean", True)
    variables.setdefault("no_epilepsy_history", True)

    # Merge ANSWER_BANK values for this guideline
    bank_key = guideline_id
    if guideline_id == "NG136_jordan":
        bank_key = "NG136"
    bank = ANSWER_BANK.get(bank_key, {})
    for k, v in bank.items():
        variables.setdefault(k, v)

    # Medication-aware: detect if patient is already on BP treatment
    meds = patient.get("medications") or []
    bp_drugs = {
        "amlodipine", "nifedipine", "felodipine", "lercanidipine",
        "ramipril", "lisinopril", "enalapril", "perindopril",
        "losartan", "candesartan", "valsartan", "irbesartan",
        "bendroflumethiazide", "indapamide", "chlorthalidone",
        "spironolactone", "doxazosin", "bisoprolol", "atenolol",
    }
    on_bp_treatment = False
    for m in meds:
        med_name = (m.get("name", "") if isinstance(m, dict) else str(m)).lower()
        if any(drug in med_name for drug in bp_drugs):
            on_bp_treatment = True
            break

    if on_bp_treatment:
        bp_str = variables.get("clinic_bp", "")
        if bp_str:
            bp = parse_bp(bp_str)
            if bp:
                target_sys = 150 if (isinstance(age, (int, float)) and age >= 80) else 140
                if bp[0] >= target_sys or bp[1] >= 90:
                    variables["target_bp_achieved"] = False

    # Remove treatment-outcome vars that are False for new patients
    treatment_outcome_vars = {
        "treatment_response", "treatment_completed",
        "back_up_antibiotic_prescription_given", "immediate_antibiotic_prescription_given",
        "backup_antibiotic_given", "no_antibiotic_given", "immediate_antibiotic_not_given",
        "reassessment_needed_due_to_worsening_symptoms",
        "insufficient_iop_reduction_post_surgery", "reduced_effects_of_initial_slt",
    }
    if not on_bp_treatment:
        treatment_outcome_vars.add("target_bp_achieved")
    # NG222: only remove remission_achieved if False
    if not variables.get("remission_achieved"):
        treatment_outcome_vars.add("remission_achieved")

    for var in treatment_outcome_vars:
        if var in variables and not variables[var]:
            del variables[var]

    return variables


def run_scenario(scenario_id: str, scenario: Dict[str, Any]) -> PipelineResult:
    """Run a single scenario through the offline pipeline."""
    # Determine actual guideline_id (strip patient suffix like _jordan)
    guideline_id = scenario["expected_guideline"]
    patient = PATIENTS[scenario["patient"]]
    result = PipelineResult(scenario_id, guideline_id)
    result.patient = patient

    # Load guideline data
    g_data = get_guideline(guideline_id)
    if not g_data:
        result.errors.append(f"Guideline {guideline_id} not found in backend data")
        return result

    # Build variables
    variables = build_variables_for_scenario(scenario, patient, scenario_id)
    result.variables = variables

    # Traverse guideline graph
    nodes = g_data["guideline"]["nodes"]
    edges = g_data["guideline"]["edges"]
    evaluator = g_data["merged_evaluator"]

    traversal = traverse_guideline_graph(nodes, edges, evaluator, variables)
    result.traversal_path = [f"{p[0]}({p[2]})" for p in traversal["path"]]
    result.reached_actions = traversal["reached_actions"]
    result.missing_variables = traversal["missing_variables"]

    # Check for problems
    if not traversal["reached_actions"]:
        result.errors.append(
            f"No action nodes reached. Missing: {traversal['missing_variables']}. "
            f"Path: {' -> '.join(result.traversal_path)}"
        )
    elif traversal["missing_variables"]:
        result.warnings.append(
            f"Reached actions but still missing: {traversal['missing_variables']}"
        )

    # Format recommendation
    if traversal["reached_actions"]:
        scenario_text = f"{patient['age']} year old {patient['gender']}"
        if patient.get("conditions"):
            scenario_text += f" with {', '.join(patient['conditions'])}"

        recommendation = format_recommendation_template(
            guideline_id,
            scenario_text,
            traversal["reached_actions"],
            variables,
            medications=patient.get("medications"),
        )
        result.recommendation = recommendation

        # Validate recommendation
        for must in scenario.get("must_contain", []):
            if must.lower() not in recommendation.lower():
                result.errors.append(f"Recommendation missing expected text: '{must}'")

        for must_not in scenario.get("must_not_contain", []):
            if must_not.lower() in recommendation.lower():
                result.errors.append(f"Recommendation contains forbidden text: '{must_not}'")

    return result


# ──────────────────────────────────────────────────────────────────
# pytest-compatible parametrised tests
# ──────────────────────────────────────────────────────────────────

import pytest


SCENARIO_IDS = list(SCENARIOS.keys())


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_reaches_actions(scenario_id):
    """Each scenario must reach at least one action node."""
    result = run_scenario(scenario_id, SCENARIOS[scenario_id])
    assert result.reached_actions, (
        f"{scenario_id}: No action nodes reached. "
        f"Missing vars: {result.missing_variables}. "
        f"Path: {' -> '.join(result.traversal_path)}"
    )


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_recommendation_contains_expected(scenario_id):
    """Recommendation text must contain all must_contain strings."""
    scenario = SCENARIOS[scenario_id]
    result = run_scenario(scenario_id, scenario)
    for expected in scenario.get("must_contain", []):
        assert expected.lower() in result.recommendation.lower(), (
            f"{scenario_id}: Recommendation missing '{expected}'. "
            f"Got: {result.recommendation[:200]}"
        )


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_recommendation_excludes_forbidden(scenario_id):
    """Recommendation text must not contain any must_not_contain strings."""
    scenario = SCENARIOS[scenario_id]
    result = run_scenario(scenario_id, scenario)
    for forbidden in scenario.get("must_not_contain", []):
        assert forbidden.lower() not in result.recommendation.lower(), (
            f"{scenario_id}: Recommendation contains forbidden '{forbidden}'. "
            f"Got: {result.recommendation[:200]}"
        )


def test_all_guidelines_loaded():
    """Verify all 10 guidelines load successfully."""
    all_data = load_all_guidelines()
    assert len(all_data) >= 10, f"Expected >=10 guidelines, got {len(all_data)}"


def test_no_double_periods_in_recommendations():
    """No recommendation should contain '..' (double period)."""
    for scenario_id, scenario in SCENARIOS.items():
        result = run_scenario(scenario_id, scenario)
        if result.recommendation:
            assert ".." not in result.recommendation, (
                f"{scenario_id}: Double period in recommendation: "
                f"{result.recommendation[:200]}"
            )


# ──────────────────────────────────────────────────────────────────
# Standalone fallback — `python -m tests.test_pipeline_e2e`
# ──────────────────────────────────────────────────────────────────


def main():
    """Run all scenarios and print results (standalone mode)."""
    all_data = load_all_guidelines()
    print(f"Loaded {len(all_data)} guidelines: {', '.join(sorted(all_data.keys()))}")

    passes = failures = 0
    for scenario_id, scenario in SCENARIOS.items():
        r = run_scenario(scenario_id, scenario)
        if r.passed:
            passes += 1
        else:
            failures += 1
        print(r.summary())

    print(f"\n{passes} passed, {failures} failed out of {passes + failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
