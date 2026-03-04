"""
Orchestration dependencies — real implementations using the guideline engine
and OpenAI API.

Each callable is injected into the LangGraph nodes via build_orchestration_deps().
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict
from uuid import UUID

from sqlalchemy import select

from app.core.config import settings
from app.db.models import Patient
from app.db.session import AsyncSessionLocal
from app.guideline_engine import (
    VAR_DESCRIPTIONS,
    build_patient_record,
    extract_best_question,
    extract_json_from_text,
    fix_variable_extraction,
    fix_variable_extraction_v2,
    get_var_description,
    format_recommendation_template,
    get_guideline,
    get_all_variables_from_evaluator,
    get_missing_variables_for_next_step,
    load_all_guidelines,
    traverse_guideline_graph,
)
from app.llm import generate, generate_api_only

logger = logging.getLogger(__name__)


class OrchestrationDeps(TypedDict):
    fetch_patient: Callable[..., Any]
    triage_agent: Callable[..., Any]
    gpt_clarifier: Callable[..., Any]
    select_guideline: Callable[..., Any]
    extract_variables_20b: Callable[..., Any]
    walk_guideline_graph: Callable[..., Any]
    format_output_20b: Callable[..., Any]


# ===================================================================
# 1. Fetch patient from DB
# ===================================================================


async def fetch_patient(patient_id: str) -> Dict[str, Any]:
    """Load a patient record from DB as a plain dict."""
    pid = UUID(patient_id)
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Patient).where(Patient.id == pid))
        p = result.scalars().first()
        if not p:
            return {"id": patient_id}

        return {
            "id": str(p.id),
            "nhs_number": p.nhs_number,
            "first_name": p.first_name,
            "last_name": p.last_name,
            "age": p.age,
            "gender": p.gender,
            "conditions": p.conditions or [],
            "medications": p.medications or [],
            "allergies": p.allergies or [],
            "recent_vitals": p.recent_vitals or {},
            "clinical_notes": p.clinical_notes or [],
        }


# ===================================================================
# 2. Triage agent — GPT-4 API always (from medical_chatbot_triage_testing.ipynb)
# ===================================================================

TRIAGE_SYSTEM_PROMPT = """You are a medical triage assistant. Assess the urgency and suggest which NICE guideline applies.

Patient symptoms: "{symptoms}"
Patient age: {age}
Medical history: {medical_history}
Current medications: {medications}

Available NICE guidelines:
1. NG232 - Head injury: assessment and early management
2. NG136 - Hypertension in adults: diagnosis and management
3. NG91 - Otitis Media (Acute): Antimicrobial Prescribing
4. NG133 - Hypertension in pregnancy: diagnosis and management
5. NG112 - Urinary tract infection (recurrent): antimicrobial prescribing
6. NG184 - Antimicrobial prescribing for human and animal bites
7. NG222 - Depression in adults: preventing relapse
8. NG81_GLAUCOMA - Chronic Open Angle Glaucoma Management
9. NG81_HYPERTENSION - Management of Ocular Hypertension and Glaucoma
10. NG84 - Sore Throat (Acute): Antimicrobial Prescribing

---
URGENCY ASSESSMENT ALGORITHM (FOLLOW IN ORDER):
---

STEP 1: Check for EMERGENCY RED FLAGS (if ANY present -> Emergency):
- Neurological: Loss of consciousness, confusion, altered mental state, sudden vision loss
- Cardiovascular: Severe chest pain, BP >=180/120 WITH any symptoms (headache/visual disturbance/chest pain)
- Respiratory: Shortness of breath, airway compromise (drooling, stridor, unable to swallow)
- Infection: Signs of sepsis (fever + confusion + tachypnea), mastoiditis (swelling behind ear causing protrusion)
- Renal: Pyelonephritis (fever + flank/back pain + chills/rigors)
- Ophthalmologic: Acute angle-closure glaucoma (sudden severe eye pain + vision loss +/- nausea)
- Obstetric: Pregnant with BP >=160/110 OR severe headache + visual disturbance + swelling
- Psychiatric: Suicidal ideation with plan/intent, severe self-harm risk
- Trauma: Uncontrollable bleeding, severe injury

STEP 2: If no red flags, check for URGENT criteria (same-day assessment):
- Fever >38.5C with moderate pain/symptoms
- Moderate infection signs without sepsis
- BP 160-179/100-119 WITH mild symptoms
- Pregnant with BP 150-159/100-109 OR BP 140-149/90-99 WITH proteinuria
- Significant pain affecting function
- Acute worsening of chronic condition
- Recent injury with concerning features
- Suspected bacterial infection needing antibiotics (strep throat, otitis media with fever)

STEP 3: If not urgent, check for MODERATE criteria (1-3 day assessment):
- Mild-moderate symptoms, stable condition
- Low-grade fever (<38.5C) with mild symptoms
- BP 140-159/90-99 WITHOUT symptoms
- Pregnant with BP 140-149/90-99 WITHOUT proteinuria or symptoms
- Manageable pain not affecting daily function
- Stable chronic condition with minor change
- Non-infected wound needing assessment

STEP 4: Default to ROUTINE (routine GP appointment):
- Very mild symptoms
- Monitoring of stable chronic condition
- Preventive care
- Medication review for controlled condition
- No concerning features

---
SPECIFIC CLINICAL CRITERIA:
---

**HYPERTENSION (NG136, NG133):**
Emergency: BP >=180/120 WITH symptoms OR BP >=200/130 regardless
Urgent: BP 160-179/100-119 WITH symptoms
Moderate: BP 140-159/90-99 WITHOUT symptoms
Routine: Controlled BP, regular monitoring

**PREGNANCY HYPERTENSION (NG133):**
Emergency: BP >=160/110 OR BP >=140/90 WITH severe headache + visual disturbance + swelling
Urgent: BP 150-159/100-109 OR BP 140-149/90-99 WITH proteinuria
Moderate: BP 140-149/90-99 WITHOUT proteinuria or symptoms
Routine: BP <140/90

**URINARY TRACT INFECTION (NG112):**
Emergency: Pyelonephritis (fever + flank pain + chills), sepsis signs
Urgent: Recurrent UTI with fever >38C
Moderate: Recurrent UTI without fever
Routine: Mild dysuria, no fever

**OTITIS MEDIA (NG91):**
Emergency: Mastoiditis, meningitis signs
Urgent: Severe ear pain with fever >38.5C
Moderate: Moderate ear pain, fever <38.5C
Routine: Mild ear discomfort, no fever

**SORE THROAT (NG84):**
Emergency: Airway compromise (unable to swallow, drooling, stridor)
Urgent: Severe throat pain with high fever >38.5C
Moderate: Moderate throat pain with fever, white patches
Routine: Mild sore throat, no fever

**EYE CONDITIONS:**
Use NG81_GLAUCOMA: diagnosed glaucoma, vision loss, optic nerve damage, acute angle-closure
Use NG81_HYPERTENSION: elevated IOP without nerve damage, ocular hypertension, risk assessment

**DEPRESSION (NG222):**
Emergency: Active suicidal ideation with plan/intent
Urgent: Relapse with significant functional impairment
Moderate: Mild relapse symptoms
Routine: Stable, routine review

**HEAD INJURY (NG232):**
Emergency: LOC >5min, vomiting >=2, confusion, amnesia, seizure
Urgent: LOC <5min, persistent headache + dizziness
Moderate: Mild headache, no neurological signs
Routine: Very minor bump, no symptoms

**BITES (NG184):**
Emergency: Uncontrollable bleeding, deep bite with infection signs
Urgent: Cat/dog bite that broke skin, moderate swelling
Moderate: Puncture wound without severe features
Routine: Superficial scratch

---
OUTPUT FORMAT (STRICT JSON):
---
{
  "urgency": "emergency|urgent|moderate|routine",
  "reasoning": "Brief clinical reasoning citing specific criteria used",
  "suggested_guideline": "EXACT_ID (e.g., NG136, NG81_GLAUCOMA)",
  "guideline_confidence": "high|medium|low",
  "red_flags": ["specific red flag 1", "specific red flag 2"],
  "assessment": "One sentence clinical assessment summary"
}

CRITICAL: Follow the 4-step urgency algorithm in order. Return ONLY valid JSON."""


def _format_meds(medications: list) -> str:
    """Format medications list (may be strings or dicts with name/dose)."""
    parts = []
    for m in (medications or []):
        if isinstance(m, dict):
            name = m.get("name", "")
            dose = m.get("dose", "")
            parts.append(f"{name} {dose}".strip() if name else str(m))
        else:
            parts.append(str(m))
    return ", ".join(parts) or "None"


def _format_triage_prompt(
    symptoms: str, patient_record: dict
) -> str:
    """Build patient-specific triage prompt."""
    age = patient_record.get("age", "N/A")
    history = ", ".join(patient_record.get("conditions", []) or patient_record.get("medical_history", [])) or "None"
    meds = _format_meds(patient_record.get("medications", []))
    return (
        f'Patient symptoms: "{symptoms}"\n'
        f"Patient age: {age}\n"
        f"Medical history: {history}\n"
        f"Current medications: {meds}"
    )


async def triage_agent(
    symptoms: str, history: list, patient_record: dict
) -> Dict[str, Any]:
    """LLM-based triage — ALWAYS uses GPT-4 API regardless of LLM_MODE.

    Ported from medical_chatbot_triage_testing.ipynb.
    Returns urgency (emergency/urgent/moderate/routine), suggested guideline,
    reasoning, red flags, and assessment.
    """
    # External API override if configured
    if settings.TRIAGE_API_URL:
        import httpx

        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            r = await client.post(
                settings.TRIAGE_API_URL.rstrip("/") + "/triage",
                json={
                    "symptoms": symptoms,
                    "history": history,
                    "patient": patient_record,
                },
            )
            r.raise_for_status()
            return r.json()

    # Build system prompt with patient details filled in
    age = patient_record.get("age", "N/A")
    med_history = ", ".join(
        patient_record.get("conditions", []) or patient_record.get("medical_history", [])
    ) or "None"
    meds = _format_meds(patient_record.get("medications", []))

    system_prompt = TRIAGE_SYSTEM_PROMPT.replace("{symptoms}", symptoms or "")
    system_prompt = system_prompt.replace("{age}", str(age))
    system_prompt = system_prompt.replace("{medical_history}", med_history)
    system_prompt = system_prompt.replace("{medications}", meds)

    user_prompt = _format_triage_prompt(symptoms, patient_record)

    try:
        raw = await generate_api_only(
            user_prompt,
            max_tokens=400,
            temperature=0.0,
            system_message=system_prompt,
        )

        # Strip markdown code fences if present
        content = raw.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]  # remove ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)

        # Normalise urgency to lowercase for graph.py
        if "urgency" in result:
            result["urgency"] = result["urgency"].lower()

        # Ensure assessment field exists
        if "assessment" not in result:
            result["assessment"] = result.get("reasoning", "Triage complete")

        logger.info(
            "Triage result: urgency=%s, guideline=%s, red_flags=%s",
            result.get("urgency"),
            result.get("suggested_guideline"),
            result.get("red_flags"),
        )
        return result

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("GPT-4 triage failed (%s), falling back to heuristics", exc)

        # Keyword heuristic fallback
        s = (symptoms or "").lower()
        urgent_words = [
            "chest pain", "shortness of breath", "collapse", "seizure",
            "stroke", "unconscious", "severe bleeding", "heart attack",
            "suicidal", "loss of consciousness", "drooling", "stridor",
        ]
        if any(w in s for w in urgent_words):
            return {
                "urgency": "emergency",
                "assessment": "Potentially urgent symptoms detected (heuristic fallback)",
                "suggested_guideline": "",
                "red_flags": [w for w in urgent_words if w in s],
            }
        return {
            "urgency": "moderate",
            "assessment": "Initial non-urgent triage (heuristic fallback)",
            "suggested_guideline": "",
            "red_flags": [],
        }


# ===================================================================
# 3. GPT Clarifier — LLM-driven clarification using guideline engine
# ===================================================================


async def gpt_clarifier(
    symptoms: str,
    history: list,
    patient_record: dict,
    triage: dict,
    answers: dict,
    selected_guideline: str = "",
) -> Dict[str, Any]:
    """Generate clarification questions driven by the guideline evaluator.

    Uses the guideline engine to find missing variables, extracts what we can
    from symptoms text + patient record + previous answers using regex helpers,
    then only asks about truly missing variables.

    Called on every clarification round — incorporates previous answers to
    check if more variables are still needed (e.g. ABPM tolerated → now need
    the actual ABPM reading).
    """
    # Use the already-selected guideline (set before clarify in the pipeline),
    # then triage suggestion, then keyword guess as fallback
    guideline_id = selected_guideline or (triage or {}).get("suggested_guideline", "")
    if not guideline_id or not get_guideline(guideline_id):
        s = (symptoms or "").lower()
        guideline_id = _guess_guideline(s)

    if not guideline_id:
        return {"done": True, "questions": []}

    g_data = get_guideline(guideline_id)
    if not g_data:
        return {"done": True, "questions": []}

    # Build known variables from patient record
    known_vars: Dict[str, Any] = {}
    if patient_record.get("age"):
        known_vars["age"] = patient_record["age"]
    if patient_record.get("gender"):
        known_vars["gender"] = patient_record["gender"]

    # Extract variables from symptom text using regex helpers (from notebook)
    symptom_text = symptoms or ""
    extracted = fix_variable_extraction({}, symptom_text)
    extracted = fix_variable_extraction_v2(extracted, symptom_text)

    # Extract comorbidities from patient record (source of truth)
    conditions = patient_record.get("conditions", [])
    conds_lower = [c.lower() for c in conditions]
    if "diabetes" in known_vars or "diabetes" in (g_data.get("all_vars") or set()):
        extracted["diabetes"] = any("diabetes" in c for c in conds_lower)
    if "cardiovascular_disease" in known_vars or "cardiovascular_disease" in (g_data.get("all_vars") or set()):
        extracted["cardiovascular_disease"] = any(w in c for c in conds_lower for w in ("cardiovascular", "heart disease", "cvd", "coronary"))
    if "renal_disease" in known_vars or "renal_disease" in (g_data.get("all_vars") or set()):
        extracted["renal_disease"] = any(w in c for c in conds_lower for w in ("renal", "kidney", "ckd"))
    if any("hypertension" in c for c in conds_lower):
        extracted.setdefault("hypertension_history", True)

    # Extract BP from recent vitals
    vitals = patient_record.get("recent_vitals", {})
    if vitals.get("last_bp") and "clinic_bp" not in extracted:
        extracted["clinic_bp"] = vitals["last_bp"]
        extracted["bp"] = vitals["last_bp"]

    # Incorporate clarification answers into extracted variables.
    # Questions are tagged with "[var:variable_name]" so we know exactly
    # which variable each answer maps to — no fragile keyword matching.
    import re as _re
    for q, a in (answers or {}).items():
        a_str = str(a) if a else ""
        a_lower = a_str.lower()

        # Extract target variable from tag (e.g. "[var:abpm_tolerated] Is ABPM tolerated?")
        var_tag_match = _re.match(r"^\[var:(\w+)\]\s*", q)
        target_var = var_tag_match.group(1) if var_tag_match else None
        q_text = q[var_tag_match.end():] if var_tag_match else q
        q_lower = q_text.lower()

        # Detect "not known" / "unknown" — skip these entirely.
        # "I don't know if they have diabetes" ≠ "they don't have diabetes".
        _unknown_phrases = ("not known", "unknown", "unsure", "don't know", "no idea",
                            "not sure", "not available", "n/a", "can't say", "not recorded")
        is_unknown = any(phrase in a_lower for phrase in _unknown_phrases)

        # Parse the answer value based on variable type
        bp_match = _re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", a_str)
        num_match = _re.search(r"(\d+\.?\d*)", a_str)
        is_yes = any(w in a_lower for w in ("yes", "true", "confirmed", "positive", "present", "done", "tolerated", "completed", "accepted"))
        is_no = any(w in a_lower for w in ("no", "false", "negative", "denied", "absent", "none", "declined", "refused"))

        # If the answer is "unknown"/"not known", skip — don't set the variable
        # at all. The tree will handle the missing variable via its default path.
        # Exception: if the answer also contains a concrete value (BP reading, number),
        # still extract that value.
        if is_unknown and not bp_match and not num_match:
            continue

        if target_var:
            # We know exactly which variable this answer is for
            _bp_vars = {"clinic_bp", "abpm_daytime", "hbpm_average", "bp"}
            _bool_vars_from_evaluator = set()  # Will be inferred from answer type

            if target_var in _bp_vars and bp_match:
                extracted[target_var] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif target_var == "qrisk_10yr":
                # Check qualifier phrases BEFORE raw numbers — "less than 10%"
                # contains "10" which the regex would incorrectly match as 10.0
                if any(w in a_lower for w in ("less than 10", "below 10", "under 10")):
                    extracted["qrisk_10yr"] = 5
                elif any(w in a_lower for w in ("greater than 10", "above 10", "over 10")):
                    extracted["qrisk_10yr"] = 15
                elif a_lower.strip() in ("low", "low risk"):
                    extracted["qrisk_10yr"] = 5
                elif a_lower.strip() in ("high", "high risk"):
                    extracted["qrisk_10yr"] = 15
                else:
                    qrisk_num = _re.search(r"(\d+\.?\d*)\s*%?", a_str)
                    if qrisk_num:
                        extracted["qrisk_10yr"] = float(qrisk_num.group(1))
            elif target_var == "not_black_african_caribbean":
                # Negated variable: "Is patient of African/Caribbean origin?"
                # yes → not_black_african_caribbean = False
                # no → not_black_african_caribbean = True
                if is_yes:
                    extracted["not_black_african_caribbean"] = False
                elif is_no:
                    extracted["not_black_african_caribbean"] = True
            elif target_var == "abpm_tolerated":
                if is_yes:
                    extracted["abpm_tolerated"] = True
                elif is_no:
                    extracted["abpm_tolerated"] = False
                # Also extract BP reading if included in the answer
                if bp_match:
                    extracted["abpm_daytime"] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif target_var == "target_bp_achieved":
                # "Is BP at target?" — yes/no boolean
                # But also accept a BP reading as evidence BP is NOT at target
                if bp_match:
                    reading = f"{bp_match.group(1)}/{bp_match.group(2)}"
                    extracted["clinic_bp"] = reading
                    # If they gave a reading, BP is likely not at target
                    systolic = int(bp_match.group(1))
                    extracted["target_bp_achieved"] = systolic < 140
                elif is_yes:
                    extracted["target_bp_achieved"] = True
                elif is_no:
                    extracted["target_bp_achieved"] = False
            elif target_var in ("temperature", "fever"):
                if num_match:
                    val = float(num_match.group(1))
                    extracted["temperature"] = val
                    extracted["fever"] = val >= 38.0
                elif is_yes:
                    extracted["fever"] = True
                elif is_no:
                    extracted["fever"] = False
            elif target_var in ("gcs_score",) and num_match:
                extracted["gcs_score"] = int(float(num_match.group(1)))
            elif target_var in ("iop", "intraocular_pressure") and num_match:
                val = int(float(num_match.group(1)))
                extracted["iop"] = val
                extracted["intraocular_pressure"] = val
            elif target_var in ("duration", "vomiting_count", "gestational_age") and num_match:
                extracted[target_var] = int(float(num_match.group(1)))
            elif target_var in ("centor_score", "feverpain_score") and num_match:
                extracted[target_var] = int(float(num_match.group(1)))
            elif bp_match:
                # Variable expects a BP reading
                extracted[target_var] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif is_yes or is_no:
                # Generic boolean — handle negated variable names
                if target_var.startswith("not_") or target_var.startswith("no_"):
                    extracted[target_var] = is_no
                else:
                    extracted[target_var] = is_yes
            elif num_match:
                # Generic numeric
                extracted[target_var] = float(num_match.group(1))
            else:
                # Free-text answer (e.g. "not known") — store as string
                extracted[target_var] = a_str
        else:
            # Legacy: untagged questions — fall back to keyword matching
            if bp_match and ("bp" in q_lower or "blood pressure" in q_lower):
                reading = f"{bp_match.group(1)}/{bp_match.group(2)}"
                if "abpm" in q_lower or "ambulatory" in q_lower:
                    extracted["abpm_daytime"] = reading
                elif "home" in q_lower or "hbpm" in q_lower:
                    extracted["hbpm_average"] = reading
                else:
                    extracted["clinic_bp"] = reading
            elif is_yes or is_no:
                for var_name in list(VAR_DESCRIPTIONS.keys()):
                    var_words = var_name.lower().replace("_", " ").split()
                    if any(w in q_lower for w in var_words if len(w) > 3):
                        if var_name.startswith("not_") or var_name.startswith("no_"):
                            extracted[var_name] = is_no
                        else:
                            extracted[var_name] = is_yes
                        break

    # Merge extracted into known_vars (extracted values override)
    for k, v in extracted.items():
        if v is not None:
            known_vars[k] = v

    # Infer target_bp_achieved from clinic_bp if we have a reading but
    # haven't determined if BP is at target yet. This avoids asking the
    # clinician "Is BP at target?" when we already know it's 160/100.
    if "target_bp_achieved" not in known_vars and "clinic_bp" in known_vars:
        from app.guideline_engine import parse_bp
        bp = parse_bp(known_vars["clinic_bp"])
        if bp:
            age = known_vars.get("age", 0)
            target_sys = 150 if (isinstance(age, (int, float)) and age >= 80) else 140
            known_vars["target_bp_achieved"] = bp[0] < target_sys and bp[1] < 90

    # Check what's still missing to advance to the next step in the tree.
    # This only asks about variables needed at the current decision point,
    # avoiding useless questions about branches the patient won't reach.
    nodes = g_data["guideline"]["nodes"]
    edges = g_data["guideline"]["edges"]
    evaluator = g_data["merged_evaluator"]

    missing = get_missing_variables_for_next_step(nodes, edges, evaluator, known_vars)

    # Filter out variables that shouldn't be asked via clarification:
    # - Comorbidities should come from the patient record, not be asked
    # - Treatment-outcome vars shouldn't be asked for patients not on treatment
    _record_only_vars = {"diabetes", "cardiovascular_disease", "renal_disease",
                         "target_organ_damage"}
    _treatment_vars = {"target_bp_achieved", "treatment_response", "treatment_completed"}
    patient_meds = patient_record.get("medications") or []
    on_treatment = len(patient_meds) > 0

    filtered_missing = []
    for var in missing:
        if var in _record_only_vars:
            # Default to False — these come from structured patient data
            known_vars[var] = False
            continue
        if var in _treatment_vars and not on_treatment:
            # Skip treatment-outcome questions for patients not on treatment
            continue
        filtered_missing.append(var)
    missing = filtered_missing

    if not missing:
        return {"done": True, "questions": []}

    # Variable descriptions for better questions
    var_desc = {
        "clinic_bp": "the patient's most recent clinic blood pressure reading (e.g. 155/95)",
        "age": "the patient's age",
        "gender": "the patient's gender",
        "abpm_tolerated": "whether the patient has had or can tolerate ambulatory blood pressure monitoring (ABPM)",
        "abpm_daytime": "the patient's ABPM daytime average blood pressure reading (e.g. 150/95)",
        "hbpm_average": "the patient's home blood pressure monitoring (HBPM) average reading (e.g. 145/90)",
        "not_black_african_caribbean": "whether the patient is of black African or African-Caribbean family origin (answer yes/no)",
        "fever": "whether the patient has a fever and if so what temperature",
        "duration": "how long the symptoms have been present (in days)",
        "vomiting_count": "how many times the patient has vomited",
        "gcs_score": "the patient's Glasgow Coma Scale score",
        "loss_of_consciousness": "whether the patient lost consciousness",
        "head_injury_present": "whether there was a head injury",
        "recurrent_uti": "whether this is a recurrent urinary tract infection",
        "gestational_age": "how many weeks pregnant the patient is",
        "bite_type": "what type of animal caused the bite (cat, dog, human, etc.)",
        "broken_skin": "whether the skin is broken",
        "ear_pain": "whether the patient has ear pain",
        "centor_score": "the FeverPAIN or Centor score components (fever, tonsillar exudate, tender lymph nodes, absence of cough)",
        "iop": "the patient's intraocular pressure reading",
        "treatment_completed": "whether previous treatment has been completed",
        "acute_treatment": "what acute treatment the patient received",
        "emergency_signs": "whether there are any emergency/red flag signs",
        "newly_diagnosed": "whether this is a new diagnosis",
        "cardiovascular_disease": "whether the patient has cardiovascular disease",
        "renal_disease": "whether the patient has renal/kidney disease",
        "diabetes": "whether the patient has diabetes",
        "target_organ_damage": "whether there is evidence of target organ damage",
        "target_bp_achieved": "whether the patient's blood pressure is at target on current treatment",
    }

    # Generate questions for missing variables.
    # Each question is tagged with the target variable name using a
    # "[var:variable_name]" prefix so the answer can be mapped back
    # to the correct variable without fragile keyword matching.
    questions: List[str] = []
    for target_var in missing:
        desc = var_desc.get(target_var, target_var)
        try:
            prompt = f"""You are a medical assistant collecting clinical information from a doctor for NICE guideline {guideline_id}.

Patient: {patient_record.get('age', 'unknown')} year old {patient_record.get('gender', 'patient')}
Symptoms: {symptoms}
What we already know: {json.dumps({k: v for k, v in known_vars.items() if v is not None}, default=str)}

We need to determine: {desc}

Generate ONE specific, direct clinical question to ask. Do NOT ask about scheduling appointments or tests.
Ask directly for the clinical value or finding we need. Be concise (one sentence).
Example good questions:
- "What is the patient's clinic blood pressure reading?"
- "Has the patient experienced any loss of consciousness?"
- "How many days has the patient had these symptoms?"
"""

            raw = await generate(prompt, max_tokens=100, temperature=0.1)
            question = extract_best_question(raw)
            # Tag with target variable so answer parsing knows what to set
            questions.append(f"[var:{target_var}] {question}")
        except Exception as e:
            logger.warning("Failed to generate clarification question: %s", e)
            questions.append(f"[var:{target_var}] What is the patient's {desc}?")

    return {"done": False, "questions": questions}


# ===================================================================
# 4. Select guideline — keyword mapping + optional LLM
# ===================================================================


def _guess_guideline(symptoms_lower: str) -> Optional[str]:
    """Map symptom keywords to NICE guideline IDs."""
    mapping = [
        (["sore throat", "throat pain", "tonsil"], "NG84"),
        (["head injury", "hit head", "head trauma", "fell", "concussion"], "NG232"),
        (["blood pressure", "hypertension", "bp reading", "bp "], "NG136"),
        (["depression", "antidepressant", "low mood"], "NG222"),
        (["uti", "urinary tract", "urinary infection", "dysuria"], "NG112"),
        (["pregnant", "pregnancy", "gestational", "pre-eclampsia"], "NG133"),
        (["bite", "cat bite", "dog bite", "animal bite"], "NG184"),
        (["ear pain", "otitis", "ear infection"], "NG91"),
        (["glaucoma", "iop", "intraocular"], "NG81_GLAUCOMA"),
        (["ocular hypertension", "raised iop", "eye pressure"], "NG81_HYPERTENSION"),
    ]
    for keywords, gid in mapping:
        if any(kw in symptoms_lower for kw in keywords):
            return gid
    return None


async def select_guideline_fn(
    symptoms: str, triage: dict, answers: dict, patient_record: dict
) -> str:
    """Select the most appropriate NICE guideline for the patient's symptoms."""
    # Use triage's suggested guideline if available and valid
    triage_suggestion = (triage or {}).get("suggested_guideline", "")
    if triage_suggestion:
        g_data = get_guideline(triage_suggestion)
        if g_data:
            logger.info("Using triage-suggested guideline: %s", triage_suggestion)
            return triage_suggestion

    s = (symptoms or "").lower()

    # Try keyword mapping first
    gid = _guess_guideline(s)
    if gid:
        return gid

    # Fallback: ask LLM to pick from available guidelines
    if settings.OPENAI_API_KEY:
        try:
            data = load_all_guidelines()
            available = ", ".join(sorted(data.keys()))
            prompt = f"""Given a patient with these symptoms: {symptoms}

Available NICE guidelines: {available}

Which single guideline ID is most appropriate? Reply with ONLY the guideline ID."""

            raw = await generate(prompt, max_tokens=20, temperature=0.0)
            candidate = raw.strip().upper().replace(" ", "_")
            if candidate in data:
                return candidate
        except Exception as e:
            logger.warning("LLM guideline selection failed: %s", e)

    return "NG204"  # fallback


# ===================================================================
# 5. Extract variables — LLM + regex helpers
# ===================================================================


async def extract_variables_20b(
    guideline: str, history: list, patient: dict, clarifications: dict
) -> Dict[str, Any]:
    """Extract clinical variables from conversation using LLM + regex helpers.

    Matches the notebook pipeline approach:
    1. Build extraction prompt with patient record and variable descriptions
    2. Call LLM for JSON extraction
    3. Apply regex-based fix_variable_extraction helpers
    """
    # Build scenario from conversation history
    scenario_parts = []
    for msg in history or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            scenario_parts.append(msg.get("content", ""))
    scenario = " ".join(scenario_parts)

    # Add patient context
    if patient.get("age"):
        scenario += f" {patient['age']} year old"
    if patient.get("gender"):
        scenario += f" {patient['gender']}"
    if patient.get("conditions"):
        scenario += f" with {', '.join(patient['conditions'])}"

    # Get required variables for this guideline
    g_data = get_guideline(guideline)
    if not g_data:
        return {}

    all_vars = list(
        set(
            v
            for spec in g_data["merged_evaluator"].values()
            if isinstance(spec, dict)
            for v in _extract_var_names(spec)
        )
    )[:15]  # Limit to keep prompt manageable

    if not all_vars:
        return {}

    patient_record_section = build_patient_record(scenario)
    var_list = [get_var_description(v) for v in all_vars]

    prompt = f"""You are extracting clinical variables from a patient conversation.

PATIENT RECORD:
{patient_record_section}

CLINICAL SCENARIO:
{scenario}

Extract these variables in JSON format:
{chr(10).join(['- ' + v for v in var_list])}

Output ONLY valid JSON with snake_case keys. Use exact key names without descriptions.

JSON:
"""

    try:
        raw = await generate(prompt, max_tokens=300, temperature=0.0)
        extracted = extract_json_from_text(raw)
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        extracted = {}

    # Apply regex-based fixes
    extracted = fix_variable_extraction(extracted, scenario)
    extracted = fix_variable_extraction_v2(extracted, scenario)

    # Normalize variable names — LLMs often use aliases instead of the exact
    # evaluator variable names. Map common aliases to correct names.
    _var_aliases = {
        # NG136 hypertension
        "abpm_accepted": "abpm_tolerated",
        "abpm_done": "abpm_tolerated",
        "abpm_average": "abpm_daytime",
        "abpm_reading": "abpm_daytime",
        "abpm_result": "abpm_daytime",
        "cvd": "cardiovascular_disease",
        "cv_disease": "cardiovascular_disease",
        "heart_disease": "cardiovascular_disease",
        "kidney_disease": "renal_disease",
        "bp": "clinic_bp",
        "blood_pressure": "clinic_bp",
        "repeat_bp": "repeat_clinic_bp",
        "emergency": "emergency_signs",
        "ethnicity_not_black": "not_black_african_caribbean",
        # NG232 head injury
        "gcs": "gcs_score",
        "glasgow_coma_scale": "gcs_score",
        "loc": "loss_of_consciousness",
        "vomiting": "persistent_vomiting",
        "seizure": "seizure_present",
        "seizures": "seizure_present",
        "amnesia": "amnesia_since_injury",
        "skull_fracture": "basal_skull_fracture",
        "spine_injury": "suspected_cervical_spine_injury",
        # NG84 sore throat
        "centor": "centor_score",
        "feverpain": "feverpain_score",
        "fever_pain_score": "feverpain_score",
        # NG91 otitis media
        "ear_discharge": "otorrhoea",
        "bilateral_infection": "infection_both_ears",
        "penicillin_allergy": "penicillin_allergy_intolerance",
        # NG112 UTI
        "uti": "current_episode_uti",
        "menopausal": "perimenopause_or_menopause",
        # NG81 glaucoma
        "iop_reading": "intraocular_pressure",
    }
    for alias, correct in _var_aliases.items():
        if alias in extracted and correct not in extracted:
            extracted[correct] = extracted.pop(alias)
        elif alias in extracted:
            del extracted[alias]

    # Enrich from patient record — map structured patient data to evaluator vars.
    # IMPORTANT: For comorbidities (diabetes, CVD, renal), the patient record is
    # the source of truth. The LLM may hallucinate these from symptom text.
    # We OVERRIDE LLM values with patient record data for these fields.
    if patient.get("age") and "age" not in extracted:
        extracted["age"] = patient["age"]
    if patient.get("conditions"):
        conds_lower = [c.lower() for c in patient["conditions"]]

        # Detect which conditions the patient HAS from their medical record
        has_diabetes = any("diabetes" in c for c in conds_lower)
        has_cvd = any(w in c for c in conds_lower for w in ("cardiovascular", "heart disease", "cvd", "coronary"))
        has_renal = any(w in c for c in conds_lower for w in ("renal", "kidney", "ckd"))
        has_hypertension = any("hypertension" in c for c in conds_lower)

        # Override LLM-extracted comorbidities with patient record truth
        if "diabetes" in all_vars:
            extracted["diabetes"] = has_diabetes
        if "cardiovascular_disease" in all_vars:
            extracted["cardiovascular_disease"] = has_cvd
        if "renal_disease" in all_vars:
            extracted["renal_disease"] = has_renal
        if has_hypertension:
            extracted.setdefault("hypertension_history", True)
    else:
        # No conditions on record — OVERRIDE LLM values to False.
        # The LLM may hallucinate comorbidities from symptom text.
        # With no conditions in the patient record, they are definitively False.
        if "diabetes" in all_vars:
            extracted["diabetes"] = False
        if "cardiovascular_disease" in all_vars:
            extracted["cardiovascular_disease"] = False
        if "renal_disease" in all_vars:
            extracted["renal_disease"] = False

    if patient.get("recent_vitals", {}).get("last_bp") and "clinic_bp" not in extracted:
        extracted["clinic_bp"] = patient["recent_vitals"]["last_bp"]

    # Estimate QRISK when not explicitly provided.
    # QRISK >= 10% is very likely for patients over 60 with hypertension.
    # This avoids the tree stopping at n19 for lack of QRISK data.
    if "qrisk_10yr" in all_vars and "qrisk_10yr" not in extracted:
        age = extracted.get("age") or patient.get("age") or 0
        has_hyp = extracted.get("hypertension_history", False)
        has_diab = extracted.get("diabetes", False)
        if isinstance(age, (int, float)) and age >= 60 and has_hyp:
            extracted["qrisk_10yr"] = 15  # conservative estimate
        elif isinstance(age, (int, float)) and age >= 50 and has_hyp and has_diab:
            extracted["qrisk_10yr"] = 12  # conservative estimate

    # Default safe values for binary flags that are typically false unless stated.
    # These are emergency/safety red flags across all guidelines that should be false
    # unless explicitly mentioned in the clinical scenario.
    _default_false_vars = {
        # NG136 hypertension — emergency flags
        "emergency_signs", "retinal_haemorrhage", "papilloedema",
        "life_threatening_symptoms", "target_organ_damage",
        # NG136 hypertension — comorbidities (default False unless patient record says otherwise)
        # These are checked earlier from patient conditions (line ~386), so only
        # default to False here if the patient record didn't already set them True.
        "cardiovascular_disease", "renal_disease", "diabetes",
        # NG232 head injury
        "basal_skull_fracture", "suspected_open_fracture", "intubation_needed",
        "suspicion_non_accidental_injury", "suspected_cervical_spine_injury",
        "suspicion_cervical_spine_injury", "clotting_disorder_present",
        # NG84/NG91 infections
        "systemically_very_unwell", "severe_systemic_infection_or_severe_complications",
        "signs_of_serious_illness_condition", "serious_illness_condition",
        # NG184 bites
        "signs_serious_illness_or_penetrating_wound",
        # NG133 pre-eclampsia
        "intrauterine_death", "placental_abruption",
        "previous_severe_eclampsia",
        # NG81 glaucoma/ocular hypertension — treatment-path flags
        # These are false for newly diagnosed patients not yet on treatment
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
        "satisfactory_adherence_and_technique",
        "needs_additional_iop_reduction",
        # NG91 otitis media — treatment outcome flags
        "high_risk_complications",
        # NG84 sore throat — high risk of complications (with "of_" in name)
        "high_risk_of_complications",
    }
    for var in _default_false_vars:
        if var in all_vars and var not in extracted:
            extracted[var] = False

    # Infer not_black_african_caribbean from patient record ethnicity if available,
    # otherwise leave as None so the clarification system asks the user
    if "not_black_african_caribbean" in all_vars and "not_black_african_caribbean" not in extracted:
        ethnicity = (patient.get("ethnicity") or "").lower()
        if ethnicity:
            is_black = any(k in ethnicity for k in ("black", "african", "caribbean"))
            extracted["not_black_african_caribbean"] = not is_black

    # Infer no_epilepsy_history: default True unless patient has epilepsy
    if "no_epilepsy_history" in all_vars and "no_epilepsy_history" not in extracted:
        conds = [c.lower() for c in (patient.get("conditions") or [])]
        has_epilepsy = any("epilep" in c for c in conds)
        extracted["no_epilepsy_history"] = not has_epilepsy

    # Determine if patient is already on treatment (check medications)
    patient_meds = patient.get("medications") or []
    on_treatment = len(patient_meds) > 0

    # Antihypertensive drug classes for detecting current BP treatment
    _bp_drug_keywords = {
        "amlodipine", "nifedipine", "felodipine", "lercanidipine",  # CCBs
        "ramipril", "lisinopril", "enalapril", "perindopril",  # ACE inhibitors
        "losartan", "candesartan", "valsartan", "irbesartan", "olmesartan",  # ARBs
        "bendroflumethiazide", "indapamide", "chlorthalidone",  # Thiazides
        "spironolactone", "doxazosin", "bisoprolol", "atenolol",  # Step 4
    }
    on_bp_treatment = False
    for m in patient_meds:
        med_name = (m.get("name", "") if isinstance(m, dict) else str(m)).lower()
        if any(drug in med_name for drug in _bp_drug_keywords):
            on_bp_treatment = True
            break

    # For patients already on BP treatment with target_bp_achieved=False,
    # KEEP it — this is a valid clinical finding that drives Step 2/3/4 selection.
    # Only remove treatment-outcome vars for patients NOT yet on treatment.
    _treatment_outcome_vars = {
        "treatment_response", "treatment_completed",
        "remission_achieved",  # NG222
        "back_up_antibiotic_prescription_given", "immediate_antibiotic_prescription_given",  # NG84
        "backup_antibiotic_given", "no_antibiotic_given", "immediate_antibiotic_not_given",  # NG91
        "reassessment_needed_due_to_worsening_symptoms",  # NG84
        "insufficient_iop_reduction_post_surgery", "reduced_effects_of_initial_slt",  # NG81
    }
    if not on_bp_treatment:
        _treatment_outcome_vars.add("target_bp_achieved")

    # Collect variable names provided via clarification answers (tagged questions).
    # These are clinician-confirmed values and should NOT be deleted.
    clarification_var_names = set()
    import re as _re_pre
    for q in (clarifications or {}).keys():
        m = _re_pre.match(r"^\[var:(\w+)\]", q)
        if m:
            clarification_var_names.add(m.group(1))

    # Treatment outcome vars represent decisions/outcomes that haven't happened
    # yet during an initial consultation. The LLM can hallucinate both True and
    # False values. Delete ALL LLM-guessed values — only keep if confirmed via
    # clarification answers (e.g. follow-up visit).
    for var in _treatment_outcome_vars:
        if var in extracted and var not in clarification_var_names:
            del extracted[var]

    # Variables that represent clinical actions/procedures — the LLM must NOT
    # guess these because they depend on what the clinician actually did.
    # Only set them from clarification answers (handled below).
    _clinician_action_vars = {
        "abpm_tolerated", "abpm_daytime", "hbpm_average",  # NG136 — diagnostic procedures
        "abpm_accepted", "abpm_done",  # aliases (mapped above but just in case)
    }
    for var in _clinician_action_vars:
        if var in extracted and var not in clarification_var_names:
            del extracted[var]

    # Ethnicity-based variables — the LLM must NOT guess these from context.
    # They should only come from: (1) patient record, (2) clarification answer.
    # The LLM frequently defaults not_black_african_caribbean to True, skipping
    # the ethnicity question entirely.
    _ethnicity_vars = {"not_black_african_caribbean"}
    for var in _ethnicity_vars:
        if var in extracted and var not in clarification_var_names:
            del extracted[var]

    # Infer target_bp_achieved from clinic_bp if we have a reading AND the
    # patient is already on BP treatment.  For newly-diagnosed patients who
    # aren't on any antihypertensive yet, we must NOT set this variable —
    # otherwise the decision tree cascades through every treatment step
    # (Step 1→2→3→4) instead of stopping at "offer treatment" (Step 1).
    if on_bp_treatment and "target_bp_achieved" in all_vars and "target_bp_achieved" not in extracted:
        bp_str = extracted.get("clinic_bp", "")
        if bp_str:
            from app.guideline_engine import parse_bp
            bp = parse_bp(bp_str)
            if bp:
                age = extracted.get("age", 0)
                target_sys = 150 if (isinstance(age, (int, float)) and age >= 80) else 140
                extracted["target_bp_achieved"] = bp[0] < target_sys and bp[1] < 90

    # Merge in clarification answers — use [var:...] tag to map answers
    # to the correct variable without fragile keyword matching.
    import re as _re
    for q, a in (clarifications or {}).items():
        a_str = str(a) if a else ""
        a_lower = a_str.lower()

        # Extract target variable from tag
        var_tag_match = _re.match(r"^\[var:(\w+)\]\s*", q)
        target_var = var_tag_match.group(1) if var_tag_match else None
        q_text = q[var_tag_match.end():] if var_tag_match else q
        q_lower = q_text.lower()

        # Detect "not known" / "unknown" — skip these entirely.
        _unknown_phrases = ("not known", "unknown", "unsure", "don't know", "no idea",
                            "not sure", "not available", "n/a", "can't say", "not recorded")
        is_unknown = any(phrase in a_lower for phrase in _unknown_phrases)

        bp_match = _re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", a_str)
        num_match = _re.search(r"(\d+\.?\d*)", a_str)
        is_yes = any(w in a_lower for w in ("yes", "true", "confirmed", "positive", "done", "tolerated", "completed"))
        is_no = any(w in a_lower for w in ("no", "false", "negative", "denied", "absent", "none", "declined", "refused"))

        # Skip "unknown" answers — don't set the variable at all
        if is_unknown and not bp_match and not num_match:
            continue

        if target_var:
            # Direct variable mapping from tag
            _bp_vars = {"clinic_bp", "abpm_daytime", "hbpm_average", "bp"}
            if target_var in _bp_vars and bp_match:
                extracted[target_var] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif target_var == "qrisk_10yr":
                # Check qualifier phrases BEFORE raw numbers — "less than 10%"
                # contains "10" which the regex would incorrectly match as 10.0
                if any(w in a_lower for w in ("less than 10", "below 10", "under 10")):
                    extracted["qrisk_10yr"] = 5
                elif any(w in a_lower for w in ("greater than 10", "above 10", "over 10")):
                    extracted["qrisk_10yr"] = 15
                elif a_lower.strip() in ("low", "low risk"):
                    extracted["qrisk_10yr"] = 5
                elif a_lower.strip() in ("high", "high risk"):
                    extracted["qrisk_10yr"] = 15
                else:
                    qrisk_num = _re.search(r"(\d+\.?\d*)\s*%?", a_str)
                    if qrisk_num:
                        extracted["qrisk_10yr"] = float(qrisk_num.group(1))
            elif target_var == "not_black_african_caribbean":
                if is_yes:
                    extracted["not_black_african_caribbean"] = False
                elif is_no:
                    extracted["not_black_african_caribbean"] = True
            elif target_var == "abpm_tolerated":
                if is_yes:
                    extracted["abpm_tolerated"] = True
                elif is_no:
                    extracted["abpm_tolerated"] = False
                if bp_match and "abpm_daytime" not in extracted:
                    extracted["abpm_daytime"] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif target_var == "target_bp_achieved":
                if bp_match:
                    reading = f"{bp_match.group(1)}/{bp_match.group(2)}"
                    extracted["clinic_bp"] = reading
                    systolic = int(bp_match.group(1))
                    extracted["target_bp_achieved"] = systolic < 140
                elif is_yes:
                    extracted["target_bp_achieved"] = True
                elif is_no:
                    extracted["target_bp_achieved"] = False
            elif bp_match:
                extracted[target_var] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            elif is_yes or is_no:
                if target_var.startswith("not_") or target_var.startswith("no_"):
                    extracted[target_var] = is_no
                else:
                    extracted[target_var] = is_yes
            elif num_match:
                extracted[target_var] = float(num_match.group(1))
            else:
                extracted[target_var] = a_str
        else:
            # Legacy: untagged questions — fall back to keyword matching
            for var_name in all_vars:
                var_words = var_name.lower().replace("_", " ").split()
                if any(w in q_lower for w in var_words if len(w) > 3):
                    if var_name not in extracted:
                        if is_yes or is_no:
                            if var_name.startswith("not_") or var_name.startswith("no_"):
                                extracted[var_name] = is_no
                            else:
                                extracted[var_name] = is_yes
                        else:
                            extracted[var_name] = a_str
                    break

    logger.debug("Final extracted variables: %s", {k: v for k, v in extracted.items() if v is not None})
    return extracted


def _extract_var_names(spec: dict) -> List[str]:
    """Recursively extract variable names from a condition spec."""
    names = []
    if "variable" in spec:
        names.append(spec["variable"])
    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            if isinstance(sub, dict):
                names.extend(_extract_var_names(sub))
    if "conditions" in spec and isinstance(spec["conditions"], list):
        for sub in spec["conditions"]:
            if isinstance(sub, dict):
                names.extend(_extract_var_names(sub))
    # Safety: handle "variables" shorthand (list of variable name strings)
    if "variables" in spec and isinstance(spec["variables"], list):
        for v in spec["variables"]:
            if isinstance(v, str):
                names.append(v)
    return names


# ===================================================================
# 6. Walk guideline graph — real decision tree traversal
# ===================================================================


async def walk_guideline_graph_fn(
    guideline: str,
    variables: dict,
    current_node: Optional[str],
    pathway: list,
) -> Dict[str, Any]:
    """Walk the NICE guideline decision tree using the real graph engine.

    Returns traversal result with reached actions, path, and missing variables.
    """
    g_data = get_guideline(guideline)
    if not g_data:
        logger.warning("Guideline %s not found, returning terminal", guideline)
        return {
            "current_node": current_node or "start",
            "pathway_walked": list(pathway or []),
            "terminal": True,
        }

    nodes = g_data["guideline"]["nodes"]
    edges = g_data["guideline"]["edges"]
    evaluator = g_data["merged_evaluator"]

    result = traverse_guideline_graph(nodes, edges, evaluator, variables)

    walked = [f"{p[0]}({p[2]})" for p in result["path"]]
    last_node = result["path"][-1][0] if result["path"] else (current_node or "start")

    has_missing = bool(result["missing_variables"])
    return {
        "current_node": last_node,
        "pathway_walked": walked,
        "terminal": not has_missing,
        "reached_actions": result["reached_actions"],
        "missing_variables": result["missing_variables"],
    }


# ===================================================================
# 7. Format output — template-based (no LLM needed)
# ===================================================================


async def format_output_20b(
    guideline: str,
    triage: dict,
    variables: dict,
    pathway: list,
    patient: dict,
) -> Dict[str, Any]:
    """Format final recommendation using template-based formatting.

    Uses action nodes from guideline graph traversal. Falls back to LLM
    if no actions are available in the traversal result.
    """
    g_data = get_guideline(guideline)

    # Get action nodes from a fresh traversal
    actions = []
    if g_data:
        nodes = g_data["guideline"]["nodes"]
        edges = g_data["guideline"]["edges"]
        evaluator = g_data["merged_evaluator"]
        result = traverse_guideline_graph(nodes, edges, evaluator, variables)
        actions = result["reached_actions"]

    if actions:
        # Build scenario string from patient data
        scenario_parts = []
        if patient.get("age"):
            scenario_parts.append(f"{patient['age']} year old")
        if patient.get("gender"):
            scenario_parts.append(patient["gender"])
        if patient.get("conditions"):
            scenario_parts.append(f"with {', '.join(patient['conditions'])}")
        scenario = " ".join(scenario_parts) if scenario_parts else ""

        recommendation = format_recommendation_template(
            guideline, scenario, actions, variables,
            medications=patient.get("medications"),
        )
        return {
            "final_recommendation": recommendation,
            "citation": guideline,
        }

    # Fallback: LLM-based formatting
    if settings.OPENAI_API_KEY:
        try:
            name = patient.get("first_name", "the patient")
            prompt = f"""Provide a concise clinical recommendation for {name} based on
guideline {guideline}. Known variables: {json.dumps(variables)}.
Keep it under 3 sentences and professional. Do not start with 'Based on NICE'."""

            rec = await generate(prompt, max_tokens=200, temperature=0.0)
            return {"final_recommendation": rec, "citation": guideline}
        except Exception as e:
            logger.warning("LLM formatting failed: %s", e)

    # Final fallback
    name = patient.get("first_name", "there")
    return {
        "final_recommendation": (
            f"{name}, please seek medical advice if symptoms worsen or persist."
        ),
        "citation": guideline,
    }


# ===================================================================
# Build all dependencies
# ===================================================================


def build_orchestration_deps() -> OrchestrationDeps:
    """Assemble all orchestration callables."""
    # Pre-load guidelines at startup
    load_all_guidelines()

    return {
        "fetch_patient": fetch_patient,
        "triage_agent": triage_agent,
        "gpt_clarifier": gpt_clarifier,
        "select_guideline": select_guideline_fn,
        "extract_variables_20b": extract_variables_20b,
        "walk_guideline_graph": walk_guideline_graph_fn,
        "format_output_20b": format_output_20b,
    }
