"""
NICE Guideline Graph Traversal Engine

Loads guideline decision trees and evaluator logic from JSON files,
traverses them based on patient variables, and provides helper functions
for variable extraction, clarification question parsing, and
recommendation formatting.

Ported from the gpt_oss_20b_final_test_suite.ipynb notebook.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory (relative to this file)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_GUIDELINES_DIR = _DATA_DIR / "guidelines"
_EVALUATORS_DIR = _DATA_DIR / "evaluators"

# Filename -> guideline ID mapping
_FILENAME_TO_ID = {
    "ng84": "NG84",
    "ng91": "NG91",
    "ng112": "NG112",
    "ng133": "NG133",
    "ng136": "NG136",
    "ng184": "NG184",
    "ng222": "NG222",
    "ng232": "NG232",
    "ng81_chronic_glaucoma": "NG81_GLAUCOMA",
    "ng81_ocular_hypertension": "NG81_HYPERTENSION",
}

# Reverse mapping: guideline ID -> filename stem
_ID_TO_FILENAME = {v: k for k, v in _FILENAME_TO_ID.items()}


# ===================================================================
# Guideline data loading
# ===================================================================

_guideline_cache: Dict[str, Dict[str, Any]] = {}


def load_all_guidelines() -> Dict[str, Dict[str, Any]]:
    """Load all guideline + evaluator JSON files into memory.

    Returns a dict keyed by guideline ID (e.g. 'NG84') with values:
        {
            "guideline": { "nodes": [...], "edges": [...] },
            "evaluator": { "n1": {...}, ... },
            "merged_evaluator": { ... }  # combined evaluator entries
        }
    """
    global _guideline_cache
    if _guideline_cache:
        return _guideline_cache

    data: Dict[str, Dict[str, Any]] = {}

    for stem, gid in _FILENAME_TO_ID.items():
        guideline_path = _GUIDELINES_DIR / f"{stem}.json"
        evaluator_path = _EVALUATORS_DIR / f"{stem}_eval.json"

        if not guideline_path.exists():
            logger.warning("Guideline file not found: %s", guideline_path)
            continue

        with open(guideline_path) as f:
            guideline = json.load(f)

        evaluator = {}
        if evaluator_path.exists():
            with open(evaluator_path) as f:
                evaluator = json.load(f)

        # Build merged evaluator (flatten if nested under keys)
        merged = {}
        if isinstance(evaluator, dict):
            for key, val in evaluator.items():
                if isinstance(val, dict) and "variable" not in val and "type" not in val:
                    # Nested structure — merge inner keys
                    merged.update(val)
                else:
                    merged[key] = val

        effective_eval = merged if merged else evaluator
        data[gid] = {
            "guideline": guideline,
            "evaluator": evaluator,
            "merged_evaluator": effective_eval,
            "all_vars": set(get_all_variables_from_evaluator(effective_eval)),
        }

    node_count = sum(
        len(d["guideline"].get("nodes", [])) for d in data.values()
    )
    logger.info(
        "Loaded %d guidelines with %d total nodes", len(data), node_count
    )

    _guideline_cache = data
    return data


def get_guideline(guideline_id: str) -> Optional[Dict[str, Any]]:
    """Get a single guideline's data by ID. Loads all if not cached."""
    data = load_all_guidelines()
    return data.get(guideline_id)


# ===================================================================
# Blood pressure parsing
# ===================================================================


def parse_bp(bp_string) -> Optional[Tuple[int, int]]:
    """Parse a blood pressure string like '180/120' into (systolic, diastolic)."""
    if bp_string is None:
        return None
    if isinstance(bp_string, (list, tuple)) and len(bp_string) == 2:
        try:
            return (int(bp_string[0]), int(bp_string[1]))
        except (ValueError, TypeError):
            return None
    if not isinstance(bp_string, str):
        return None
    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(bp_string))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# ===================================================================
# Condition evaluation
# ===================================================================


def _compare(value, threshold, op):
    """Apply a comparison operator between value and threshold."""
    ops = {
        ">=": lambda v, t: v >= t,
        "<=": lambda v, t: v <= t,
        ">": lambda v, t: v > t,
        "<": lambda v, t: v < t,
        "==": lambda v, t: v == t,
        "!=": lambda v, t: v != t,
    }
    fn = ops.get(op)
    return fn(value, threshold) if fn else None


def evaluate_single_condition(condition_spec, variables):
    """Evaluate a single condition specification against patient variables.

    Returns True/False for boolean conditions, None if a required variable
    is missing, or a string (edge label) for treatment_type conditions.
    """
    if condition_spec is None:
        return None

    ctype = condition_spec.get("type")

    # Shorthand nested 'and' key
    if "and" in condition_spec and ctype is None and "variable" not in condition_spec:
        sub_conditions = condition_spec["and"]
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is None for r in results):
            return None
        return all(results)

    # Simple variable check (no 'type' field)
    if ctype is None and "variable" in condition_spec:
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lower = val.lower()
            if lower in ("true", "yes", "1"):
                return True
            if lower in ("false", "no", "0", ""):
                return False
            return True
        if isinstance(val, (int, float)):
            return bool(val)
        return bool(val)

    # Numeric compare (also used for age_compare)
    if ctype in ("numeric_compare", "age_compare"):
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if val is None:
            return None
        try:
            val = float(val)
        except (ValueError, TypeError):
            return None
        threshold = float(condition_spec["threshold"])
        return _compare(val, threshold, condition_spec["op"])

    # Blood pressure compare
    if ctype == "bp_compare":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        bp_val = parse_bp(variables[var_name])
        bp_thresh = parse_bp(condition_spec["threshold"])
        if bp_val is None or bp_thresh is None:
            return None
        op = condition_spec["op"]
        sys_v, dia_v = bp_val
        sys_t, dia_t = bp_thresh
        if op == ">=":
            return sys_v >= sys_t or dia_v >= dia_t
        elif op == ">":
            return sys_v > sys_t or dia_v > dia_t
        elif op == "<=":
            return sys_v <= sys_t and dia_v <= dia_t
        elif op == "<":
            return sys_v < sys_t and dia_v < dia_t
        elif op == "==":
            return sys_v == sys_t and dia_v == dia_t
        return None

    # Blood pressure range
    if ctype == "bp_range":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        bp_val = parse_bp(variables[var_name])
        if bp_val is None:
            return None
        sys_v, dia_v = bp_val
        sys_min = condition_spec.get("systolic_min", 0)
        sys_max = condition_spec.get("systolic_max", 999)
        dia_min = condition_spec.get("diastolic_min", 0)
        dia_max = condition_spec.get("diastolic_max", 999)
        return (sys_min <= sys_v <= sys_max) and (dia_min <= dia_v <= dia_max)

    # AND logic
    if ctype == "and":
        sub_conditions = condition_spec.get("conditions", [])
        # Safety: handle "variables" shorthand (list of variable name strings)
        if not sub_conditions and "variables" in condition_spec:
            sub_conditions = [{"variable": v} for v in condition_spec["variables"]]
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is None for r in results):
            return None
        return all(results)

    # OR logic
    if ctype == "or":
        sub_conditions = condition_spec.get("conditions", [])
        # Safety: handle "variables" shorthand (list of variable name strings)
        if not sub_conditions and "variables" in condition_spec:
            sub_conditions = [{"variable": v} for v in condition_spec["variables"]]
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is True for r in results):
            return True
        if all(r is None for r in results):
            return None
        # If at least one condition evaluated to False and none to True,
        # treat the OR as False.  This handles clinical scoring ORs where
        # only one scoring system is used (e.g. FeverPAIN provided but
        # Centor missing — if FeverPAIN doesn't match, the answer is No).
        return False

    # Treatment type map (NG222)
    if ctype == "treatment_type":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if val is None:
            return None
        treatment_map = condition_spec.get("map", {})
        if val in treatment_map:
            return treatment_map[val]
        val_lower = str(val).lower().strip()
        for key, label in treatment_map.items():
            if key.lower().strip() == val_lower:
                return label
        return None

    return None


def evaluate_condition(node_id: str, evaluator: dict, variables: dict):
    """Look up a node in the evaluator and evaluate its condition."""
    if not evaluator or node_id not in evaluator:
        return None
    return evaluate_single_condition(evaluator[node_id], variables)


# ===================================================================
# Graph traversal
# ===================================================================


def traverse_guideline_graph(
    nodes: list, edges: list, evaluator: dict, variables: dict
) -> Dict[str, Any]:
    """Walk a guideline decision tree and determine which actions are reached.

    Returns:
        {
            "reached_actions": [str, ...],
            "path": [(node_id, node_text, decision), ...],
            "missing_variables": [str, ...]
        }
    """
    if not nodes:
        return {"reached_actions": [], "path": [], "missing_variables": []}

    node_map = {n["id"]: n for n in nodes}
    edges_from: Dict[str, List[Tuple[str, str]]] = {}
    incoming = set()
    for e in edges:
        src = e["from"]
        tgt = e["to"]
        label = e.get("label", "")
        edges_from.setdefault(src, []).append((tgt, label))
        incoming.add(tgt)

    all_node_ids = [n["id"] for n in nodes]
    roots = [nid for nid in all_node_ids if nid not in incoming]
    if not roots:
        roots = [all_node_ids[0]]

    reached_actions: List[str] = []
    path: List[Tuple[str, str, str]] = []
    missing_variables: List[str] = []
    visited = set()
    max_steps = len(nodes) * 3

    queue = list(roots)
    step_count = 0

    while queue and step_count < max_steps:
        step_count += 1
        current_id = queue.pop(0)

        if current_id in visited:
            continue
        visited.add(current_id)

        if current_id not in node_map:
            continue

        node = node_map[current_id]
        node_type = node.get("type", "")
        node_text = node.get("text", "")

        if node_type == "action":
            reached_actions.append(node_text)
            path.append((current_id, node_text, "action"))
            for tgt, label in edges_from.get(current_id, []):
                if label == "next":
                    queue.append(tgt)

        elif node_type == "condition":
            result = evaluate_condition(current_id, evaluator, variables)

            if result is None:
                path.append((current_id, node_text, "missing_variable"))
                _collect_missing_vars(current_id, evaluator, variables, missing_variables)

            elif isinstance(result, str):
                path.append((current_id, node_text, f"treatment_type:{result}"))
                matched = False
                for tgt, label in edges_from.get(current_id, []):
                    if label == result:
                        queue.append(tgt)
                        matched = True
                if not matched:
                    path.append((current_id, node_text, f"no_matching_edge:{result}"))

            elif result is True:
                path.append((current_id, node_text, "yes"))
                for tgt, label in edges_from.get(current_id, []):
                    if label in ("yes", "next"):
                        queue.append(tgt)

            elif result is False:
                path.append((current_id, node_text, "no"))
                for tgt, label in edges_from.get(current_id, []):
                    if label == "no":
                        queue.append(tgt)
        else:
            path.append((current_id, node_text, f"unknown_type:{node_type}"))

    # Deduplicate missing variables preserving order
    seen = set()
    unique_missing = []
    for v in missing_variables:
        if v not in seen:
            seen.add(v)
            unique_missing.append(v)

    return {
        "reached_actions": reached_actions,
        "path": path,
        "missing_variables": unique_missing,
    }


def _collect_missing_vars(node_id, evaluator, variables, missing_list):
    """Extract variable names needed by a node that are not in variables."""
    if not evaluator or node_id not in evaluator:
        return
    _collect_missing_from_spec(evaluator[node_id], variables, missing_list)


def _collect_missing_from_spec(spec, variables, missing_list):
    """Recursively collect missing variable names from a condition spec."""
    if spec is None:
        return

    if "variable" in spec:
        var_name = spec["variable"]
        if var_name not in variables:
            missing_list.append(var_name)

    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            _collect_missing_from_spec(sub, variables, missing_list)

    ctype = spec.get("type")
    if ctype in ("and", "or") and "conditions" in spec:
        for sub in spec["conditions"]:
            _collect_missing_from_spec(sub, variables, missing_list)

    # Safety: handle "variables" shorthand (list of variable name strings)
    if ctype in ("and", "or") and "variables" in spec:
        for var_name in spec["variables"]:
            if var_name not in variables:
                missing_list.append(var_name)

    if ctype == "treatment_type" and "variable" in spec:
        var_name = spec["variable"]
        if var_name not in variables:
            missing_list.append(var_name)


def get_all_variables_from_evaluator(evaluator: dict) -> List[str]:
    """Extract all variable names referenced anywhere in an evaluator."""
    if not evaluator:
        return []
    all_vars: set = set()
    for node_id, spec in evaluator.items():
        _collect_vars_from_spec(spec, all_vars)
    return sorted(all_vars)


def _collect_vars_from_spec(spec, var_set):
    """Recursively collect all variable names from a condition spec."""
    if spec is None or not isinstance(spec, dict):
        return
    if "variable" in spec:
        var_set.add(spec["variable"])
    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            _collect_vars_from_spec(sub, var_set)
    if "conditions" in spec and isinstance(spec["conditions"], list):
        for sub in spec["conditions"]:
            _collect_vars_from_spec(sub, var_set)
    # Safety: handle "variables" shorthand
    if "variables" in spec and isinstance(spec["variables"], list):
        for v in spec["variables"]:
            if isinstance(v, str):
                var_set.add(v)


def get_missing_variables_for_next_step(
    nodes: list, edges: list, evaluator: dict, known_vars: dict
) -> List[str]:
    """Traverse the graph and return variable names needed to advance."""
    result = traverse_guideline_graph(nodes, edges, evaluator, known_vars)
    return result["missing_variables"]


# ===================================================================
# JSON extraction from LLM output
# ===================================================================


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from model output that may contain extra text."""
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    result = {}
    pattern = r'["\'](\w+)["\']\s*:\s*([^,}\n]+)'
    matches = re.findall(pattern, text)
    for key, value in matches:
        value = value.strip().strip("\"'")
        if value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif value.lower() in ("null", "none"):
            result[key] = None
        elif value.isdigit():
            result[key] = int(value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value

    return result if result else {}


# ===================================================================
# Variable extraction helpers (regex-based post-processing)
# ===================================================================


def fix_variable_extraction(extracted: dict, scenario_text: str) -> dict:
    """Regex-based variable extraction from scenario text to fill gaps."""
    fixed = extracted.copy()
    text_lower = scenario_text.lower()

    # Age
    if "age" not in fixed or fixed["age"] is None:
        for pattern in [
            r"(\d+)\s*(?:year|yr|y\.?o\.?)",
            r"age[:\s]+(\d+)",
            r"(\d+)\s*(?:month|mo)\s+old",
        ]:
            match = re.search(pattern, text_lower)
            if match:
                age_val = int(match.group(1))
                if "month" in match.group(0) or "mo" in match.group(0):
                    fixed["age"] = age_val / 12.0
                else:
                    fixed["age"] = age_val
                break

    # Gender
    if "gender" not in fixed or fixed["gender"] is None:
        if re.search(r"\b(male|man|boy|father|husband|mr\.?)\b", text_lower):
            fixed["gender"] = "male"
        elif re.search(
            r"\b(female|woman|girl|mother|wife|mrs\.?|ms\.?|pregnant)\b", text_lower
        ):
            fixed["gender"] = "female"

    # GCS Score
    if "gcs_score" not in fixed or fixed["gcs_score"] is None:
        gcs_match = re.search(r"gcs[:\s]+(\d+)", text_lower)
        if gcs_match:
            fixed["gcs_score"] = int(gcs_match.group(1))

    # Blood Pressure
    if "clinic_bp" not in fixed or fixed["clinic_bp"] is None:
        for pattern in [
            r"(\d{2,3})\s*/\s*(\d{2,3})",
            r"bp[:\s]+(\d{2,3})\s*/\s*(\d{2,3})",
            r"blood pressure[:\s]+(\d{2,3})\s*/\s*(\d{2,3})",
        ]:
            match = re.search(pattern, text_lower)
            if match:
                fixed["clinic_bp"] = f"{match.group(1)}/{match.group(2)}"
                break

    if "bp" not in fixed or fixed["bp"] is None:
        if "clinic_bp" in fixed and fixed["clinic_bp"]:
            fixed["bp"] = fixed["clinic_bp"]

    # Gestational Age
    if "gestational_age" not in fixed or fixed["gestational_age"] is None:
        gest_match = re.search(r"(\d+)\s+weeks?\s+(?:pregnant|gestation)", text_lower)
        if gest_match:
            fixed["gestational_age"] = int(gest_match.group(1))

    # Fever
    if "fever" not in fixed or fixed["fever"] is None:
        if re.search(r"fever|febrile|temperature\s+\d", text_lower):
            temp_match = re.search(r"(\d{2}\.?\d?)\s*[°c]", text_lower)
            if temp_match:
                fixed["fever"] = float(temp_match.group(1))
            else:
                fixed["fever"] = True
        elif re.search(r"no\s+fever|afebrile|apyrexial", text_lower):
            fixed["fever"] = False

    # Head injury
    if "head_injury_present" not in fixed or fixed["head_injury_present"] is None:
        if re.search(r"head\s+injur|hit\s+head|head\s+trauma|struck\s+head", text_lower):
            fixed["head_injury_present"] = True

    # Recurrent UTI
    if "recurrent_uti" not in fixed or fixed["recurrent_uti"] is None:
        if re.search(r"recurrent\s+uti|multiple\s+utis?", text_lower):
            fixed["recurrent_uti"] = True

    # Diabetes
    if "diabetes" not in fixed or fixed["diabetes"] is None:
        if re.search(r"\bdiabetes|diabetic|type [12] diabetes", text_lower):
            fixed["diabetes"] = True
        elif re.search(r"no\s+diabetes|non-diabetic", text_lower):
            fixed["diabetes"] = False

    return fixed


def fix_variable_extraction_v2(extracted: dict, scenario_text: str) -> dict:
    """Additional edge case handling — call AFTER fix_variable_extraction()."""
    fixed = extracted.copy()
    text_lower = scenario_text.lower()

    negation_patterns = [
        (r"no\s+vomiting", "vomiting", False),
        (r"no\s+fever", "fever", False),
        (r"no\s+headache", "headache", False),
        (r"denies\s+chest\s+pain", "emergency_signs", False),
        (r"no\s+loss\s+of\s+consciousness", "loc", False),
        (r"no\s+visual\s+disturbance", "emergency_signs", False),
    ]

    for pattern, var_name, correct_value in negation_patterns:
        if re.search(pattern, text_lower):
            if var_name in fixed:
                fixed[var_name] = correct_value

    # NG222 — Better remission detection
    if "remission" not in fixed or fixed.get("remission") is None:
        if re.search(r"well-controlled|stable mood|asymptomatic", text_lower):
            fixed["remission"] = "full"

    # NG91 — High risk for very young children
    if "high_risk" not in fixed or fixed.get("high_risk") is None:
        age = fixed.get("age")
        if age is not None and age < 1:
            fixed["high_risk"] = True

    # NG84 — Infer Centor score from components
    if "centor_score" not in fixed or fixed.get("centor_score") is None:
        score = 0
        if fixed.get("fever"):
            score += 1
        if fixed.get("tonsillar_exudate") or fixed.get("purulent_tonsils"):
            score += 1
        if fixed.get("tender_lymph_nodes"):
            score += 1
        if fixed.get("cough") is False:
            score += 1
        if score > 0:
            fixed["centor_score"] = score

    return fixed


# ===================================================================
# LLM output helpers
# ===================================================================


def extract_best_question(raw_response: str) -> str:
    """Multi-strategy question extraction from LLM output."""
    text = raw_response.strip()

    # Strategy 1: Quoted question
    quoted = re.findall(r'"([^"]*\?)"', text)
    for q in quoted:
        if len(q) > 15 and q.lower() != "we?":
            return q

    # Strategy 2: "Question:" marker
    if "Question:" in text:
        parts = text.split("Question:")
        candidate = parts[-1].strip().split("\n")[0].strip()
        if "?" in candidate:
            candidate = candidate[: candidate.rindex("?") + 1]
            if len(candidate) > 15:
                return candidate

    # Strategy 3: Real question sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    questions = [
        s.strip()
        for s in sentences
        if "?" in s
        and len(s.strip()) > 15
        and not s.strip().lower().startswith("we?")
        and not s.strip().lower().startswith("we need")
    ]
    if questions:
        return max(questions, key=len)

    # Strategy 4: rindex to find last ?
    if "?" in text:
        last_q = text[: text.rindex("?") + 1]
        for marker in [". ", "\n", ": "]:
            idx = last_q.rfind(marker)
            if idx > 0:
                candidate = last_q[idx + len(marker) :].strip()
                if len(candidate) > 15:
                    return candidate
        if len(last_q) > 15:
            return last_q

    # Fallback
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and not s.lower().startswith("we "):
            return s

    return text[:100] if text else "Could you provide more information?"


def build_patient_record(scenario: str) -> str:
    """Pre-parse scenario into structured PATIENT RECORD for better extraction."""
    scenario_lower = scenario.lower()
    patient_lines = []

    age_match = re.search(
        r"(\d{1,3})\s*(?:year|yr|yo)(?:\s+old)?", scenario_lower
    )
    if not age_match:
        age_match = re.search(r"(?:age|aged)[:\s]+(\d{1,3})", scenario_lower)
    if age_match:
        patient_lines.append(f"Age: {age_match.group(1)} years")

    if re.search(r"\b(male|man|boy)\b", scenario_lower) and not re.search(
        r"\b(female|woman|girl)\b", scenario_lower
    ):
        patient_lines.append("Gender: Male")
    elif re.search(r"\b(female|woman|girl|pregnant|pregnancy)\b", scenario_lower):
        patient_lines.append("Gender: Female")

    conditions = []
    if any(
        kw in scenario_lower
        for kw in ["diabetes", "diabetic", "type 2 dm", "t2dm"]
    ):
        conditions.append("diabetes")
    if any(
        kw in scenario_lower
        for kw in ["hypertension", "high blood pressure", "htn"]
    ):
        conditions.append("hypertension")
    if any(
        kw in scenario_lower
        for kw in ["ckd", "chronic kidney", "renal impairment"]
    ):
        conditions.append("chronic kidney disease")
    if conditions:
        patient_lines.append(f"Medical History: {', '.join(conditions)}")

    return "\n".join(patient_lines) if patient_lines else "No documented medical history"


# Variable descriptions for better extraction prompts
VAR_DESCRIPTIONS = {
    "age": "age (patient's age in years)",
    "mechanism": "mechanism (how injury occurred: fall, assault, RTC, etc.)",
    "vomiting_count": "vomiting_count (number of vomiting episodes as integer)",
    "gcs_score": "gcs_score (Glasgow Coma Scale score 3-15)",
    "emergency_signs": "emergency_signs (true if life-threatening symptoms present)",
    "loss_of_consciousness": "loss_of_consciousness (true/false)",
    "fever": "fever (true if fever present, or temperature value)",
    "duration": "duration (duration in days as integer)",
    "clinic_bp": "clinic_bp (clinic blood pressure reading as string e.g. '155/95')",
    "gestational_age": "gestational_age (weeks of pregnancy as integer)",
    "gender": "gender (male/female)",
    "recurrent_uti": "recurrent_uti (true/false)",
    "head_injury_present": "head_injury_present (true/false)",
    "bite_type": "bite_type (type of animal: cat, dog, human, etc.)",
    "broken_skin": "broken_skin (true/false - whether skin is broken)",
    "high_risk_area": "high_risk_area (true/false - hand, face, genitals, etc.)",
    "treatment_completed": "treatment_completed (true/false)",
    "acute_treatment": "acute_treatment (type of treatment received)",
    "ear_pain": "ear_pain (true/false)",
    "newly_diagnosed": "newly_diagnosed (true/false)",
    "iop": "iop (intraocular pressure as integer)",
    "iop_level": "iop_level (intraocular pressure as integer)",
    "family_history_glaucoma": "family_history_glaucoma (true/false)",
    # NG136 hypertension-specific variables
    "abpm_tolerated": "abpm_tolerated (true if ABPM was done and tolerated, false if declined/not tolerated)",
    "abpm_daytime": "abpm_daytime (ABPM daytime average BP as string e.g. '150/95')",
    "hbpm_average": "hbpm_average (home BP monitoring average as string e.g. '145/90')",
    "repeat_clinic_bp": "repeat_clinic_bp (repeat clinic BP reading as string e.g. '180/120')",
    "not_black_african_caribbean": "not_black_african_caribbean (true if patient is NOT of black African or African-Caribbean family origin)",
    "cardiovascular_disease": "cardiovascular_disease (true if patient has cardiovascular disease)",
    "target_organ_damage": "target_organ_damage (true if target-organ damage present)",
    "diabetes": "diabetes (true if patient has diabetes)",
    "renal_disease": "renal_disease (true if patient has renal/kidney disease)",
    "qrisk_10yr": "qrisk_10yr (10-year QRISK cardiovascular risk score as percentage, e.g. 15)",
    "retinal_haemorrhage": "retinal_haemorrhage (true/false)",
    "papilloedema": "papilloedema (true/false)",
    "life_threatening_symptoms": "life_threatening_symptoms (true if life-threatening symptoms present)",
    "target_bp_achieved": "target_bp_achieved (true if BP is at target on current treatment - only set if patient is already on treatment)",
    # NG232 head injury variables
    "amnesia_since_injury": "amnesia_since_injury (true if patient has amnesia since the injury)",
    "basal_skull_fracture": "basal_skull_fracture (true if signs of basal skull fracture present)",
    "clotting_disorder_present": "clotting_disorder_present (true if patient has a clotting disorder or is on anticoagulants)",
    "consciousness_assessment_needed": "consciousness_assessment_needed (true/false)",
    "drowsiness_present": "drowsiness_present (true if patient is drowsy)",
    "intubation_needed": "intubation_needed (true/false)",
    "no_epilepsy_history": "no_epilepsy_history (true if patient does NOT have epilepsy)",
    "persistent_vomiting": "persistent_vomiting (true if patient has had persistent vomiting)",
    "seizure_present": "seizure_present (true if patient has had a seizure after the injury)",
    "suspected_cervical_spine_injury": "suspected_cervical_spine_injury (true/false)",
    "suspected_open_fracture": "suspected_open_fracture (true/false)",
    "suspicion_non_accidental_injury": "suspicion_non_accidental_injury (true/false)",
    "urgent_diagnosis_needed": "urgent_diagnosis_needed (true/false)",
    # NG84 sore throat variables
    "centor_score": "centor_score (Centor score 0-4)",
    "feverpain_score": "feverpain_score (FeverPAIN score 0-5)",
    "systemically_very_unwell": "systemically_very_unwell (true if patient is systemically very unwell)",
    "high_risk_of_complications": "high_risk_of_complications (true if patient at high risk of complications)",
    # NG91 otitis media variables
    "otorrhoea": "otorrhoea (true if ear discharge present)",
    "infection_both_ears": "infection_both_ears (true if infection in both ears)",
    "penicillin_allergy_intolerance": "penicillin_allergy_intolerance (true if patient has penicillin allergy)",
    # NG112 UTI variables
    "current_episode_uti": "current_episode_uti (true if patient currently has a UTI)",
    "perimenopause_or_menopause": "perimenopause_or_menopause (true if patient is perimenopausal or menopausal)",
    # NG184 bite variables
    "person_with_comorbidities": "person_with_comorbidities (true if patient has comorbidities increasing infection risk)",
    "wound_could_be_deep": "wound_could_be_deep (true if wound could involve deep tissue)",
    # NG222 meningitis variables
    "remission_achieved": "remission_achieved (true if remission has been achieved)",
    "higher_risk_of_relapse": "higher_risk_of_relapse (true if patient is at higher risk of relapse)",
    # NG81 glaucoma variables
    "intraocular_pressure": "intraocular_pressure (IOP reading as integer mmHg)",
    "newly_diagnosed_coag": "newly_diagnosed_coag (true if newly diagnosed chronic open-angle glaucoma)",
    "slt_not_suitable": "slt_not_suitable (true if selective laser trabeculoplasty is not suitable)",
    "risk_of_visual_impairment": "risk_of_visual_impairment (true if patient is at risk of visual impairment)",
    # NG133 pre-eclampsia variables
    "high_risk_factors_1_or_more": "high_risk_factors_1_or_more (true if 1+ high risk factors for pre-eclampsia)",
    "moderate_risk_factors_2_or_more": "moderate_risk_factors_2_or_more (true if 2+ moderate risk factors)",
}


def auto_describe_variable(var_name: str) -> str:
    """Generate a description from a snake_case variable name.

    Used as fallback when a variable is not in VAR_DESCRIPTIONS.
    """
    readable = var_name.replace("_", " ")
    # Detect boolean-style names
    bool_prefixes = ("is", "has", "not", "no", "can", "needs", "should")
    bool_keywords = ("present", "needed", "given", "achieved", "suitable",
                     "effective", "indicated", "suspected", "diagnosed")
    if (any(readable.startswith(p + " ") for p in bool_prefixes) or
            any(k in readable for k in bool_keywords)):
        return f"{var_name} (true/false)"
    # Score-like
    if "score" in readable:
        return f"{var_name} (numeric score)"
    # BP-like
    if "bp" in readable or "pressure" in readable:
        return f"{var_name} (blood pressure reading as string e.g. '140/90', or numeric value)"
    return f"{var_name} (extract this clinical value)"


def get_var_description(var_name: str) -> str:
    """Get description for a variable, using VAR_DESCRIPTIONS or auto-generating."""
    return VAR_DESCRIPTIONS.get(var_name, auto_describe_variable(var_name))


def _split_treatment_steps(actions: List[str]) -> tuple:
    """Separate immediate clinical actions from treatment-ladder steps.

    Treatment steps typically start with "Step N:" and represent a sequential
    escalation ladder — only the first step is the immediate recommendation.
    """
    step_pattern = re.compile(r"^step\s+\d", re.IGNORECASE)
    immediate = []
    steps = []
    for a in actions:
        if step_pattern.match(a.strip()):
            steps.append(a.strip())
        else:
            immediate.append(a.strip())
    return immediate, steps


def _detect_current_treatment_step(medications: list) -> int:
    """Detect which treatment step the patient is currently on based on medications.

    Returns 0 if not on treatment, 1-4 for the detected step.
    """
    if not medications:
        return 0

    med_names = []
    for m in medications:
        name = (m.get("name", "") if isinstance(m, dict) else str(m)).lower()
        med_names.append(name)

    has_ccb = any(d in n for n in med_names for d in ("amlodipine", "nifedipine", "felodipine", "lercanidipine"))
    has_ace_arb = any(d in n for n in med_names for d in ("ramipril", "lisinopril", "enalapril", "perindopril", "losartan", "candesartan", "valsartan", "irbesartan", "olmesartan"))
    has_thiazide = any(d in n for n in med_names for d in ("bendroflumethiazide", "indapamide", "chlorthalidone"))
    has_step4 = any(d in n for n in med_names for d in ("spironolactone", "doxazosin", "bisoprolol", "atenolol"))

    if has_step4:
        return 4
    if has_thiazide and (has_ccb or has_ace_arb):
        return 3
    if (has_ccb and has_ace_arb) or has_thiazide:
        return 2
    if has_ccb or has_ace_arb:
        return 1
    return 0


def format_recommendation_template(
    guideline_id: str,
    scenario: str,
    actions: List[str],
    known_vars: dict,
    medications: Optional[list] = None,
) -> str:
    """Template-based formatting of action nodes into recommendation text.

    No LLM call needed — formats directly from action node text.
    Handles treatment ladders by recommending only the first applicable step.
    """
    # Deduplicate actions preserving order + remove actions whose text
    # is fully contained within another (longer) action
    seen: set = set()
    unique_actions = []
    for a in actions:
        a_lower = a.strip().lower()
        if a_lower not in seen:
            seen.add(a_lower)
            unique_actions.append(a.strip())

    # Remove actions that overlap significantly with another longer action.
    # Check both full substring and trailing-overlap (shared ending words).
    def _has_significant_overlap(shorter: str, longer: str) -> bool:
        s, l = shorter.lower(), longer.lower()
        if s in l:
            return True
        # Check if the last 8+ words of shorter match the end of longer
        s_words = s.split()
        l_words = l.split()
        if len(s_words) >= 8:
            tail = " ".join(s_words[-8:])
            if tail in " ".join(l_words[-12:]):
                return True
        return False

    filtered = []
    for i, a in enumerate(unique_actions):
        is_redundant = False
        for j, b in enumerate(unique_actions):
            if i != j and len(a) < len(b) and _has_significant_overlap(a, b):
                is_redundant = True
                break
        if not is_redundant:
            filtered.append(a)
    unique_actions = filtered if filtered else unique_actions

    # Separate immediate actions from treatment-ladder steps
    immediate, steps = _split_treatment_steps(unique_actions)

    # Filter out process/diagnostic actions that are already completed
    # (e.g. "Measure clinic BP", "Perform ABPM") when we have later actions
    _completed_indicators = {
        "measure clinic bp", "offer abpm to confirm diagnosis", "perform abpm",
        "offer hbpm instead", "offer hbpm", "continue monitoring",
    }
    if len(immediate) > 2:
        immediate = [a for a in immediate if a.strip().lower() not in _completed_indicators] or immediate

    # Detect current treatment step from medications to skip completed steps
    current_step = _detect_current_treatment_step(medications or [])
    if current_step > 0 and len(steps) > current_step:
        # Skip steps the patient is already on — recommend the next step
        steps = steps[current_step:]
    elif current_step > 0 and len(steps) == current_step:
        # Patient is on the last available step — recommend specialist review
        steps = []

    # Build the recommendation text
    parts = []

    # Format immediate actions (diagnoses, treatment decisions)
    # Strip trailing periods from each action to avoid ".." when joining
    immediate = [a.rstrip(".") for a in immediate]
    if immediate:
        if len(immediate) == 1:
            parts.append(immediate[0])
        elif len(immediate) == 2:
            parts.append(f"{immediate[0]}, and {immediate[1][0].lower() + immediate[1][1:]}")
        else:
            parts.append(". ".join(immediate))

    # Add only the first treatment step as the immediate recommendation
    if steps:
        if current_step > 0:
            parts.append(f"Since current treatment has not achieved target BP, escalate to {steps[0]}")
        else:
            parts.append(steps[0])
        if len(steps) > 1:
            parts.append(f"If target BP is not achieved, subsequent steps include "
                         f"{', then '.join(s for s in steps[1:])}")

    # Strip trailing periods from each part before joining to avoid ".."
    actions_text = ". ".join(p.rstrip(".") for p in parts)
    recommendation = f"Based on NICE {guideline_id}, {actions_text}"

    # Append patient context
    age_match = re.search(r"(\d{1,3})\s*(?:year|yr|yo)", scenario.lower())
    if age_match:
        age = age_match.group(1)
        if age not in recommendation:
            recommendation += f" (Patient: {age} years old)"

    if recommendation and recommendation[-1] not in ".!":
        recommendation += "."

    return recommendation
