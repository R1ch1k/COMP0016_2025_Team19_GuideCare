# Guideline JSON Specification & Validation Rules

This document defines the required structure for NICE guideline JSON files and their evaluator files. It serves as a specification for an automated validation/fixing script that should run when guidelines are uploaded.

## File Structure

Each guideline consists of two files:

```
backend/data/guidelines/{guideline_stem}.json    # Decision tree structure
backend/data/evaluators/{guideline_stem}_eval.json  # Condition evaluation logic
```

The `guideline_stem` maps to a guideline ID via `_FILENAME_TO_ID` in `guideline_engine.py`. For example, `ng84` maps to `NG84`, and `ng81_chronic_glaucoma` maps to `NG81`.

---

## 1. Guideline JSON Schema

```json
{
  "guideline_id": "NG84",
  "name": "Human-readable name",
  "version": "...",
  "citation": "...",
  "citation_url": "https://...",
  "rules": ["IF ... THEN ...", ...],
  "nodes": [
    { "id": "n1", "type": "condition", "text": "..." },
    { "id": "n2", "type": "action", "text": "..." }
  ],
  "edges": [
    { "from": "n1", "to": "n2", "label": "yes" }
  ]
}
```

### Node Types

| Type | Purpose | Evaluator Entry? |
|------|---------|-----------------|
| `condition` | Decision point evaluated against patient variables | **YES** (required) |
| `action` | Clinical recommendation or action to take | **NO** (must not have one) |

### Edge Labels

| Label | Meaning | Used On |
|-------|---------|---------|
| `yes` | Condition evaluated to `True` | condition nodes |
| `no` | Condition evaluated to `False` | condition nodes |
| `next` | Sequential flow (always follow) | action nodes, or condition→next-step |
| `otherwise` | Fallback/default path | condition nodes (treated as `no` by engine) |
| Custom strings | Treatment type mapping (e.g. `"yes, antidepressants"`) | condition nodes with `treatment_type` evaluator |

---

## 2. Evaluator JSON Schema

The evaluator file maps **condition node IDs** to their evaluation logic:

```json
{
  "n1": { "variable": "some_boolean_variable" },
  "n3": {
    "type": "numeric_compare",
    "variable": "age",
    "threshold": 16,
    "op": ">="
  }
}
```

### Supported Condition Types

#### Simple boolean variable
```json
{ "variable": "systemically_very_unwell" }
```
Returns `True`/`False` based on the variable value, or `None` if missing.

#### Numeric compare (also `age_compare`)
```json
{
  "type": "numeric_compare",
  "variable": "gcs_score",
  "threshold": 12,
  "op": "<="
}
```
Operators: `>=`, `>`, `<=`, `<`, `==`

#### Blood pressure compare (`bp_compare`)
```json
{
  "type": "bp_compare",
  "variable": "clinic_bp",
  "threshold": "140/90",
  "op": ">="
}
```

#### Blood pressure range (`bp_range`)
```json
{
  "type": "bp_range",
  "variable": "abpm_daytime",
  "systolic_min": 135, "systolic_max": 149,
  "diastolic_min": 85, "diastolic_max": 94
}
```

#### AND logic
```json
{
  "type": "and",
  "conditions": [
    { "variable": "pregnant" },
    { "variable": "penicillin_allergy" }
  ]
}
```
Returns `True` if ALL conditions are `True`, `None` if any is missing, `False` otherwise.

#### OR logic
```json
{
  "type": "or",
  "conditions": [
    { "variable": "systemically_very_unwell" },
    { "variable": "serious_illness_condition" }
  ]
}
```
Returns `True` if ANY condition is `True`, `None` only if ALL are missing, `False` if at least one is `False` and none is `True`.

#### Treatment type map (special)
```json
{
  "type": "treatment_type",
  "variable": "acute_treatment",
  "map": {
    "psychological therapy alone": "yes, psychological therapy",
    "antidepressants alone": "yes, antidepressants"
  }
}
```
Returns the edge label string (not `True`/`False`). The returned string must match an edge label on the condition node.

---

## 3. Validation Rules (for auto-fix script)

### Rule 1: Single Connected Graph

**Requirement:** All nodes must be reachable from a single root via undirected BFS.

**How to check:**
1. Build adjacency list (ignoring edge direction)
2. Find roots = nodes with no incoming edges
3. BFS from roots[0] — all nodes must be visited

**Common problems and fixes:**
- **Multiple disconnected subgraphs** (e.g. age-based branches): Add connecting edges between subgraphs. Typically the first condition's `no` branch should lead to the next subgraph's root.
- **No root (cycle)**: Break the cycle by replacing the back-edge with a new terminal action node (e.g. "Continue current treatment and monitor").
- **Orphan nodes** (in no edges at all): Either connect them or remove them.

### Rule 2: Single Root

**Requirement:** Exactly one node should have in-degree 0 (no incoming edges).

**How to check:** Count nodes where `node_id not in {e["to"] for e in edges}`.

**Fix:** If multiple roots exist, they usually represent parallel clinical pathways that should be connected via a branching condition (typically age-based or severity-based).

### Rule 3: Evaluator Keys Match Condition Nodes Only

**Requirement:**
- Every key in the evaluator file MUST correspond to a node with `"type": "condition"` in the guideline
- No evaluator key should reference an action node or a non-existent node

**How to check:**
```python
for key in evaluator:
    assert key in guideline_nodes, f"{key} not in guideline"
    assert guideline_nodes[key]["type"] == "condition", f"{key} is not a condition"
```

**Fix:** Remove entries for action nodes or non-existent nodes.

### Rule 4: All Condition Nodes Have Evaluator Entries

**Requirement:** Every node with `"type": "condition"` MUST have a corresponding entry in the evaluator file.

**How to check:**
```python
for node in nodes:
    if node["type"] == "condition":
        assert node["id"] in evaluator, f"condition {node['id']} missing from evaluator"
```

**Fix:** Generate a placeholder evaluator entry. For simple yes/no conditions, derive a variable name from the node text (e.g. "Person is systemically unwell" → `systemically_unwell`).

### Rule 5: No Phantom Evaluator Entries

**Requirement:** Evaluator must not contain entries for node IDs that don't exist in the guideline's node list.

**How to check:**
```python
guideline_node_ids = {n["id"] for n in nodes}
for key in evaluator:
    assert key in guideline_node_ids, f"phantom evaluator entry: {key}"
```

**Fix:** Remove phantom entries.

### Rule 6: Edge References Valid Nodes

**Requirement:** Every `from` and `to` in edges must reference a node ID that exists.

**How to check:**
```python
node_ids = {n["id"] for n in nodes}
for e in edges:
    assert e["from"] in node_ids
    assert e["to"] in node_ids
```

### Rule 7: Condition Nodes Have Outgoing Edges

**Requirement:** Every condition node should have at least one outgoing edge (otherwise the BFS stops dead).

**How to check:**
```python
outgoing = {e["from"] for e in edges}
for node in nodes:
    if node["type"] == "condition":
        assert node["id"] in outgoing, f"condition {node['id']} has no outgoing edges"
```

### Rule 8: Evaluator Variable Names Are Consistent

**Recommendation:** Variable names should use `snake_case` and match the naming conventions used in `deps.py`:
- Boolean flags: `systemically_very_unwell`, `high_risk_of_complications`
- Numeric values: `age`, `gcs_score`, `feverpain_score`
- BP readings: `clinic_bp`, `abpm_daytime`
- Negated booleans: `not_black_african_caribbean` (prefix with `not_`)

---

## 4. How the Engine Uses These Files

### Loading (`guideline_engine.py:load_all_guidelines`)
- Reads all guideline + evaluator JSONs at startup
- Caches in `_guideline_cache` keyed by guideline ID
- Creates `merged_evaluator` (flattened evaluator dict)

### Traversal (`guideline_engine.py:traverse_guideline_graph`)
1. Finds **roots** (nodes with in-degree 0); falls back to first node if no root
2. BFS queue starting from all roots
3. For each node:
   - **Action node**: Add text to `reached_actions`, follow `"next"` edges
   - **Condition node**: Call `evaluate_condition(node_id, evaluator, variables)`
     - `True` → follow `"yes"` and `"next"` edges
     - `False` → follow `"no"` edges
     - `None` (missing variable) → record in `missing_variables`, stop this branch
     - String (treatment type) → follow edge matching the returned label
4. Cycle protection: `visited` set prevents re-processing nodes
5. Step limit: `max_steps = len(nodes) * 3` prevents infinite loops

### Variable Extraction (`deps.py:extract_variables_20b`)
- LLM extracts variables from patient record + conversation
- `fix_variable_extraction()` and `fix_variable_extraction_v2()` apply regex corrections
- Comorbidities overridden from patient record (not LLM-guessed)
- Treatment outcome variables deleted unless confirmed via clarification
- `_default_false_vars` set: emergency/safety flags default to `False`

### Clarification (`deps.py:gpt_clarifier`)
- Calls `get_missing_variables_for_next_step()` to find what's needed at the current decision point
- Only asks about variables needed for the NEXT step (not all variables in the tree)
- Questions tagged with `[var:variable_name]` prefix for answer mapping
- Re-checks after each answer in case new variables are needed

---

## 5. Common Pitfalls Found During Manual Audit

| Problem | Example | Impact |
|---------|---------|--------|
| Evaluator node IDs shifted | NG81 chronic: all 10 IDs were off by 2 | Wrong conditions evaluated at every node |
| Evaluator keys for action nodes | NG91: n4, n6 (actions) had evaluator entries | Engine tries to evaluate action nodes as conditions |
| Phantom evaluator entries | NG91: n10-n20 didn't exist | Harmless but confusing; indicates copy-paste errors |
| Disconnected subgraphs | NG184: 10 separate fragments | BFS explores all fragments, mixing unrelated actions |
| Swapped evaluator conditions | NG91: n1 had n6's logic and vice versa | Patients routed to wrong clinical pathway |
| Cycle with no root | NG81 chronic: n17→n1 back-edge | No node has in-degree 0, BFS falls back to first node |
| OR evaluation returning None | FeverPAIN=4 but Centor missing → OR(False, None) | Fixed: now returns False (not None) when at least one branch is definitively False |

---

## 6. Suggested Auto-Fix Script Structure

```python
def validate_and_fix_guideline(guideline_path, evaluator_path):
    """Validate a guideline + evaluator pair and return issues + fixes."""

    guideline = load_json(guideline_path)
    evaluator = load_json(evaluator_path)

    issues = []
    fixes = []

    # 1. Check connectivity
    roots, unreachable = check_connectivity(guideline)
    if unreachable:
        issues.append(f"Disconnected: {len(unreachable)} unreachable nodes")
        # Suggest connecting edges based on node text analysis

    # 2. Check single root
    if len(roots) != 1:
        issues.append(f"Multiple roots: {roots}")
    if len(roots) == 0:
        issues.append("No root (cycle detected)")

    # 3. Check evaluator keys
    for key in evaluator:
        node = find_node(guideline, key)
        if not node:
            issues.append(f"Phantom evaluator entry: {key}")
            fixes.append(("remove_evaluator_key", key))
        elif node["type"] != "condition":
            issues.append(f"Evaluator key {key} is {node['type']}, not condition")
            fixes.append(("remove_evaluator_key", key))

    # 4. Check all conditions have evaluators
    for node in guideline["nodes"]:
        if node["type"] == "condition" and node["id"] not in evaluator:
            issues.append(f"Condition {node['id']} missing from evaluator")
            # Auto-generate: derive variable name from node text
            var_name = text_to_variable_name(node["text"])
            fixes.append(("add_evaluator", node["id"], {"variable": var_name}))

    # 5. Check edge references
    # 6. Check condition nodes have outgoing edges
    # ... etc.

    return issues, fixes
```

The script should:
1. **Validate** — report all issues without modifying files
2. **Fix** — apply automatic fixes where safe (remove phantoms, add missing evaluators)
3. **Flag** — highlight issues that need human review (disconnected subgraphs, wrong variable mappings)

For connectivity fixes, the script can suggest edges but should not auto-apply them without review, since connecting subgraphs requires understanding the clinical logic.
