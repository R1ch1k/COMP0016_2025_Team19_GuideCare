
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.orchestration.state import ConversationState
from app.orchestration.utils import log_step, with_retry_timeout


def _infer_symptoms(state: ConversationState) -> str:
    return state.get("last_user_message") or state.get("current_symptoms") or ""


def build_graph(deps):
    # ---- node definitions (closures capture deps) ----

    async def load_patient(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")

        # Skip if patient already loaded (follow-up turn)
        if state.get("patient_record"):
            log_step(cid, "load_patient_skip", reason="already_loaded")
            return {}

        log_step(cid, "load_patient_start", patient_id=state.get("patient_id"))

        rec = await with_retry_timeout(
            deps["fetch_patient"],
            state["patient_id"],
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        log_step(cid, "load_patient_done")
        return {"patient_record": rec}

    async def triage(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")

        # Skip triage on follow-up turns if we already triaged on a previous turn.
        # The graph may not have reached select_guideline yet (e.g. stopped at
        # clarify to ask a question), but triage_result proves we already ran triage.
        # Re-triaging on just the clarification answer text loses the original
        # symptom context and picks the wrong guideline.
        if state.get("triage_result"):
            log_step(cid, "triage_skip", reason="already_triaged",
                     suggested=state.get("triage_result", {}).get("suggested_guideline"))
            return {}

        symptoms = _infer_symptoms(state)
        history_window = (state.get("conversation_history") or [])[-settings.MODEL_HISTORY_MAX_MESSAGES :]

        log_step(cid, "triage_start", symptoms=symptoms)

        triage_result = await with_retry_timeout(
            deps["triage_agent"],
            symptoms,
            history_window,
            state.get("patient_record", {}),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        urgent = triage_result.get("urgency") in {"emergency", "high", "999", "ed_now"}
        log_step(cid, "triage_done", urgency=triage_result.get("urgency"), urgent=urgent)

        return {
            "triage_result": triage_result,
            "urgent_escalation": bool(urgent),
            "current_symptoms": symptoms,
        }

    async def clarify(state: ConversationState) -> dict:
        """
        WebSocket-friendly clarification loop (no interrupt()):
        - ask one question -> END (wait for user answer)
        - next user turn provides answer -> ask next pending question
        - after all questions answered, re-check for more missing vars
        - no artificial round cap — keeps going until the tree can advance
        """
        cid = state.get("conversation_id", "unknown")
        history_window = (state.get("conversation_history") or [])[-settings.MODEL_HISTORY_MAX_MESSAGES :]
        pending = list(state.get("clarification_questions") or [])
        answers = dict(state.get("clarification_answers") or {})
        awaiting = bool(state.get("awaiting_clarification_answer", False))

        # Ask next question and stop the graph
        if pending and not awaiting:
            q = pending[0]
            # Strip [var:...] tag for display — keep it in state for answer parsing
            import re as _re
            display_q = _re.sub(r"^\[var:\w+\]\s*", "", q)
            log_step(cid, "clarify_ask", question=display_q)
            return {
                "awaiting_clarification_answer": True,
                "assistant_event": {
                    "type": "clarification_question",
                    "content": display_q,
                    "meta": {"question": q},  # keep tagged version in meta
                },
            }

        # Consume answer from current user turn
        if pending and awaiting:
            q = pending[0]
            a = state.get("last_user_message", "")
            answers[q] = a
            remaining = pending[1:]

            log_step(cid, "clarify_answer", question=q, remaining=len(remaining))

            if remaining:
                # More pre-generated questions to ask
                return {
                    "clarification_answers": answers,
                    "clarification_questions": remaining,
                    "awaiting_clarification_answer": False,
                    "clarification_needed": True,
                    "assistant_event": {},  # clear event
                }

            # All pre-generated questions answered — fall through to re-check
            # whether more variables are still missing (e.g. user said "yes ABPM
            # was done" but didn't provide the actual BP reading).

        # Generate (or re-generate) clarification questions based on what's
        # still missing after incorporating previous answers.
        log_step(cid, "clarify_generate_start")
        result = await with_retry_timeout(
            deps["gpt_clarifier"],
            state.get("current_symptoms", ""),
            history_window,
            state.get("patient_record", {}),
            state.get("triage_result", {}),
            answers,
            state.get("selected_guideline", ""),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        questions = result.get("questions") or []
        if result.get("done") or not questions:
            log_step(cid, "clarify_done", needed=False)
            return {
                "clarification_needed": False,
                "clarification_questions": [],
                "clarification_answers": answers,
                "awaiting_clarification_answer": False,
                "assistant_event": {},
            }

        log_step(cid, "clarify_done", needed=True, count=len(questions))
        return {
            "clarification_needed": True,
            "clarification_questions": questions,
            "clarification_answers": answers,
            "awaiting_clarification_answer": False,
            "assistant_event": {},
        }

    async def select_guideline(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")

        # Preserve guideline from previous turn (don't re-select mid-conversation)
        existing = state.get("selected_guideline")
        if existing:
            log_step(cid, "select_guideline_skip", reason="already_selected",
                     guideline=existing)
            return {"selected_guideline": existing}

        log_step(cid, "select_guideline_start")

        guideline = await with_retry_timeout(
            deps["select_guideline"],
            state.get("current_symptoms", ""),
            state.get("triage_result", {}),
            state.get("clarification_answers", {}),
            state.get("patient_record", {}),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        log_step(cid, "select_guideline_done", guideline=guideline)
        return {"selected_guideline": guideline}

    async def extract_variables(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")
        history_window = (state.get("conversation_history") or [])[-settings.MODEL_HISTORY_MAX_MESSAGES :]
        log_step(cid, "extract_variables_start", guideline=state.get("selected_guideline"))

        vars_ = await with_retry_timeout(
            deps["extract_variables_20b"],
            state["selected_guideline"],
            history_window,
            state.get("patient_record", {}),
            state.get("clarification_answers", {}),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        log_step(cid, "extract_variables_done", var_count=len(vars_ or {}))
        return {"extracted_variables": vars_ or {}}

    async def walk_graph(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")
        log_step(cid, "walk_graph_start", guideline=state.get("selected_guideline"))

        out = await with_retry_timeout(
            deps["walk_guideline_graph"],
            state.get("selected_guideline", ""),
            state.get("extracted_variables", {}),
            state.get("current_node"),
            state.get("pathway_walked", []),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        # Track walk count to prevent infinite clarify→walk loops
        out["_walk_graph_count"] = (state.get("_walk_graph_count") or 0) + 1

        log_step(cid, "walk_graph_done", terminal=out.get("terminal", False))
        return out

    async def format_output(state: ConversationState) -> dict:
        cid = state.get("conversation_id", "unknown")
        guideline = state.get("selected_guideline") or "NICE"
        log_step(cid, "format_output_start", guideline=guideline)

        if state.get("urgent_escalation"):
            triage = state.get("triage_result") or {}
            urgency = triage.get("urgency", "emergency").upper()
            reason = triage.get("reasoning") or triage.get("reason") or "Clinical assessment indicates potential emergency."
            suggested = triage.get("suggested_guideline") or ""

            rec_parts = [
                f"**URGENT — {urgency} PRIORITY**\n",
                "This patient's presentation requires **immediate clinical review**.\n",
            ]
            if reason:
                rec_parts.append(f"**Reason:** {reason}\n")
            rec_parts.append("**Recommended actions:**")
            rec_parts.append("- Escalate to senior clinician or emergency team immediately")
            rec_parts.append("- Do NOT delay treatment pending further investigation")
            rec_parts.append("- Follow local emergency protocols")
            if suggested:
                rec_parts.append(f"\nRefer to NICE {suggested} for clinical pathway guidance.")

            rec = "\n".join(rec_parts)
            log_step(cid, "format_output_done", urgent=True)
            return {
                "final_recommendation": rec,
                "citation": suggested or guideline,
                "urgent_escalation": True,
            }

        out = await with_retry_timeout(
            deps["format_output_20b"],
            guideline,
            state.get("triage_result", {}),
            state.get("extracted_variables", {}),
            state.get("pathway_walked", []),
            state.get("patient_record", {}),
            timeout=settings.AI_TIMEOUT_SECONDS,
            retries=settings.AI_RETRIES,
        )

        log_step(cid, "format_output_done", urgent=False)
        return {
            "final_recommendation": out.get("final_recommendation", ""),
            "citation": out.get("citation", guideline),
        }

    # ---- conditional routers ----

    def after_triage(state: ConversationState) -> str:
        return "format_output" if state.get("urgent_escalation") else "select_guideline"

    def after_clarify(state: ConversationState) -> str:
        # If a question was emitted, stop and wait for next user message
        if (state.get("assistant_event") or {}).get("type") == "clarification_question":
            return "end"
        if state.get("clarification_needed"):
            return "clarify"
        return "extract_variables"

    def after_walk_graph(state: ConversationState) -> str:
        # If graph traversal hit missing variables, loop back to clarify
        # so the system can ask the user about them.
        # Only retry once to prevent infinite loops if clarification can't
        # resolve the missing data.
        missing = state.get("missing_variables") or []
        walk_count = state.get("_walk_graph_count") or 1
        log_step(state.get("conversation_id", "?"), "after_walk_graph",
                 missing=missing, walk_count=walk_count,
                 decision="clarify" if (missing and walk_count <= 1) else "format_output")
        if missing and walk_count <= 1:
            return "clarify"
        return "format_output"

    # ---- build graph ----

    sg = StateGraph(ConversationState)

    sg.add_node("load_patient", load_patient)
    sg.add_node("triage", triage)
    sg.add_node("select_guideline", select_guideline)
    sg.add_node("clarify", clarify)
    sg.add_node("extract_variables", extract_variables)
    sg.add_node("walk_graph", walk_graph)
    sg.add_node("format_output", format_output)

    sg.set_entry_point("load_patient")
    sg.add_edge("load_patient", "triage")
    sg.add_conditional_edges("triage", after_triage, {"select_guideline": "select_guideline", "format_output": "format_output"})
    sg.add_edge("select_guideline", "clarify")
    sg.add_conditional_edges(
        "clarify",
        after_clarify,
        {"clarify": "clarify", "extract_variables": "extract_variables", "end": END},
    )
    sg.add_edge("extract_variables", "walk_graph")
    sg.add_conditional_edges(
        "walk_graph",
        after_walk_graph,
        {"clarify": "clarify", "format_output": "format_output"},
    )
    sg.add_edge("format_output", END)

    return sg.compile(checkpointer=MemorySaver())
