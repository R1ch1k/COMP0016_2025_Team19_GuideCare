from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


ConversationHistory = Annotated[List[Dict[str, Any]], add]


class ConversationState(TypedDict, total=False):
    # identifiers
    conversation_id: str
    patient_id: str
    error: Optional[str]

    # conversation inputs
    patient_record: Dict[str, Any]
    conversation_history: ConversationHistory
    last_user_message: str

    # triage/clarify
    current_symptoms: str
    triage_result: Dict[str, Any]
    urgent_escalation: bool

    clarification_needed: bool
    clarification_questions: List[str]
    clarification_answers: Dict[str, Any]
    awaiting_clarification_answer: bool
    clarification_rounds: int  # deprecated, kept for backwards compat
    clarification_total_asked: int

    # guideline processing
    selected_guideline: str
    extracted_variables: Dict[str, Any]
    current_node: str
    pathway_walked: List[str]
    terminal: bool
    reached_actions: List[str]
    missing_variables: List[str]
    _walk_graph_count: int  # prevents infinite clarify→walk loops

    # output
    final_recommendation: str
    citation: str

    # websocket event payload
    assistant_event: Dict[str, Any]
