"""
Shared graph state (TypedDict), following the same pattern taught in
Aula 01/02 - Introducao ao LangGraph of the reference material: each node
reads and writes fields of this dictionary, and LangGraph takes care of
propagating the state between nodes.
"""
from typing import TypedDict


class AssistantState(TypedDict, total=False):
    # input
    patient_id: str
    question: str
    run_id: str

    # output of the "Exam Verifier" node
    pending_exams: list[dict]
    exam_notes: str

    # output of the "Context (RAG)" node
    rag_documents: list[dict]

    # output of the "Treatment Suggestion" node
    response: dict  # AssistantResponse.model_dump()

    # output of the "Guardrails" node
    guardrail_approved: bool
    guardrail_reason: str
    guardrail_warnings: list[str]

    # output of the "Alerts / Log" node
    pending_human_validation: bool
    medical_team_alert: str
