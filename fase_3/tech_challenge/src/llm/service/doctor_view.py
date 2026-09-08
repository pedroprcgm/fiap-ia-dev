"""
Presentation filter for the doctor-facing UI (see Secao 4 and 5 of
`docs/Documento de Especificacoes - UI do Assistente Medico`).

This is the single choke point responsible for deciding what the assistant's
internal state is safe to show a doctor. No other module downstream of this
one (the Node API, the Angular app) should ever see the raw AssistantState -
they only ever see a DoctorView.

Explicitly NEVER exposed here (per RF07): confidence_level, guardrail_approved,
guardrail_reason, guardrail_warnings, medical_team_alert, pending_human_validation,
run_id, or anything from the audit log. Those keep being written to
logs/audit_log.jsonl exactly as before - they just never reach this function's
output.
"""
from dataclasses import dataclass
from typing import Optional

from src.llm.guardrails.rules import AI_SUGGESTION_DISCLAIMER
from src.llm.langgraph_flow.state import AssistantState

# Neutral, non-technical fallback shown when the guardrail blocks a response
# (Secao 5): never show the raw blocked response_text, never mention
# "guardrail", the internal reason, or any score.
BLOCKED_RESPONSE_MESSAGE = (
    "Nao foi possivel gerar uma sugestao para esta pergunta dentro dos "
    "criterios de seguranca do assistente. Consulte o protocolo interno ou "
    "reformule a pergunta."
)


@dataclass
class DoctorView:
    response_text: str
    sources: list[str]
    pending_exam_notice: Optional[str] = None


def to_doctor_view(final_state: AssistantState) -> DoctorView:
    """Converts the full AssistantState produced by the LangGraph flow into
    the minimal, clean view the doctor is allowed to see."""
    if not final_state.get("guardrail_approved", False):
        return DoctorView(response_text=BLOCKED_RESPONSE_MESSAGE, sources=[])

    response = final_state.get("response", {})
    pending_exam_notice = (
        final_state.get("exam_notes")
        if final_state.get("pending_exams")
        else None
    )

    return DoctorView(
        response_text=_strip_ai_disclaimer(response.get("response_text", "")),
        sources=list(response.get("sources", [])),
        pending_exam_notice=pending_exam_notice,
    )


def _strip_ai_disclaimer(response_text: str) -> str:
    """Removes the human-validation disclaimer that
    src/llm/guardrails/rules.py::append_disclaimer bakes into response_text.

    That disclaimer is exactly the kind of "alerta de validacao humana" the
    UI spec (RF07) says must never reach the doctor - it stays in the raw
    state (and therefore in the CLI output and the audit log) but is cut
    here, at the single choke point this module exists for."""
    return response_text.replace(AI_SUGGESTION_DISCLAIMER, "").rstrip()
