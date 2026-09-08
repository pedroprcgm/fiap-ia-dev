"""
Exam Verification Agent - follows the perceive -> decide -> act pattern
taught in Aula 06 - Criacao de Agents of the reference material (there, the
agent queries a PostgreSQL action table; here, it queries the patient's
exam SQLite table).

Implemented as a simple class (without LangChain's AgentExecutor), because
the decision here is driven by clear business rules over structured data -
it doesn't need an LLM deciding which tool to call. This same agent is
reused as the first node of the LangGraph graph (src/langgraph_flow/nodes.py).
"""
from dataclasses import dataclass, field

from src.llm.langchain_app.document_loaders import load_pending_exams


@dataclass
class ExamVerificationResult:
    patient_id: str
    pending_exams: list[dict] = field(default_factory=list)
    can_proceed: bool = True
    notes: str = ""


class ExamVerificationAgent:
    """Perceives (queries the exam base), decides (is there a critical
    pending exam?) and acts (signals whether the flow can move on to the
    treatment suggestion or should flag missing exams first)."""

    def perceive(self, patient_id: str) -> list[dict]:
        return load_pending_exams(patient_id)

    def decide(self, pending_exams: list[dict]) -> tuple[bool, str]:
        if not pending_exams:
            return True, "Nenhum exame pendente - pode prosseguir com a sugestao de conduta."
        names = ", ".join(e["exam_name"] for e in pending_exams)
        return True, (
            f"Atencao: exame(s) pendente(s) ({names}). A sugestao de conduta sera "
            f"gerada mesmo assim, mas deve considerar que esses resultados ainda faltam."
        )

    def act(self, patient_id: str) -> ExamVerificationResult:
        pending_exams = self.perceive(patient_id)
        can_proceed, notes = self.decide(pending_exams)
        return ExamVerificationResult(
            patient_id=patient_id,
            pending_exams=pending_exams,
            can_proceed=can_proceed,
            notes=notes,
        )
