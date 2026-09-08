"""
The 5 nodes of the medical assistant graph, per Secao 5.6 of the
specification document:

    Exam Verifier -> Context (RAG) -> Treatment Suggestion
        -> Guardrails -> Alerts / Log

Each function receives the current state (AssistantState) and returns a
dict with the fields it updates - the standard LangGraph node pattern.
"""
from src.llm.guardrails.rules import append_disclaimer, validate_response
from src.llm.langchain_app.agent import ExamVerificationAgent
from src.llm.langchain_app.chains import MedicalAssistantChain
from src.llm.langgraph_flow.state import AssistantState
from src.llm.logging_utils.audit_log import log_event

_exam_agent = ExamVerificationAgent()
_assistant_chain = MedicalAssistantChain()


def exam_verifier_node(state: AssistantState) -> dict:
    result = _exam_agent.act(state["patient_id"])

    log_event(state["run_id"], "exam_verifier", {
        "patient_id": state["patient_id"],
        "pending_exams": result.pending_exams,
        "notes": result.notes,
    })

    return {
        "pending_exams": result.pending_exams,
        "exam_notes": result.notes,
    }


def rag_context_node(state: AssistantState) -> dict:
    documents = _assistant_chain.knowledge_base.search(state["question"], k=3)

    log_event(state["run_id"], "rag_context", {
        "question": state["question"],
        "retrieved_sources": [d["source"] for d in documents],
    })

    return {"rag_documents": documents}


def treatment_suggestion_node(state: AssistantState) -> dict:
    response = _assistant_chain.invoke(state["patient_id"], state["question"])

    log_event(state["run_id"], "treatment_suggestion", {
        "patient_id": state["patient_id"],
        "response": response.model_dump(),
    })

    return {"response": response.model_dump()}


def guardrails_node(state: AssistantState) -> dict:
    response = state["response"]
    result = validate_response(response["response_text"], response["confidence_level"])

    if result.approved:
        response["response_text"] = append_disclaimer(response["response_text"])

    log_event(state["run_id"], "guardrails", {
        "approved": result.approved,
        "reason": result.reason,
        "warnings": result.warnings,
    })

    return {
        "response": response,
        "guardrail_approved": result.approved,
        "guardrail_reason": result.reason,
        "guardrail_warnings": result.warnings,
    }


def alerts_log_node(state: AssistantState) -> dict:
    if not state["guardrail_approved"]:
        alert = (
            f"BLOQUEADO pelos guardrails: {state['guardrail_reason']} "
            f"Encaminhado para revisao humana obrigatoria antes de qualquer uso."
        )
    elif state["pending_exams"]:
        alert = (
            f"Sugestao gerada com exame(s) pendente(s) para o paciente "
            f"{state['patient_id']}. Equipe medica notificada para validar "
            f"a sugestao considerando essa pendencia."
        )
    else:
        alert = (
            f"Sugestao gerada para o paciente {state['patient_id']} e enviada "
            f"para validacao humana (nenhuma acao e tomada automaticamente)."
        )

    log_event(state["run_id"], "alerts_log", {
        "alert": alert,
        "pending_human_validation": True,
    })

    return {
        "pending_human_validation": True,
        "medical_team_alert": alert,
    }
