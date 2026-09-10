import sqlite3
from pathlib import Path

import pytest

from src.llm.langgraph_flow.graph import build_graph
from src.llm.logging_utils.audit_log import new_run_id, read_run_events

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "db" / "hospital.sqlite3"


def _first_patient_id() -> str:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    patient_id = conn.execute("SELECT patient_id FROM patients LIMIT 1").fetchone()[0]
    conn.close()
    return patient_id


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def test_full_flow_produces_response_with_human_validation(graph):
    patient_id = _first_patient_id()
    run_id = new_run_id()

    final_state = graph.invoke({
        "patient_id": patient_id,
        "question": "Qual a conduta recomendada para este paciente?",
        "run_id": run_id,
    })

    assert final_state["response"]["response_text"]
    assert final_state["response"]["sources"]
    assert final_state["guardrail_approved"] is True
    # Regra inegociável do desafio: toda sugestão passa por validação humana.
    assert final_state["pending_human_validation"] is True
    assert final_state["response"]["requires_human_validation"] is True


def test_general_question_with_no_patient_still_produces_response(graph):
    """The doctor can ask a general question with no patient selected (Tela
    1 of the UI now offers this alongside the patient list) - patient_id is
    simply None/absent, and every patient-specific node (exam_verifier,
    treatment_suggestion's chains.py::invoke) should skip its lookups
    instead of failing."""
    run_id = new_run_id()

    final_state = graph.invoke({
        "patient_id": None,
        "question": "Qual o protocolo interno para Hipertensao Arterial Sistemica?",
        "run_id": run_id,
    })

    assert final_state["response"]["response_text"]
    assert final_state["pending_exams"] == []
    assert final_state["pending_human_validation"] is True
    assert "geral" in final_state["medical_team_alert"].lower()


def test_all_nodes_log_audit_events(graph):
    patient_id = _first_patient_id()
    run_id = new_run_id()

    graph.invoke({
        "patient_id": patient_id,
        "question": "Quais exames estao pendentes?",
        "run_id": run_id,
    })

    events = read_run_events(run_id)
    expected_nodes = {
        "exam_verifier", "rag_context", "treatment_suggestion", "guardrails", "alerts_log",
    }
    assert expected_nodes.issubset({e["node"] for e in events})
