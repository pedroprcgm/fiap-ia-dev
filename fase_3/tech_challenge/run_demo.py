"""
End-to-end demo of the medical assistant: runs the full LangGraph graph for
a patient and a question, and prints each step of the flow - useful both to
validate the pipeline and as a script for the demo video requested by the
challenge (showing the flow running, the contextualized response and the
logs/validation).
"""
# Allows `Type | None` (PEP 604) on Python 3.9, which only supports that
# syntax natively from 3.10 onward.
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.llm.langgraph_flow.graph import build_graph
from src.llm.logging_utils.audit_log import LOG_PATH, new_run_id, read_run_events

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "db" / "hospital.sqlite3"


def _connect_read_only():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def list_patients():
    conn = _connect_read_only()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT patient_id, name, main_condition FROM patients").fetchall()
    conn.close()
    print("Pacientes disponiveis na base sintetica:\n")
    for row in rows:
        print(f"  {row['patient_id']} - {row['name']} ({row['main_condition']})")


def _first_patient_id() -> str:
    conn = _connect_read_only()
    patient_id = conn.execute("SELECT patient_id FROM patients LIMIT 1").fetchone()[0]
    conn.close()
    return patient_id


def run_demo(patient_id: str | None, question: str):
    if not DB_PATH.exists():
        raise SystemExit(
            "Base de pacientes nao encontrada. Rode antes:\n"
            "  python -m src.data_prep.generate_synthetic_hospital_data\n"
            "  python -m src.data_prep.prepare_public_datasets\n"
            "  python -m src.data_prep.build_fine_tuning_dataset\n"
            "  python -m src.data_prep.generate_synthetic_patients\n"
            "  python -m src.fine_tuning.train_demo_cpu"
        )

    graph = build_graph()
    run_id = new_run_id()

    print("=" * 78)
    print(f"ASSISTENTE MEDICO VIRTUAL - HOSPITAL XPTO  |  execucao {run_id}")
    print("=" * 78)
    print(f"Paciente: {patient_id or '(pergunta geral, sem paciente associado)'}")
    print(f"Pergunta do medico: {question}\n")

    final_state = graph.invoke({
        "patient_id": patient_id,
        "question": question,
        "run_id": run_id,
    })

    print("--- 1) Verificador de Exames ---")
    print(final_state["exam_notes"], "\n")

    print("--- 2) Contexto recuperado via RAG ---")
    for doc in final_state["rag_documents"]:
        print(f"  [{doc['similarity']:.2f}] {doc['source']}")
    print()

    print("--- 3) Sugestao de conduta (LLM de dominio) ---")
    print(final_state["response"]["response_text"], "\n")

    print("--- 4) Guardrails ---")
    print(f"Aprovado: {final_state['guardrail_approved']} | {final_state['guardrail_reason']}")
    if final_state["guardrail_warnings"]:
        print("Avisos:", "; ".join(final_state["guardrail_warnings"]))
    print()

    print("--- 5) Alertas / validacao humana ---")
    print(final_state["medical_team_alert"], "\n")

    print("--- Fontes citadas (explainability) ---")
    for source in final_state["response"]["sources"]:
        print(f"  - {source}")
    print(f"\nNivel de confianca: {final_state['response']['confidence_level']}")

    print(f"\nLog de auditoria completo desta execucao em: {LOG_PATH}")
    print(f"({len(read_run_events(run_id))} eventos registrados para a execucao {run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paciente", default=None, help="ID do paciente (ex.: PAC0001)")
    parser.add_argument("--pergunta", default="Qual a conduta recomendada para este paciente?")
    parser.add_argument("--listar-pacientes", action="store_true")
    parser.add_argument(
        "--sem-paciente",
        action="store_true",
        help="Pergunta geral, sem associar a um paciente (ver Tela 1 da UI).",
    )
    args = parser.parse_args()

    if args.listar_pacientes:
        list_patients()
    elif args.sem_paciente:
        run_demo(None, args.pergunta)
    else:
        patient_id = args.paciente or _first_patient_id()
        run_demo(patient_id, args.pergunta)
