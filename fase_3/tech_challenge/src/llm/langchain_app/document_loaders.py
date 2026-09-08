"""
Document Loaders for the structured patient base (challenge requirement 2:
"query structured databases, such as medical records and registries"),
following the same spirit as Aula 02 - Document Loaders of the reference
material (PyPDFLoader, CSVLoader etc.), adapted to our SQLite base + plain
text medical record files.

The SQLite connection is always opened read-only (`mode=ro`): the assistant
never writes to the patient base, and this also avoids locking issues in
cloud-synced folders (see README).
"""
# Allows `Type | None` (PEP 604) on Python 3.9, which only supports that
# syntax natively from 3.10 onward.
from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parents[3]
DB_PATH = BASE_DIR / "data" / "db" / "hospital.sqlite3"
CSV_PATH = BASE_DIR / "data" / "processed" / "pacientes.csv"
MEDICAL_RECORDS_DIR = BASE_DIR / "data" / "processed" / "prontuarios"


def _connect_read_only() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class PatientRecordsLoader:
    """LangChain-style loader that reads pending exams and medical record
    history for a given patient from the SQLite base."""

    def __init__(self, patient_id: str | None = None):
        self.patient_id = patient_id

    def load(self) -> list[Document]:
        conn = _connect_read_only()
        query = "SELECT * FROM medical_record_events"
        params: tuple = ()
        if self.patient_id:
            query += " WHERE patient_id = ?"
            params = (self.patient_id,)
        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [
            Document(
                page_content=row["description"],
                metadata={
                    "patient_id": row["patient_id"],
                    "event_date": row["event_date"],
                    "source": "medical_record_events",
                },
            )
            for row in rows
        ]


def load_patient(patient_id: str) -> dict | None:
    """Returns the patient's own record (name, age, sex, main_condition,
    admission_date) - used by MedicalAssistantChain to ground the RAG search
    query in the patient's known condition (see chains.py::invoke), not just
    the doctor's free-text question."""
    conn = _connect_read_only()
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def load_pending_exams(patient_id: str) -> list[dict]:
    conn = _connect_read_only()
    rows = conn.execute(
        "SELECT * FROM exams WHERE patient_id = ? AND status = 'pendente'", (patient_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_medical_record_history(patient_id: str) -> list[dict]:
    conn = _connect_read_only()
    rows = conn.execute(
        "SELECT * FROM medical_record_events WHERE patient_id = ? ORDER BY event_date", (patient_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_medical_record_text(patient_id: str) -> Document | None:
    path = MEDICAL_RECORDS_DIR / f"{patient_id}.txt"
    if not path.exists():
        return None
    return TextLoader(str(path), encoding="utf-8").load()[0]


def load_patients_csv() -> list[Document]:
    return CSVLoader(str(CSV_PATH), encoding="utf-8").load()


def list_patients() -> list[dict]:
    """Used by the doctor-facing service (src/llm/service/app.py, GET
    /patients) to populate the patient selector - the same query
    run_demo.py's --listar-pacientes prints to the terminal."""
    conn = _connect_read_only()
    rows = conn.execute(
        "SELECT patient_id, name, main_condition FROM patients"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
