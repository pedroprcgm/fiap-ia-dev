"""
Generates the synthetic structured patient base for the "Hospital Pos Tech":
registry, exams (pending/completed) and medical record events. This base
simulates the "structured patient data" that the assistant (LangChain layer)
needs to query to contextualize responses, and that the "Exam Verification
Agent" (LangGraph) reads at the start of the flow.

Generated formats:
    data/db/hospital.sqlite3           - relational database (source of truth)
    data/processed/pacientes.csv       - export for use with CSVLoader (LangChain)
    data/processed/prontuarios/*.txt   - one file per patient, for use with
                                          DirectoryLoader/TextLoader (LangChain)

All data is fictional (names, IDs and histories are made up) - there is no
real patient data.
"""
import csv
import random
import shutil
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[3]
DB_PATH = BASE_DIR / "data" / "db" / "hospital.sqlite3"
CSV_PATH = BASE_DIR / "data" / "processed" / "pacientes.csv"
MEDICAL_RECORDS_DIR = BASE_DIR / "data" / "processed" / "prontuarios"

# Nomes ficticios simples (nao ha nenhum paciente real por tras deles - ver
# docstring do modulo), escolhidos para soar naturais numa UI real em vez de
# rotulos abstratos como "Paciente Alfa"/"Paciente Beta". Com acentuacao
# correta: diferente do restante deste arquivo (mantido em ASCII por
# convencao do projeto), o nome nunca passa pelo vetorizador TF-IDF, entao
# nao ha motivo para evitar acentos aqui - e o que aparece de fato na UI.
FICTIONAL_NAMES = [
    "João Silva", "Maria Oliveira", "José Santos", "Ana Pereira",
    "Pedro Costa", "Carla Souza", "Lucas Almeida", "Fernanda Lima",
]

CONDITIONS = [
    "Hipertensao Arterial Sistemica", "Diabetes Mellitus tipo 2",
    "Pneumonia Adquirida na Comunidade", "Doenca do Refluxo Gastroesofagico",
    "Lombalgia Mecanica Aguda", "Infeccao do Trato Urinario nao complicada",
]

EXAMS_BY_CONDITION = {
    "Hipertensao Arterial Sistemica": ["Eletrocardiograma", "Perfil lipidico"],
    "Diabetes Mellitus tipo 2": ["Glicemia de jejum", "Hemoglobina glicada (HbA1c)"],
    "Pneumonia Adquirida na Comunidade": ["Radiografia de torax", "Hemograma completo"],
    "Doenca do Refluxo Gastroesofagico": ["Endoscopia digestiva alta"],
    "Lombalgia Mecanica Aguda": ["Nenhum exame de imagem indicado na fase aguda"],
    "Infeccao do Trato Urinario nao complicada": ["Urina tipo I (EAS)"],
}

EXAM_STATUS = ["pendente", "realizado"]


def _random_date(max_days_ago: int = 30) -> str:
    days = random.randint(0, max_days_ago)
    return (date.today() - timedelta(days=days)).isoformat()


def generate_patients(count: int = 8) -> list[dict]:
    patients = []
    for i, name in enumerate(FICTIONAL_NAMES[:count], start=1):
        condition = random.choice(CONDITIONS)
        patients.append({
            "patient_id": f"PAC{i:04d}",
            "name": name,
            "age": random.randint(28, 82),
            "sex": random.choice(["F", "M"]),
            "main_condition": condition,
            "admission_date": _random_date(15),
        })
    return patients


def generate_exams(patients: list[dict]) -> list[dict]:
    exams = []
    exam_seq = 1
    for patient in patients:
        condition_exams = EXAMS_BY_CONDITION.get(patient["main_condition"], [])
        for exam_name in condition_exams:
            exams.append({
                "exam_id": f"EX{exam_seq:05d}",
                "patient_id": patient["patient_id"],
                "exam_name": exam_name,
                "status": random.choice(EXAM_STATUS),
                "request_date": _random_date(10),
            })
            exam_seq += 1
    return exams


def generate_medical_record_events(patients: list[dict]) -> list[dict]:
    events = []
    event_seq = 1
    for patient in patients:
        events.append({
            "event_id": f"EVT{event_seq:05d}",
            "patient_id": patient["patient_id"],
            "event_date": patient["admission_date"],
            "description": (
                f"Admissao com quadro sugestivo de {patient['main_condition']}. "
                f"Paciente {patient['sex']}, {patient['age']} anos."
            ),
        })
        event_seq += 1
        events.append({
            "event_id": f"EVT{event_seq:05d}",
            "patient_id": patient["patient_id"],
            "event_date": _random_date(5),
            "description": "Reavaliacao clinica em curso, aguardando resultado de exames complementares.",
        })
        event_seq += 1
    return events


def save_sqlite(patients, exams, events):
    # SQLite needs real file locking (random-access), which some
    # network/sync mount points (e.g. cloud-synced folders) don't support
    # well. So we build the database in a local temp file and only copy
    # (plain file copy) the finished file into data/db/ at the end.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.gettempdir()) / "hospital_build.sqlite3"
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            sex TEXT,
            main_condition TEXT,
            admission_date TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE exams (
            exam_id TEXT PRIMARY KEY,
            patient_id TEXT,
            exam_name TEXT,
            status TEXT,
            request_date TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)
    cur.execute("""
        CREATE TABLE medical_record_events (
            event_id TEXT PRIMARY KEY,
            patient_id TEXT,
            event_date TEXT,
            description TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    cur.executemany(
        "INSERT INTO patients VALUES (:patient_id, :name, :age, :sex, :main_condition, :admission_date)",
        patients,
    )
    cur.executemany(
        "INSERT INTO exams VALUES (:exam_id, :patient_id, :exam_name, :status, :request_date)",
        exams,
    )
    cur.executemany(
        "INSERT INTO medical_record_events VALUES (:event_id, :patient_id, :event_date, :description)",
        events,
    )
    conn.commit()
    conn.close()

    # Overwrites the destination file's contents instead of delete+recreate:
    # in synced folders (OneDrive) or network mounts, a file that has
    # already been opened by a sqlite connection can end up with a residual
    # lock that blocks unlink()/remove(), even after the process has
    # finished. Overwriting the contents (shutil.copy) works fine.
    shutil.copy(tmp_path, DB_PATH)
    tmp_path.unlink()


def save_csv(patients):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(patients[0].keys()))
        writer.writeheader()
        writer.writerows(patients)


def save_medical_records_text(patients, exams, events):
    MEDICAL_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    for patient in patients:
        pid = patient["patient_id"]
        patient_exams = [e for e in exams if e["patient_id"] == pid]
        patient_events = [e for e in events if e["patient_id"] == pid]

        lines = [
            f"PRONTUARIO - {pid}",
            f"Nome: {patient['name']}",
            f"Idade: {patient['age']} | Sexo: {patient['sex']}",
            f"Condicao principal: {patient['main_condition']}",
            f"Data de admissao: {patient['admission_date']}",
            "",
            "Exames:",
        ]
        for exam in patient_exams:
            lines.append(f"  - {exam['exam_name']}: {exam['status']} (solicitado em {exam['request_date']})")

        lines.append("")
        lines.append("Historico de eventos:")
        for event in patient_events:
            lines.append(f"  - {event['event_date']}: {event['description']}")

        (MEDICAL_RECORDS_DIR / f"{pid}.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    patients = generate_patients()
    exams = generate_exams(patients)
    events = generate_medical_record_events(patients)

    save_sqlite(patients, exams, events)
    save_csv(patients)
    save_medical_records_text(patients, exams, events)

    print(f"Gerados {len(patients)} pacientes, {len(exams)} exames e {len(events)} eventos de prontuario.")
    print(f"  - SQLite: {DB_PATH}")
    print(f"  - CSV: {CSV_PATH}")
    print(f"  - Prontuarios em texto: {MEDICAL_RECORDS_DIR}")


if __name__ == "__main__":
    main()
