"""
Structured logging for tracking and auditing (challenge requirement 3:
"implement detailed logging for tracking and auditing").

Each run of the flow writes one event per LangGraph node to
logs/audit_log.jsonl (JSON Lines format - one object per line, easy to
process later with pandas/jq). We use plain text append mode (no database)
on purpose: it's the most robust format for incremental writes in
cloud-synced folders, with no risk of file locking (see the technical note
in the README).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
LOG_PATH = BASE_DIR / "logs" / "audit_log.jsonl"


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(run_id: str, node: str, data: dict) -> None:
    """Logs an audit event for a graph node (e.g. 'exam_verifier',
    'rag_context', 'treatment_suggestion', 'guardrails', 'alerts_log')."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "data": data,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_run_events(run_id: str) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    events = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event["run_id"] == run_id:
                events.append(event)
    return events
