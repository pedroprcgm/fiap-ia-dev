"""
Internal HTTP service that exposes the medical assistant's LangGraph flow to
the Node.js API (src/api) - see Secao 7 of
`docs/Documento de Especificacoes - UI do Assistente Medico`.

This service is NOT meant to be reachable from the browser: only the Node
API calls it, over localhost. It keeps the RAG index and the domain model
loaded in memory for the lifetime of the process (building them fresh on
every request would be far too slow), which is why this is a long-running
service instead of a script invoked per-request.

Run with:
    uvicorn src.llm.service.app:app --port 8001

Endpoints (mirrored by the public ones on the Node API):
    GET  /patients
    POST /ask   {"patient_id": "...", "question": "..."}
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.llm.langchain_app.document_loaders import list_patients
from src.llm.langgraph_flow.graph import build_graph
from src.llm.logging_utils.audit_log import new_run_id
from src.llm.service.doctor_view import to_doctor_view

app = FastAPI(title="Assistente Medico Virtual - servico interno")

# Built once at process startup (importing src.llm.langgraph_flow.nodes -
# which build_graph() triggers - already constructs the RAG knowledge base
# and the domain LLM as module-level singletons). Every request reuses this
# same compiled graph instead of rebuilding it.
_graph = build_graph()


class AskRequest(BaseModel):
    patient_id: str
    question: str


class AskResponse(BaseModel):
    response_text: str
    sources: list[str]
    pending_exam_notice: Optional[str] = None


@app.get("/patients")
def get_patients() -> list[dict]:
    return list_patients()


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        final_state = _graph.invoke({
            "patient_id": payload.patient_id,
            "question": payload.question,
            "run_id": new_run_id(),
        })
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500 to the Node API
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    doctor_view = to_doctor_view(final_state)
    return AskResponse(
        response_text=doctor_view.response_text,
        sources=doctor_view.sources,
        pending_exam_notice=doctor_view.pending_exam_notice,
    )
