from src.llm.guardrails.rules import AI_SUGGESTION_DISCLAIMER
from src.llm.service.doctor_view import BLOCKED_RESPONSE_MESSAGE, to_doctor_view


def test_blocked_response_never_leaks_raw_text():
    state = {
        "guardrail_approved": False,
        "guardrail_reason": "Resposta bloqueada: contem linguagem de prescricao/ordem direta",
        "response": {"response_text": "RESPOSTA CRUA QUE NAO DEVERIA APARECER", "sources": ["Fonte X"]},
        "pending_exams": [],
    }
    view = to_doctor_view(state)
    assert view.response_text == BLOCKED_RESPONSE_MESSAGE
    assert "CRUA" not in view.response_text
    assert view.sources == []


def test_approved_response_passes_through_sources_and_text():
    state = {
        "guardrail_approved": True,
        "response": {"response_text": "Resposta ok", "sources": ["FAQ - X"]},
        "pending_exams": [],
        "exam_notes": "Nenhum exame pendente.",
    }
    view = to_doctor_view(state)
    assert view.response_text == "Resposta ok"
    assert view.sources == ["FAQ - X"]
    assert view.pending_exam_notice is None


def test_pending_exam_notice_only_appears_when_exams_are_pending():
    state = {
        "guardrail_approved": True,
        "response": {"response_text": "Resposta ok", "sources": []},
        "pending_exams": [{"exam_name": "Hemograma"}],
        "exam_notes": "Atencao: exame pendente (Hemograma).",
    }
    view = to_doctor_view(state)
    assert view.pending_exam_notice == "Atencao: exame pendente (Hemograma)."


def test_internal_fields_never_reach_the_doctor_view():
    state = {
        "guardrail_approved": True,
        "response": {
            "response_text": "Resposta ok",
            "sources": [],
            "confidence_level": 0.9,
            "requires_human_validation": True,
        },
        "pending_exams": [],
        "confidence_level": 0.9,
        "guardrail_warnings": ["algum aviso interno"],
        "medical_team_alert": "alerta interno",
        "pending_human_validation": True,
    }
    view = to_doctor_view(state)
    assert not hasattr(view, "confidence_level")
    assert not hasattr(view, "guardrail_warnings")
    assert not hasattr(view, "medical_team_alert")
    assert not hasattr(view, "pending_human_validation")


def test_ai_disclaimer_is_stripped_from_approved_response():
    state = {
        "guardrail_approved": True,
        "response": {
            "response_text": f"Resposta ok{AI_SUGGESTION_DISCLAIMER}",
            "sources": [],
        },
        "pending_exams": [],
    }
    view = to_doctor_view(state)
    assert view.response_text == "Resposta ok"
    assert "validacao humana" not in view.response_text.lower()
