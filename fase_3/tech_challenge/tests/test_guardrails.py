from src.llm.guardrails.rules import append_disclaimer, validate_response


def test_blocks_direct_prescription():
    result = validate_response("O paciente deve tome imediatamente 500mg de X.", 0.9)
    assert result.approved is False
    assert "bloqueada" in result.reason.lower()


def test_approves_institutional_suggestion():
    result = validate_response("Sugere-se iniciar metformina conforme protocolo interno.", 0.7)
    assert result.approved is True
    assert result.warnings == []


def test_warns_on_low_confidence():
    result = validate_response("Sugestao com base em dados limitados.", 0.05)
    assert result.approved is True
    assert any("confianca" in warning.lower() for warning in result.warnings)


def test_disclaimer_is_added_once():
    text = "Sugestao de conduta."
    with_disclaimer = append_disclaimer(text)
    assert "validacao humana" in with_disclaimer.lower()

    double_disclaimer = append_disclaimer(with_disclaimer)
    assert double_disclaimer.count("validacao humana obrigatoria") == 1
