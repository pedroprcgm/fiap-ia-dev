from src.llm.data_prep.anonymize import anonymize_text, curate_example


def test_anonymizes_cpf():
    text = "Paciente com CPF 123.456.789-00 foi atendido."
    result = anonymize_text(text)
    assert "123.456.789-00" not in result
    assert "[CPF_REMOVIDO]" in result


def test_anonymizes_email():
    text = "Contato: joao@example.com"
    result = anonymize_text(text)
    assert "joao@example.com" not in result
    assert "[EMAIL_REMOVIDO]" in result


def test_anonymizes_known_name():
    text = "Maria relatou dor de cabeca."
    result = anonymize_text(text)
    assert "Maria" not in result
    assert "[NOME_REMOVIDO]" in result


def test_curation_discards_short_answer():
    assert curate_example("Pergunta valida?", "ok") is False


def test_curation_accepts_valid_example():
    assert curate_example("Pergunta valida?", "Esta e uma resposta com tamanho suficiente.") is True
