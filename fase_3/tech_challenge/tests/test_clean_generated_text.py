from src.llm.models.domain_llm import _clean_generated_text


def test_truncates_at_first_leaked_llama3_control_token():
    # Reproduces the reported bug: the model kept "talking" past its answer,
    # repeating itself across fake extra turns delimited by Llama-3's
    # chat-template control tokens.
    raw = (
        "Para Infeccao do Trato Urinario nao complicada (Infectologia), o "
        "protocolo interno recomenda: Antibioticoterapia empirica de curta "
        "duracao conforme protocolo institucional e resistencia bacteriana "
        "local.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Para Infeccao do Trato Urinario nao complicada (Infectologia), o "
        "protocolo interno recomenda: Antibioticoterapia empirica de curta "
        "duracao conforme protocolo institucional e resistencia bacteriana "
        "local.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    cleaned = _clean_generated_text(raw)

    assert cleaned == (
        "Para Infeccao do Trato Urinario nao complicada (Infectologia), o "
        "protocolo interno recomenda: Antibioticoterapia empirica de curta "
        "duracao conforme protocolo institucional e resistencia bacteriana "
        "local."
    )
    assert "<|" not in cleaned


def test_leaves_clean_text_untouched():
    text = "Uma resposta limpa, sem nenhum token de controle."
    assert _clean_generated_text(text) == text


def test_strips_stray_control_tokens_even_without_repeated_turns():
    raw = "Resposta ok.<|end_of_text|>"
    assert _clean_generated_text(raw) == "Resposta ok."


def test_handles_empty_string():
    assert _clean_generated_text("") == ""
