from src.llm.guardrails.rules import validate_response

# Real example currently in data/raw/pubmedqa_sample.json (see
# src/llm/data_prep/prepare_public_datasets.py) - this is exactly the kind
# of text the demo/retrieval backend could return verbatim for a vague
# question with no RAG context (src/llm/models/domain_llm.py::
# _generate_with_demo_index), which is the real bug this guardrail exists
# to catch.
ENGLISH_PUBMEDQA_EXAMPLE = (
    "Results depicted mitochondrial dynamics in vivo as PCD progresses "
    "within the lace plant, and highlight the correlation of this organelle "
    "with other organelles during developmental PCD. To the best of our "
    "knowledge, this is the first report of mitochondria and chloroplasts "
    "moving on transvacuolar strands to form a ring structure surrounding "
    "the nucleus during developmental PCD."
)


def test_blocks_english_response():
    result = validate_response(ENGLISH_PUBMEDQA_EXAMPLE, 0.9)
    assert result.approved is False
    assert "portugues" in result.reason.lower()


def test_approves_portuguese_clinical_response():
    result = validate_response(
        "Iniciar beta-agonista de curta duracao inalatorio associado a "
        "corticoide sistemico, com reavaliacao em 1 hora conforme protocolo "
        "interno.",
        0.8,
    )
    assert result.approved is True


def test_does_not_block_short_neutral_text():
    # No Portuguese or English marker words at all (e.g. a bare drug/dosage
    # fragment) - should get the benefit of the doubt, not be blocked.
    result = validate_response("Metformina 500mg 2x/dia.", 0.8)
    assert result.approved is True


def test_prescription_language_is_still_checked_first():
    # Forbidden-phrase check must still win over the language check when a
    # response trips both (this one is in Portuguese, so it only trips the
    # forbidden-phrase rule) - reason should mention the prescription block,
    # not the language one.
    result = validate_response("O paciente deve tome imediatamente 500mg de X.", 0.9)
    assert result.approved is False
    assert "bloqueada" in result.reason.lower()
    assert "prescricao" in result.reason.lower() or "ordem direta" in result.reason.lower()
