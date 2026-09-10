from pathlib import Path

from src.llm.data_prep.build_fine_tuning_dataset import (
    FINE_TUNING_ONLY_CONDITIONS,
    RAW_DIR,
    _finetuning_only_examples,
)

RAW_FILES = ("protocolos_internos.json", "faqs_medicos.json", "modelos_laudos.json")


def test_finetuning_only_examples_cover_every_configured_condition():
    examples = _finetuning_only_examples()
    assert examples, "Deveria gerar ao menos um exemplo por condicao configurada."

    for case in FINE_TUNING_ONLY_CONDITIONS:
        matching = [ex for ex in examples if case["condition"] in ex["input"] or case["condition"] in ex["output"]]
        assert matching, f"Nenhum exemplo gerado para a condicao exclusiva '{case['condition']}'."


def test_finetuning_only_conditions_are_absent_from_rag_raw_data():
    """The whole point of these examples is that HospitalKnowledgeBase (RAG,
    which only reads data/raw/ - see src/llm/rag/vector_store.py) has zero
    documents about them. If this ever starts failing, something started
    writing the condition into data/raw/, which would silently break the
    base-vs-fine-tuned comparison (src/llm/fine_tuning/compare_base_vs_finetuned.py)
    that depends on RAG having nothing to retrieve here."""
    raw_contents = []
    for filename in RAW_FILES:
        path = RAW_DIR / filename
        if path.exists():
            raw_contents.append(path.read_text(encoding="utf-8"))
    combined_raw_text = "\n".join(raw_contents)

    for case in FINE_TUNING_ONLY_CONDITIONS:
        assert case["condition"] not in combined_raw_text, (
            f"'{case['condition']}' foi encontrada em data/raw/ - deixou de ser exclusiva do "
            "fine-tuning."
        )
