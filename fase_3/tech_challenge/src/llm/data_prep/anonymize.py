"""
Anonymization utilities used before any data enters the fine-tuning dataset
or the RAG index - an explicit requirement of the challenge
("preprocessing, anonymization and curation").

Even though the data is synthetic/public (no real patients), we apply the
same rules we would use with real hospital data, so the pipeline is
production-ready.
"""
import re

# Patterns for personal identifiers commonly found in Portuguese clinical text.
PATTERNS = {
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "phone": re.compile(r"\b(\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "birth_date": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    ),
    "sus_card": re.compile(r"\b\d{15}\b"),
}

REPLACEMENTS = {
    "cpf": "[CPF_REMOVIDO]",
    "phone": "[TELEFONE_REMOVIDO]",
    "email": "[EMAIL_REMOVIDO]",
    "birth_date": "[DATA_REMOVIDA]",
    "sus_card": "[CARTAO_SUS_REMOVIDO]",
}

# Fictional first names used in the synthetic data - masked for safety even
# though they're made up, to validate the business rule.
KNOWN_NAMES = [
    "Joao", "Maria", "Pedro", "Ana", "Carlos", "Fernanda", "Lucas", "Beatriz",
]


def anonymize_text(text: str) -> str:
    """Applies the personal-identifier masking rules to a piece of text."""
    anonymized_text = text
    for key, pattern in PATTERNS.items():
        anonymized_text = pattern.sub(REPLACEMENTS[key], anonymized_text)

    for name in KNOWN_NAMES:
        anonymized_text = re.sub(
            rf"\b{re.escape(name)}\b", "[NOME_REMOVIDO]", anonymized_text
        )

    return anonymized_text


def curate_example(question: str, answer: str, min_length: int = 15) -> bool:
    """
    Simple curation rule: discards question/answer pairs that are too short
    or empty, which likely indicate an extraction/formatting error.
    Returns True if the example should be kept.
    """
    if not question or not answer:
        return False
    if len(answer.strip()) < min_length:
        return False
    return True
