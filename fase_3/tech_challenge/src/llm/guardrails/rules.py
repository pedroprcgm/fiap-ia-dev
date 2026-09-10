"""
Safety and validation guardrails (challenge requirement 3): scope limits for
the assistant, so it never prescribes directly nor replaces the doctor's
clinical judgment without human validation.

These rules run AFTER the response is generated (in "Treatment Suggestion")
and BEFORE any log/alert is considered valid - equivalent to the
"Guardrails" node described in the specification document.
"""
import re
from dataclasses import dataclass

# Phrases that indicate a direct order/prescription instead of a suggestion -
# if any of these appear, the response is blocked and must be rewritten/reviewed.
FORBIDDEN_PHRASES = [
    "tome imediatamente",
    "prescrevo",
    "injete agora",
    "administre agora",
    "pare de tomar",
    "aumente a dose sem consultar",
]

MIN_CONFIDENCE_NO_WARNING = 0.2

# Portuguese vs. English marker words used by _looks_like_portuguese below -
# not a general-purpose language detector (no new dependency added on
# purpose, matching the rest of this module's deterministic, rule-based
# style), just enough word overlap to catch an obviously wrong-language
# response. This exists because of a real failure mode: the demo/retrieval
# backend (src/llm/models/domain_llm.py::_generate_with_demo_index, used
# when FINE_TUNED_MODEL_PATH points at models/demo_retrieval) can return a
# training example verbatim as the "answer" - and since the fine-tuning
# dataset now includes public PubMedQA/MedQuAD samples in English (see
# src/llm/data_prep/prepare_public_datasets.py), a vague question with no
# RAG context can surface one of those in English. The hospital's assistant
# must always answer the medical team in Portuguese, regardless of backend.
_PORTUGUESE_MARKER_WORDS = {
    "de", "da", "do", "das", "dos", "que", "para", "com", "uma", "um",
    "nao", "não", "os", "na", "no", "ao", "aos", "em", "por", "mais",
    "ou", "se", "foi", "sao", "são", "esta", "está", "sobre", "como",
    "seu", "sua", "ser", "ate", "até",
}
_ENGLISH_MARKER_WORDS = {
    "the", "and", "of", "to", "is", "was", "were", "with", "this",
    "that", "these", "those", "results", "study", "patients", "we",
    "have", "were", "from", "which", "their", "not", "but", "been",
}
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _looks_like_portuguese(text: str) -> bool:
    """Heuristic only (see module note above): counts common Portuguese vs.
    English marker words. Returns True (benefit of the doubt) when there's
    no evidence either way, so it never blocks short/neutral text (numbers,
    single institution names, empty strings) - it only catches responses
    clearly dominated by English function words."""
    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return True

    portuguese_hits = sum(1 for w in words if w in _PORTUGUESE_MARKER_WORDS)
    english_hits = sum(1 for w in words if w in _ENGLISH_MARKER_WORDS)
    if portuguese_hits == 0 and english_hits == 0:
        return True

    return portuguese_hits >= english_hits


@dataclass
class GuardrailResult:
    approved: bool
    reason: str
    warnings: list[str]


def validate_response(response_text: str, confidence_level: float) -> GuardrailResult:
    lower_text = response_text.lower()
    warnings = []

    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower_text:
            return GuardrailResult(
                approved=False,
                reason=(
                    f"Resposta bloqueada: contem linguagem de prescricao/ordem direta "
                    f"('{phrase}'), incompativel com a politica do assistente (sugestao, "
                    f"nunca decisao automatica)."
                ),
                warnings=warnings,
            )

    if not _looks_like_portuguese(response_text):
        return GuardrailResult(
            approved=False,
            reason=(
                "Resposta bloqueada: o texto gerado nao parece estar em portugues - "
                "a politica do assistente exige respostas sempre em portugues para "
                "o time medico, independente do backend/idioma dos dados de treino."
            ),
            warnings=warnings,
        )

    if confidence_level < MIN_CONFIDENCE_NO_WARNING:
        warnings.append(
            f"Confianca baixa ({confidence_level:.2f}) - revisar com atencao redobrada antes de validar."
        )

    return GuardrailResult(approved=True, reason="Resposta dentro dos limites de atuacao.", warnings=warnings)


# Exposed as a module-level constant (instead of a local literal inside
# append_disclaimer) so the doctor-facing presentation filter
# (src/llm/service/doctor_view.py) can strip this exact text back out - the
# UI spec forbids showing the doctor any human-validation alert, even though
# the audit trail and the CLI (run_demo.py) keep it inline in response_text.
AI_SUGGESTION_DISCLAIMER = (
    "\n\n[Esta e uma sugestao gerada por IA, sujeita a validacao humana obrigatoria. "
    "Nao substitui o julgamento clinico do medico responsavel.]"
)


def append_disclaimer(response_text: str) -> str:
    if AI_SUGGESTION_DISCLAIMER.strip() in response_text:
        return response_text
    return response_text + AI_SUGGESTION_DISCLAIMER
