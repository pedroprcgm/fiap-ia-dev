"""
Safety and validation guardrails (challenge requirement 3): scope limits for
the assistant, so it never prescribes directly nor replaces the doctor's
clinical judgment without human validation.

These rules run AFTER the response is generated (in "Treatment Suggestion")
and BEFORE any log/alert is considered valid - equivalent to the
"Guardrails" node described in the specification document.
"""
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
