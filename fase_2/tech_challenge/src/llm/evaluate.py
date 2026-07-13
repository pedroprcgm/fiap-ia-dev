"""Avaliação da qualidade das explicações geradas pela LLM.

Requisito explícito do Tech Challenge (item 3): "Avaliar a qualidade das
interpretações geradas". Este módulo não usa outra LLM como "juiz" — aplica uma
checklist objetiva que a pessoa avaliadora preenche ao revisar cada explicação, e
formata o resultado para colar direto no relatório técnico. Uma heurística simples
(`flag_jargon`) faz uma triagem automática antes da revisão humana.
"""
from dataclasses import dataclass
from typing import List, Optional

JARGON_TERMS = [
    "class_weight", "hiperparâmetro", "hiperparametro", "fitness", "cromossomo",
    "algoritmo genético", "algoritmo genetico", "geração", "geracao",
    "f1-score", "f1 score", "overfitting",
]


def flag_jargon(explanation: str) -> List[str]:
    """Sinaliza termos técnicos de ML/otimização que vazaram para o texto — útil
    como primeira triagem automática antes da revisão humana."""
    lower = explanation.lower()
    return [term for term in JARGON_TERMS if term in lower]


@dataclass
class QualityCheck:
    """Uma linha da checklist de avaliação, para uma explicação específica.

    Os 4 campos booleanos são preenchidos manualmente pela pessoa avaliadora ao
    ler a explicação lado a lado com os dados de entrada (essa é a avaliação
    exigida pelo Tech Challenge — não precisa ser automatizada com outra LLM).
    """
    sample_id: str
    explanation: str
    cites_correct_numbers: Optional[bool] = None
    avoids_ml_jargon: Optional[bool] = None
    is_clinically_actionable: Optional[bool] = None
    does_not_invent_diagnosis: Optional[bool] = None
    notes: str = ""

    def auto_prefill(self) -> None:
        """Preenche `avoids_ml_jargon` automaticamente com base em `flag_jargon`,
        deixando os demais campos para avaliação manual."""
        jargon_found = flag_jargon(self.explanation)
        self.avoids_ml_jargon = len(jargon_found) == 0
        if jargon_found:
            self.notes = (self.notes + f" [jargão detectado: {', '.join(jargon_found)}]").strip()

    def as_row(self) -> dict:
        return {
            "amostra": self.sample_id,
            "cita_numeros_corretos": self.cites_correct_numbers,
            "evita_jargao_ml": self.avoids_ml_jargon,
            "acionavel_clinicamente": self.is_clinically_actionable,
            "nao_inventa_diagnostico": self.does_not_invent_diagnosis,
            "notas": self.notes,
        }


def build_report_table(checks: List[QualityCheck]) -> List[dict]:
    """Formata os resultados para a seção de avaliação de qualidade do relatório
    técnico (uma linha por amostra avaliada)."""
    return [c.as_row() for c in checks]
