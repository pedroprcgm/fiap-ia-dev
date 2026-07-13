"""Codificação de genes do Algoritmo Genético.

Cada indivíduo é representado como um vetor de floats em [0, 1] (cromossomo real
normalizado). Isso permite que os operadores de cruzamento e mutação sejam genéricos
(não dependem do tipo de hiperparâmetro). A conversão gene -> valor real do
hiperparâmetro é feita por uma `GeneSpec`, de acordo com o tipo declarado:

- "log_float": float em escala logarítmica (ex.: C do SVC/LogReg, que varia em ordens
  de grandeza). `bounds = (min_exp, max_exp)`, valor decodificado = 10 ** exp.
- "int": inteiro dentro de um intervalo fechado. `bounds = (min, max)`.
- "categorical": índice mapeado para uma lista de opções. `bounds = (opcao_0, opcao_1, ...)`.

Ver docs/hyperparameters.md para a justificativa de cada espaço de busca.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str  # "log_float" | "int" | "categorical"
    bounds: Tuple[Any, ...]

    def decode(self, gene: float) -> Any:
        gene = min(max(gene, 0.0), 1.0)  # clip de segurança
        if self.kind == "log_float":
            min_exp, max_exp = self.bounds
            exponent = min_exp + gene * (max_exp - min_exp)
            return float(10 ** exponent)
        if self.kind == "int":
            min_v, max_v = self.bounds
            return int(round(min_v + gene * (max_v - min_v)))
        if self.kind == "categorical":
            options = self.bounds
            idx = min(int(gene * len(options)), len(options) - 1)
            return options[idx]
        raise ValueError(f"GeneSpec.kind desconhecido: {self.kind}")


# ---------------------------------------------------------------------------
# Espaços de busca por família de modelo (ver docs/hyperparameters.md)
# ---------------------------------------------------------------------------

# Logistic Regression: solver fixado em "liblinear" (suporta tanto l1 quanto l2 e é
# adequado para datasets pequenos como este), max_iter fixado em 10000.
LOGREG_SPEC: List[GeneSpec] = [
    GeneSpec("C", "log_float", (-3, 2)),               # C em [1e-3, 1e2]
    GeneSpec("penalty", "categorical", ("l2", "l1")),   # compatível com liblinear
    GeneSpec("class_weight_pos", "int", (1, 10)),       # peso da classe maligna (1)
]

# SVC Linear: kernel fixado em "linear" (o Módulo 1 já testou RBF e descartou por
# piorar o recall).
SVC_SPEC: List[GeneSpec] = [
    GeneSpec("C", "log_float", (-2, 2)),                # C em [1e-2, 1e2]
    GeneSpec("class_weight_pos", "int", (1, 10)),       # peso da classe maligna (1)
]

# Variante da Regressão Logística com `penalty` travado em "l2" (um gene a menos que
# LOGREG_SPEC, sem a escolha categórica l1/l2). `l1` tem uma superfície de otimização
# menos suave (não-diferenciável em zero) e, com features correlacionadas como as deste
# dataset (ex.: radius_mean, perimeter_mean, area_mean), pode zerar coeficientes de
# forma pouco estável entre execuções.
LOGREG_L2_SPEC: List[GeneSpec] = [
    GeneSpec("C", "log_float", (-3, 2)),                # C em [1e-3, 1e2]
    GeneSpec("class_weight_pos", "int", (1, 10)),       # peso da classe maligna (1)
]

MODEL_SPECS: Dict[str, List[GeneSpec]] = {
    "logistic_regression": LOGREG_SPEC,
    "svc_linear": SVC_SPEC,
    "logistic_regression_l2": LOGREG_L2_SPEC,
}


def n_genes(model_family: str) -> int:
    return len(MODEL_SPECS[model_family])


def random_individual(model_family: str, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n_genes(model_family))


def decode_individual(model_family: str, individual: Sequence[float]) -> Dict[str, Any]:
    """Decodifica o cromossomo em um dict de hiperparâmetros "brutos" (nomes internos,
    ex. `class_weight_pos`), que ainda precisa ser traduzido para os kwargs reais do
    sklearn — ver `src/ga/fitness.py::build_model`."""
    spec = MODEL_SPECS[model_family]
    return {gene_spec.name: gene_spec.decode(value) for gene_spec, value in zip(spec, individual)}
