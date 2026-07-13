"""Orquestração: liga os resultados dos modelos (`src/models`, `src/ga`) às
explicações em linguagem natural geradas pela LLM.

Os modelos otimizados são sempre um `Pipeline(scaler -> classificador linear)`
(ver `src/ga/fitness.py::build_model`), então a influência de cada feature nesta
predição específica pode ser calculada diretamente a partir do coeficiente do
classificador multiplicado pelo valor já padronizado da amostra — sem precisar de
uma biblioteca externa de explicabilidade. Isso é o que garante o "grounding": a
LLM nunca inventa quais features importaram, ela só redige em linguagem natural o
que já foi calculado matematicamente aqui.
"""
from typing import Dict, List, Sequence

import numpy as np

from fase_2.tech_challenge.src.llm.client import chat_completion
from fase_2.tech_challenge.src.llm.prompts import build_comparison_prompt, build_diagnosis_prompt


def _predict_label_and_probability(model, sample_2d: np.ndarray):
    """Retorna (classe_prevista, probabilidade_da_classe_prevista).

    LogisticRegression sempre tem `predict_proba`. SVC linear, como configurado em
    `src/ga/fitness.py` (sem `probability=True`, para não pagar o custo extra de
    calibração), não tem — nesse caso aproximamos a confiança a partir da distância
    ao hiperplano (`decision_function`) via sigmoide.
    """
    prediction = int(model.predict(sample_2d)[0])
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        probability = float(model.predict_proba(sample_2d)[0][prediction])
    else:
        decision = float(model.decision_function(sample_2d)[0])
        prob_positive = 1.0 / (1.0 + np.exp(-decision))
        probability = prob_positive if prediction == 1 else 1.0 - prob_positive

    return prediction, probability


def top_features_for_sample(
    model,
    feature_names: Sequence[str],
    sample: np.ndarray,
    top_n: int = 5,
) -> List[Dict]:
    """Para os modelos lineares usados no projeto (Regressão Logística / SVC
    Linear), calcula `coeficiente * valor_padronizado` como a contribuição de cada
    feature para esta predição específica, e retorna as `top_n` mais influentes.
    """
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    sample_scaled = scaler.transform(sample.reshape(1, -1))[0]
    contributions = clf.coef_[0] * sample_scaled

    order = np.argsort(-np.abs(contributions))[:top_n]
    result = []
    for idx in order:
        influence = (
            "aumenta o risco de malignidade"
            if contributions[idx] > 0
            else "reduz o risco de malignidade"
        )
        result.append({
            "name": feature_names[idx],
            "value": float(sample[idx]),
            "influence": influence,
        })
    return result


def explain_diagnosis(
    model,
    model_name: str,
    feature_names: Sequence[str],
    sample: np.ndarray,
    top_n: int = 5,
) -> str:
    """Gera a explicação em linguagem natural para uma única amostra (diagnóstico
    individual) — Requisito 3 do Tech Challenge."""
    sample_2d = sample.reshape(1, -1)
    prediction, probability = _predict_label_and_probability(model, sample_2d)
    label = "Maligno" if prediction == 1 else "Benigno"

    top_features = top_features_for_sample(model, feature_names, sample, top_n=top_n)
    messages = build_diagnosis_prompt(model_name, label, probability, top_features)
    return chat_completion(messages)


def explain_comparison(
    model_family: str,
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    experiment_config: Dict,
) -> str:
    """Gera a explicação em linguagem natural do comparativo GA vs. baseline —
    complementa o Requisito 3 e reaproveita os resultados de `data/processed/ga_results.json`."""
    messages = build_comparison_prompt(
        model_family, baseline_metrics, optimized_metrics, experiment_config
    )
    return chat_completion(messages)
