"""Função fitness: decodifica um indivíduo em um modelo sklearn, treina, avalia e
combina as métricas em um único score escalar que o GA maximiza.
"""
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fase_2.tech_challenge.src.ga.encoding import decode_individual

warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul.*",
    category=RuntimeWarning,
)

# Pesos da combinação linear das métricas. Recall da classe maligna (1) tem o maior
# peso porque, no contexto de diagnóstico médico, um falso negativo (dizer que um
# tumor maligno é benigno) é o erro mais custoso — ver docs/hyperparameters.md.
METRIC_WEIGHTS = {"accuracy": 0.3, "recall": 0.5, "f1": 0.2}


@dataclass
class EvalResult:
    accuracy: float
    recall: float
    precision: float
    f1: float
    score: float
    params: Dict[str, Any]


def build_model(model_family: str, individual) -> Any:
    raw = decode_individual(model_family, individual)
    class_weight = {0: 1, 1: raw["class_weight_pos"]}

    if model_family == "logistic_regression":
        classifier = LogisticRegression(
            C=raw["C"],
            penalty=raw["penalty"],
            solver="liblinear",
            class_weight=class_weight,
            max_iter=10000,
            random_state=42,
        )
    elif model_family == "svc_linear":
        classifier = SVC(
            kernel="linear",
            C=raw["C"],
            class_weight=class_weight,
            random_state=42,
        )
    elif model_family == "logistic_regression_l2":
        # mesma configuração de GA que "logistic_regression", mas com penalty em l2
        classifier = LogisticRegression(
            C=raw["C"],
            penalty="l2",
            solver="liblinear",
            class_weight=class_weight,
            max_iter=10000,
            random_state=42,
        )
    else:
        raise ValueError(f"model_family desconhecida: {model_family}")

    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def _weighted_score(accuracy: float, recall: float, f1: float) -> float:
    return (
        METRIC_WEIGHTS["accuracy"] * accuracy
        + METRIC_WEIGHTS["recall"] * recall
        + METRIC_WEIGHTS["f1"] * f1
    )


def evaluate_individual(
    model_family: str,
    individual,
    X,
    y,
    cv_folds: int = 5,
    random_state: int = 42,
) -> EvalResult:
    """Avalia um indivíduo por k-fold cross-validation estratificada sobre (X, y).

    Cada indivíduo é treinado e avaliado `cv_folds` vezes, em partições diferentes
    estratificadas para preservar a proporção maligno/benigno em cada fold; as
    métricas finais são a média entre os folds.

    `X`/`y` aqui é o conjunto de treino completo disponível para o GA (nunca o X_test
    final, held-out, usado na comparação com o baseline do Módulo 1).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    accuracies, recalls, precisions, f1s = [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(model_family, individual)
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)

        accuracies.append(accuracy_score(y_fold_val, y_pred))
        recalls.append(recall_score(y_fold_val, y_pred, zero_division=0))
        precisions.append(precision_score(y_fold_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_fold_val, y_pred, zero_division=0))

    accuracy = float(np.mean(accuracies))
    recall = float(np.mean(recalls))
    precision = float(np.mean(precisions))
    f1 = float(np.mean(f1s))
    score = _weighted_score(accuracy, recall, f1)

    return EvalResult(
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        f1=f1,
        score=score,
        params=decode_individual(model_family, individual),
    )


def fitness_function(
    model_family: str, individual, X, y, cv_folds: int = 5, random_state: int = 42
) -> float:
    """Atalho usado pelo loop do GA: retorna apenas o score escalar (média k-fold)."""
    return evaluate_individual(model_family, individual, X, y, cv_folds, random_state).score
