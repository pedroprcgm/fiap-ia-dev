"""Modelos originais do Módulo 1 (baseline), com os hiperparâmetros exatamente como
definidos no notebook `notebooks/modulo1_cancer_classification.ipynb`.

Este módulo existe para servir de ponto de comparação ("modelo original") contra os
modelos com hiperparâmetros otimizados pelo Algoritmo Genético (Fase 2).

Uso:
    python -m src.models.baseline
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fase_2.tech_challenge.src.models.data import load_dataset, train_test_split_default


@dataclass
class ModelResult:
    name: str
    accuracy: float
    recall: float
    precision: float
    f1: float


# Hiperparâmetros idênticos aos escolhidos manualmente no Módulo 1.
# class_weight={0: 1, 1: 5} prioriza recall da classe maligna (1), decisão tomada no
# Módulo 1 após testar pesos de 1 a 5 (ver docs/decisions.md).
BASELINE_MODELS: Dict[str, Callable[[], Any]] = {
    "SVC Linear": lambda: SVC(kernel="linear", C=2, class_weight={0: 1, 1: 5}),
    # Nota: LinearRegression não é um classificador — é usado aqui apenas para
    # reproduzir fielmente o experimento do Módulo 1 (threshold em 0.5). Gera
    # RuntimeWarnings de overflow/divisão por zero por falta de normalização das
    # features. Não recomendado como candidato à otimização via GA.
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=10000, random_state=42, class_weight={0: 1, 1: 5}
    ),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
}


def _evaluate(y_test, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def run_baseline() -> Dict[str, ModelResult]:
    """Treina e avalia os 5 modelos originais do Módulo 1, retornando as métricas."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split_default(df)

    results: Dict[str, ModelResult] = {}
    for name, build_model in BASELINE_MODELS.items():
        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if name == "Linear Regression":
            y_pred = (y_pred >= 0.5).astype(int)

        metrics = _evaluate(y_test, y_pred)
        results[name] = ModelResult(name=name, **metrics)

    return results


if __name__ == "__main__":
    for result in run_baseline().values():
        print(
            f"{result.name:22s} | acc={result.accuracy:.4f} "
            f"recall={result.recall:.4f} precision={result.precision:.4f} f1={result.f1:.4f}"
        )
