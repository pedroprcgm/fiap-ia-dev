"""Roda os experimentos do GA (>= 3 configurações diferentes) para os modelos-alvo
(Logistic Regression e SVC Linear), compara os resultados com o baseline do Módulo 1 e
salva um relatório em JSON.
"""
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from fase_2.tech_challenge.src.ga.fitness import build_model
from fase_2.tech_challenge.src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from fase_2.tech_challenge.src.models.baseline import run_baseline
from fase_2.tech_challenge.src.models.data import load_dataset, train_test_split_default

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ga_results.json"

# "logistic_regression_l2" roda com a MESMA config de "logistic_regression" em cada
# experimento — comparação penalty travado em l2 vs. penalty livre (l1/l2).
MODEL_FAMILIES = ["logistic_regression", "logistic_regression_l2", "svc_linear"]

# Três configurações distintas do GA, variando tamanho de população e taxa de mutação,
# conforme exigido pelo enunciado (>= 3 experimentos).
EXPERIMENTS: Dict[str, GAConfig] = {
    "exp1_baseline_ga": GAConfig(
        POPULATION_SIZE=20, GENERATIONS=30, CROSSOVER_RATE=0.8, MUTATION_RATE=0.1
    ),
    "exp2_populacao_grande": GAConfig(
        POPULATION_SIZE=50, GENERATIONS=30, CROSSOVER_RATE=0.8, MUTATION_RATE=0.1
    ),
    "exp3_alta_mutacao": GAConfig(
        POPULATION_SIZE=20, GENERATIONS=30, CROSSOVER_RATE=0.8, MUTATION_RATE=0.35
    ),
}


def _final_test_metrics(model_family: str, individual, X_train, y_train, X_test, y_test) -> dict:
    """Avalia o melhor indivíduo encontrado no conjunto de teste final (held-out),
    o mesmo usado pelo baseline — garante comparação justa otimizado vs. original."""
    model = build_model(model_family, individual)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
    }


def run_all_experiments() -> dict:
    df = load_dataset()
    X_train_full, X_test, y_train_full, y_test = train_test_split_default(df)
    # O fitness de cada indivíduo usa k-fold cross-validation sobre X_train_full/
    # y_train_full inteiro (ver src/ga/fitness.py::evaluate_individual). O X_test final
    # continua 100% de fora, nunca visto pelo GA, para a comparação justa com o
    # baseline do Módulo 1.

    baseline_results = {name: asdict(res) for name, res in run_baseline().items()}

    report = {"baseline": baseline_results, "experiments": {}}

    for exp_name, config in EXPERIMENTS.items():
        report["experiments"][exp_name] = {"config": asdict(config), "models": {}}
        for model_family in MODEL_FAMILIES:
            logger.info("\n=== Experimento %s | modelo %s ===", exp_name, model_family)
            ga = GeneticAlgorithm(model_family, config)
            result = ga.run(X_train_full, y_train_full, live_plot=True, save_plot=True, live_plot_delay=0.1)

            test_metrics = _final_test_metrics(
                model_family, result.best_individual, X_train_full, y_train_full, X_test, y_test
            )
            logger.info(
                "Melhor individuo (%s, %s) -> cv_score=%.4f | teste: %s",
                exp_name, model_family, result.best_eval.score, test_metrics,
            )

            report["experiments"][exp_name]["models"][model_family] = {
                "best_params": result.best_eval.params,
                "cv_score": result.best_eval.score,
                "test_metrics": test_metrics,
                "history": [asdict(h) for h in result.history],
            }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("\nResultados salvos em %s", RESULTS_PATH)

    return report


def print_summary(report: dict) -> None:
    print("\n=== Baseline (Módulo 1, original) ===")
    for name, metrics in report["baseline"].items():
        print(
            f"{name:22s} | acc={metrics['accuracy']:.4f} recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} f1={metrics['f1']:.4f}"
        )

    print("\n=== GA: melhores modelos por experimento (avaliados no teste final) ===")
    for exp_name, exp_data in report["experiments"].items():
        cfg = exp_data["config"]
        print(f"\n-- {exp_name} (pop={cfg['POPULATION_SIZE']}, mut={cfg['MUTATION_RATE']}, gens={cfg['GENERATIONS']}) --")
        for model_family, data in exp_data["models"].items():
            m = data["test_metrics"]
            print(
                f"  {model_family:20s} | acc={m['accuracy']:.4f} recall={m['recall']:.4f} "
                f"precision={m['precision']:.4f} f1={m['f1']:.4f} | params={data['best_params']}"
            )


if __name__ == "__main__":
    report = run_all_experiments()
    print_summary(report)
