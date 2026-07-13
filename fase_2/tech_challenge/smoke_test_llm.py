"""Smoke test manual

Uso: python smoke_test_llm.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from fase_2.tech_challenge.src.ga.encoding import n_genes
from fase_2.tech_challenge.src.ga.fitness import build_model
from fase_2.tech_challenge.src.llm.explain import explain_comparison, explain_diagnosis
from fase_2.tech_challenge.src.models.data import load_dataset, train_test_split_default

print("1) Treinando Logistic Regression (config intermediária) no dataset real...")
df = load_dataset()
X_train, X_test, y_train, y_test = train_test_split_default(df)
individual = np.array([0.5] * n_genes("logistic_regression"))
model = build_model("logistic_regression", individual)
model.fit(X_train, y_train)
print("   OK.")

print("\n2) Gerando explicação de diagnóstico para 1 amostra real de teste...")
feature_names = list(X_train.columns)
sample = X_test.iloc[0].to_numpy()
explanation = explain_diagnosis(model, "Logistic Regression", feature_names, sample)
print("   Resposta da API recebida:\n")
print("   ---")
print(explanation)
print("   ---")

print("\n3) Gerando explicação do comparativo GA vs. baseline (números reais de data/processed/ga_results.json)...")
baseline_metrics = {"accuracy": 0.9737, "recall": 0.9767, "precision": 0.9545, "f1": 0.9655}
optimized_metrics = {"accuracy": 0.9386, "recall": 0.9767, "precision": 0.875, "f1": 0.9231}
config = {"population_size": 50, "generations": 15}
comparison = explain_comparison("logistic_regression", baseline_metrics, optimized_metrics, config)
print("   Resposta da API recebida:\n")
print("   ---")
print(comparison)
print("   ---")

print("\nSmoke test concluído com sucesso.")
