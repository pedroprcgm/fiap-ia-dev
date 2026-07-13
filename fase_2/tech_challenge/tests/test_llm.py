"""Testes do módulo src/llm — nunca chamam a API real da OpenAI (sempre mockada
via monkeypatch), para manter a suíte rápida, determinística e sem custo."""
import numpy as np
import pytest

from fase_2.tech_challenge.src.ga.encoding import n_genes
from fase_2.tech_challenge.src.ga.fitness import build_model
from fase_2.tech_challenge.src.llm.evaluate import QualityCheck, build_report_table, flag_jargon
from fase_2.tech_challenge.src.llm.explain import explain_comparison, explain_diagnosis, top_features_for_sample
from fase_2.tech_challenge.src.llm.prompts import build_comparison_prompt, build_diagnosis_prompt
from fase_2.tech_challenge.src.models.data import load_dataset, train_test_split_default


# ---------------------------------------------------------------------------
# Prompts — grounding (os números vêm sempre do chamador, nunca inventados)
# ---------------------------------------------------------------------------

def test_build_diagnosis_prompt_includes_all_grounded_data():
    top_features = [
        {"name": "radius_mean", "value": 17.99, "influence": "aumenta o risco de malignidade"},
        {"name": "texture_mean", "value": 10.38, "influence": "reduz o risco de malignidade"},
    ]
    messages = build_diagnosis_prompt("Logistic Regression", "Maligno", 0.87, top_features)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert "Logistic Regression" in user_text
    assert "Maligno" in user_text
    assert "87.0%" in user_text
    assert "radius_mean" in user_text
    assert "aumenta o risco de malignidade" in user_text


def test_system_prompt_forbids_ml_jargon_and_absolute_claims():
    messages = build_diagnosis_prompt("KNN", "Benigno", 0.6, [])
    system_text = messages[0]["content"]
    assert "class_weight" in system_text  # citado como termo A EVITAR
    assert "certeza absoluta" in system_text


def test_build_comparison_prompt_includes_both_metric_sets():
    baseline = {"accuracy": 0.9737, "recall": 0.9767, "precision": 0.9545, "f1": 0.9655}
    optimized = {"accuracy": 0.95, "recall": 0.99, "precision": 0.90, "f1": 0.94}
    config = {"population_size": 20, "generations": 15}

    messages = build_comparison_prompt("logistic_regression", baseline, optimized, config)
    user_text = messages[1]["content"]

    assert "97.4%" in user_text  # accuracy do baseline formatada
    assert "99.0%" in user_text  # recall do otimizado formatada
    assert "população=20" in user_text
    assert "gerações=15" in user_text


# ---------------------------------------------------------------------------
# explain.py — cálculo de features influentes (sem chamar a LLM)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_logreg():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split_default(df)
    individual = np.array([0.5] * n_genes("logistic_regression"))
    model = build_model("logistic_regression", individual)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


def test_top_features_for_sample_returns_requested_count(trained_logreg):
    model, X_train, X_test, _ = trained_logreg
    feature_names = list(X_train.columns)
    sample = X_test.iloc[0].to_numpy()

    top = top_features_for_sample(model, feature_names, sample, top_n=5)

    assert len(top) == 5
    for f in top:
        assert f["name"] in feature_names
        assert f["influence"] in (
            "aumenta o risco de malignidade",
            "reduz o risco de malignidade",
        )


def test_top_features_sorted_by_absolute_influence(trained_logreg):
    model, X_train, X_test, _ = trained_logreg
    feature_names = list(X_train.columns)
    sample = X_test.iloc[0].to_numpy()

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    contributions = clf.coef_[0] * scaler.transform(sample.reshape(1, -1))[0]
    expected_top_idx = int(np.argmax(np.abs(contributions)))

    top = top_features_for_sample(model, feature_names, sample, top_n=1)
    assert top[0]["name"] == feature_names[expected_top_idx]


# ---------------------------------------------------------------------------
# explain_diagnosis / explain_comparison — chat_completion sempre mockado
# ---------------------------------------------------------------------------

def test_explain_diagnosis_calls_mocked_chat_completion(monkeypatch, trained_logreg):
    model, X_train, X_test, _ = trained_logreg
    feature_names = list(X_train.columns)
    sample = X_test.iloc[0].to_numpy()

    captured = {}

    def fake_chat_completion(messages, **kwargs):
        captured["messages"] = messages
        return "Explicação simulada."

    monkeypatch.setattr("fase_2.tech_challenge.src.llm.explain.chat_completion", fake_chat_completion)

    result = explain_diagnosis(model, "Logistic Regression", feature_names, sample)

    assert result == "Explicação simulada."
    assert captured["messages"][0]["role"] == "system"
    assert "Logistic Regression" in captured["messages"][1]["content"]


def test_explain_comparison_calls_mocked_chat_completion(monkeypatch):
    captured = {}

    def fake_chat_completion(messages, **kwargs):
        captured["messages"] = messages
        return "Comparativo simulado."

    monkeypatch.setattr("fase_2.tech_challenge.src.llm.explain.chat_completion", fake_chat_completion)

    baseline = {"accuracy": 0.97, "recall": 0.98, "precision": 0.95, "f1": 0.96}
    optimized = {"accuracy": 0.95, "recall": 0.99, "precision": 0.90, "f1": 0.94}
    result = explain_comparison(
        "logistic_regression", baseline, optimized, {"population_size": 20, "generations": 15}
    )

    assert result == "Comparativo simulado."
    assert "logistic_regression" in captured["messages"][1]["content"]


# ---------------------------------------------------------------------------
# evaluate.py — checklist de qualidade
# ---------------------------------------------------------------------------

def test_flag_jargon_detects_known_terms():
    text_with_jargon = "O recall do modelo aumentou após ajustar o class_weight."
    text_clean = "O modelo passou a identificar mais corretamente os casos malignos."

    assert "class_weight" in flag_jargon(text_with_jargon)
    assert flag_jargon(text_clean) == []


def test_quality_check_auto_prefill_flags_jargon():
    check = QualityCheck(sample_id="s1", explanation="Cuidado com overfitting no fitness.")
    check.auto_prefill()
    assert check.avoids_ml_jargon is False
    assert "overfitting" in check.notes or "fitness" in check.notes


def test_quality_check_auto_prefill_passes_clean_text():
    check = QualityCheck(sample_id="s2", explanation="O modelo indica risco elevado de malignidade.")
    check.auto_prefill()
    assert check.avoids_ml_jargon is True


def test_build_report_table_formats_rows():
    checks = [QualityCheck(sample_id="s1", explanation="ok", cites_correct_numbers=True)]
    table = build_report_table(checks)
    assert table[0]["amostra"] == "s1"
    assert table[0]["cita_numeros_corretos"] is True
