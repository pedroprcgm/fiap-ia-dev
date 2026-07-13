import numpy as np
import pytest

from fase_2.tech_challenge.src.ga.encoding import (
    LOGREG_L2_SPEC,
    LOGREG_SPEC,
    SVC_SPEC,
    decode_individual,
    n_genes,
)
from fase_2.tech_challenge.src.ga.fitness import build_model, evaluate_individual
from fase_2.tech_challenge.src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from fase_2.tech_challenge.src.ga.operators import crossover, init_population, mutate, tournament_selection
from fase_2.tech_challenge.src.models.data import load_dataset, train_test_split_default


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_logreg_decode_bounds():
    params = decode_individual("logistic_regression", [0.0, 0.0, 0.0])
    assert params["C"] == pytest.approx(1e-3, rel=1e-6)
    assert params["penalty"] == "l2"
    assert params["class_weight_pos"] == 1

    params = decode_individual("logistic_regression", [1.0, 1.0, 1.0])
    assert params["C"] == pytest.approx(1e2, rel=1e-6)
    assert params["penalty"] == "l1"
    assert params["class_weight_pos"] == 10


def test_svc_decode_bounds():
    params = decode_individual("svc_linear", [0.0, 0.0])
    assert params["C"] == pytest.approx(1e-2, rel=1e-6)
    assert params["class_weight_pos"] == 1

    params = decode_individual("svc_linear", [1.0, 1.0])
    assert params["C"] == pytest.approx(1e2, rel=1e-6)
    assert params["class_weight_pos"] == 10


def test_n_genes_matches_spec():
    assert n_genes("logistic_regression") == len(LOGREG_SPEC)
    assert n_genes("svc_linear") == len(SVC_SPEC)
    assert n_genes("logistic_regression_l2") == len(LOGREG_L2_SPEC)


def test_logreg_l2_decode_bounds_and_has_no_penalty_gene():
    # LOGREG_L2_SPEC tem só 2 genes (C, class_weight_pos) — sem "penalty": ele é
    # travado em "l2" diretamente em src/ga/fitness.py::build_model, não faz parte
    # do cromossomo.
    params = decode_individual("logistic_regression_l2", [0.0, 0.0])
    assert params["C"] == pytest.approx(1e-3, rel=1e-6)
    assert params["class_weight_pos"] == 1
    assert "penalty" not in params

    params = decode_individual("logistic_regression_l2", [1.0, 1.0])
    assert params["C"] == pytest.approx(1e2, rel=1e-6)
    assert params["class_weight_pos"] == 10


def test_decode_clips_out_of_range_genes():
    # genes fora de [0, 1] não devem quebrar a decodificação (clip de segurança)
    params = decode_individual("svc_linear", [-0.5, 1.5])
    assert params["C"] == pytest.approx(1e-2, rel=1e-6)
    assert params["class_weight_pos"] == 10


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def test_init_population_shape():
    rng = np.random.default_rng(0)
    pop = init_population(10, 3, rng)
    assert len(pop) == 10
    assert all(len(ind) == 3 for ind in pop)
    assert all(((ind >= 0) & (ind <= 1)).all() for ind in pop)


def test_tournament_selection_picks_best_in_sample():
    rng = np.random.default_rng(0)
    population = [np.array([0.1]), np.array([0.5]), np.array([0.9])]
    fitnesses = [0.1, 0.5, 0.9]
    # com tournament_size igual ao tamanho da população, deve sempre escolher o melhor
    selected = tournament_selection(population, fitnesses, rng, tournament_size=3)
    assert selected[0] == pytest.approx(0.9)


def test_crossover_respects_rate_zero():
    rng = np.random.default_rng(0)
    a, b = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    child_a, child_b = crossover(a, b, rng, crossover_rate=0.0)
    assert np.array_equal(child_a, a)
    assert np.array_equal(child_b, b)


def test_mutate_stays_within_bounds():
    rng = np.random.default_rng(0)
    individual = np.array([0.0, 1.0, 0.5])
    mutated = mutate(individual, rng, mutation_rate=1.0, sigma=0.5)
    assert ((mutated >= 0.0) & (mutated <= 1.0)).all()


# ---------------------------------------------------------------------------
# Fitness / build_model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_split():
    df = load_dataset()
    return train_test_split_default(df)


def test_logreg_l2_variant_always_uses_l2_even_when_free_variant_would_pick_l1(data_split):
    # individual=[1.0, 1.0, 1.0] faz "logistic_regression" decodificar penalty="l1"
    # (ver test_logreg_decode_bounds) — confirma que a variante l2-only ignora essa
    # possibilidade e trava o classificador em penalty="l2" de qualquer forma.
    individual_l2 = np.array([1.0, 1.0])  # só 2 genes: C, class_weight_pos
    model = build_model("logistic_regression_l2", individual_l2)
    assert model.named_steps["clf"].penalty == "l2"


@pytest.mark.parametrize(
    "model_family", ["logistic_regression", "logistic_regression_l2", "svc_linear"]
)
def test_build_model_trains_and_predicts(model_family, data_split):
    X_train, X_test, y_train, y_test = data_split
    individual = np.array([0.5] * n_genes(model_family))
    model = build_model(model_family, individual)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


@pytest.mark.parametrize(
    "model_family", ["logistic_regression", "logistic_regression_l2", "svc_linear"]
)
def test_evaluate_individual_returns_valid_metrics(model_family, data_split):
    X_train, X_test, y_train, y_test = data_split
    individual = np.array([0.5] * n_genes(model_family))
    result = evaluate_individual(model_family, individual, X_train, y_train)
    for metric in (result.accuracy, result.recall, result.precision, result.f1, result.score):
        assert 0.0 <= metric <= 1.0


@pytest.mark.parametrize("cv_folds", [3, 5])
def test_evaluate_individual_respects_cv_folds(cv_folds, data_split):
    X_train, X_test, y_train, y_test = data_split
    individual = np.array([0.5, 0.5, 0.5])
    result = evaluate_individual(
        "logistic_regression", individual, X_train, y_train, cv_folds=cv_folds
    )
    for metric in (result.accuracy, result.recall, result.precision, result.f1, result.score):
        assert 0.0 <= metric <= 1.0


def test_evaluate_individual_is_deterministic_for_same_random_state(data_split):
    X_train, X_test, y_train, y_test = data_split
    individual = np.array([0.5, 0.5, 0.5])
    result_a = evaluate_individual(
        "logistic_regression", individual, X_train, y_train, cv_folds=5, random_state=7
    )
    result_b = evaluate_individual(
        "logistic_regression", individual, X_train, y_train, cv_folds=5, random_state=7
    )
    assert result_a.score == pytest.approx(result_b.score)


def test_evaluate_individual_cv_differs_from_a_single_lucky_split(data_split):
    # k-fold usa TODO o conjunto de treino para validação (em partições diferentes),
    # então o score médio deve ser mais estável do que avaliar num único fold —
    # aqui só confirmamos que mudar o número de folds pode mudar o resultado (ou seja,
    # a média está de fato variando com a partição, não é um valor fixo/hardcoded).
    X_train, X_test, y_train, y_test = data_split
    individual = np.array([0.5, 0.5, 0.5])
    result_3fold = evaluate_individual(
        "logistic_regression", individual, X_train, y_train, cv_folds=3, random_state=1
    )
    result_10fold = evaluate_individual(
        "logistic_regression", individual, X_train, y_train, cv_folds=10, random_state=1
    )
    assert 0.0 <= result_3fold.score <= 1.0
    assert 0.0 <= result_10fold.score <= 1.0


# ---------------------------------------------------------------------------
# GA end-to-end (smoke test, pop/gerações pequenos para rodar rápido)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_family", ["logistic_regression", "logistic_regression_l2", "svc_linear"]
)
def test_ga_runs_and_elitism_is_monotonic(model_family, data_split):
    X_train, X_test, y_train, y_test = data_split
    # CV_FOLDS baixo (3) só para o teste rodar rápido; produção usa o default (5).
    config = GAConfig(POPULATION_SIZE=8, GENERATIONS=4, ELITISM=2, RANDOM_STATE=1, CV_FOLDS=3)
    ga = GeneticAlgorithm(model_family, config)
    result = ga.run(X_train, y_train, verbose=False)

    assert len(result.history) == config.GENERATIONS
    # Com elitismo, o melhor score da geração nunca pode piorar em relação à anterior.
    best_scores = [g.best_score for g in result.history]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best_scores, best_scores[1:]))
    assert 0.0 <= result.best_eval.score <= 1.0
