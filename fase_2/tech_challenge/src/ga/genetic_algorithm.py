"""Loop principal do Algoritmo Genético: evolui uma população de hiperparâmetros de um
modelo (Logistic Regression ou SVC Linear) para maximizar a função fitness.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from fase_2.tech_challenge.src.ga.encoding import n_genes, random_individual
from fase_2.tech_challenge.src.ga.fitness import EvalResult, evaluate_individual
from fase_2.tech_challenge.src.ga.operators import crossover, init_population, mutate, tournament_selection

logger = logging.getLogger(__name__)

PROGRESS_PLOT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class _LivePlot:
    """Gráfico de linha (score x geração) atualizado ao vivo a cada geração do GA."""

    def __init__(self, model_family: str, total_generations: int, pause_seconds: float = 5.0):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.pause_seconds = pause_seconds
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.generations: List[int] = []
        self.best_scores: List[float] = []
        self.avg_scores: List[float] = []

        (self.best_line,) = self.ax.plot(
            [], [], marker="o", color="#1f77b4", label="Melhor score da geração"
        )
        (self.avg_line,) = self.ax.plot(
            [], [], marker=".", linestyle="--", color="#ff7f0e", label="Score médio da população"
        )

        self.ax.set_xlabel("Geração")
        self.ax.set_ylabel("Score (fitness)")
        self.ax.set_title(f"Evolução do Algoritmo Genético — {model_family}")
        self.ax.set_xlim(1, max(total_generations, 2))
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc="lower right")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self._plt.pause(0.001)

    def update(self, generation: int, best_score: float, avg_score: float) -> None:
        self.generations.append(generation)
        self.best_scores.append(best_score)
        self.avg_scores.append(avg_score)

        self.best_line.set_data(self.generations, self.best_scores)
        self.avg_line.set_data(self.generations, self.avg_scores)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Pausa proposital (não só um "yield" de 0.001s): dá tempo de acompanhar
        # visualmente a geração recém-desenhada antes de seguir para a próxima.
        self._plt.pause(self.pause_seconds)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=120)

    def close(self) -> None:
        self._plt.ioff()


@dataclass
class GenerationLog:
    generation: int
    best_score: float
    avg_score: float
    best_params: dict


@dataclass
class GAResult:
    model_family: str
    best_individual: np.ndarray
    best_eval: EvalResult
    history: List[GenerationLog] = field(default_factory=list)


@dataclass
class GAConfig:
    POPULATION_SIZE: int = 20
    GENERATIONS: int = 30
    CROSSOVER_RATE: float = 0.8
    MUTATION_RATE: float = 0.1
    MUTATION_SIGMA: float = 0.15
    TOURNAMENT_SIZE: int = 3
    ELITISM: int = 2
    RANDOM_STATE: int = 42
    # Número de folds usados na cross-validation estratificada do fitness
    # (ver src/ga/fitness.py::evaluate_individual).
    CV_FOLDS: int = 5


class GeneticAlgorithm:
    def __init__(self, model_family: str, config: GAConfig = GAConfig()):
        self.model_family = model_family
        self.config = config
        self.rng = np.random.default_rng(config.RANDOM_STATE)

    def run(
        self,
        X,
        y,
        verbose: bool = True,
        live_plot: bool = False,
        save_plot: bool = True,
        live_plot_delay: float = 1,
    ) -> GAResult:
        """Executa o loop do GA.

        `X`/`y`: conjunto de treino completo disponível para o GA (o X_test final,
        held-out, nunca entra aqui). O fitness de cada indivíduo é calculado por
        k-fold cross-validation sobre (X, y) — ver `src/ga/fitness.py::evaluate_individual`
        e `GAConfig.CV_FOLDS`.

        `live_plot=True` para acompanhar em tempo real a evolução do score da população a cada geração.
        `save_plot=True` para salvar o gráfico final (score x geração)
        """
        cfg = self.config
        genes = n_genes(self.model_family)
        population = init_population(cfg.POPULATION_SIZE, genes, self.rng)
        history: List[GenerationLog] = []

        best_individual = None
        best_eval = None

        plotter: Optional[_LivePlot] = (
            _LivePlot(self.model_family, cfg.GENERATIONS, pause_seconds=live_plot_delay)
            if live_plot
            else None
        )

        for generation in range(1, cfg.GENERATIONS + 1):
            evaluations = [
                evaluate_individual(
                    self.model_family, ind, X, y,
                    cv_folds=cfg.CV_FOLDS, random_state=cfg.RANDOM_STATE,
                )
                for ind in population
            ]
            scores = [e.score for e in evaluations]

            gen_best_idx = int(np.argmax(scores))
            gen_best_eval = evaluations[gen_best_idx]

            if best_eval is None or gen_best_eval.score > best_eval.score:
                best_eval = gen_best_eval
                best_individual = population[gen_best_idx].copy()

            history.append(
                GenerationLog(
                    generation=generation,
                    best_score=gen_best_eval.score,
                    avg_score=float(np.mean(scores)),
                    best_params=gen_best_eval.params,
                )
            )
            if verbose:
                logger.info(
                    "[%s] geracao %02d/%02d | best=%.4f avg=%.4f | params=%s",
                    self.model_family, generation, cfg.GENERATIONS,
                    gen_best_eval.score, float(np.mean(scores)), gen_best_eval.params,
                )

            if plotter is not None:
                plotter.update(generation, gen_best_eval.score, float(np.mean(scores)))

            # Elitismo: os melhores indivíduos passam direto para a próxima geração.
            elite_idx = np.argsort(scores)[::-1][: cfg.ELITISM]
            next_population = [population[i].copy() for i in elite_idx]

            # Preenche o restante da população via seleção + cruzamento + mutação.
            while len(next_population) < cfg.POPULATION_SIZE:
                parent_a = tournament_selection(population, scores, self.rng, cfg.TOURNAMENT_SIZE)
                parent_b = tournament_selection(population, scores, self.rng, cfg.TOURNAMENT_SIZE)
                child_a, child_b = crossover(parent_a, parent_b, self.rng, cfg.CROSSOVER_RATE)
                child_a = mutate(child_a, self.rng, cfg.MUTATION_RATE, cfg.MUTATION_SIGMA)
                child_b = mutate(child_b, self.rng, cfg.MUTATION_RATE, cfg.MUTATION_SIGMA)
                next_population.extend([child_a, child_b])

            population = next_population[: cfg.POPULATION_SIZE]

        if plotter is not None:
            if save_plot:
                plotter.save(PROGRESS_PLOT_DIR / f"ga_progress_{self.model_family}.png")
            plotter.close()

        assert best_individual is not None and best_eval is not None
        return GAResult(
            model_family=self.model_family,
            best_individual=best_individual,
            best_eval=best_eval,
            history=history,
        )
