"""Operadores genéticos: inicialização de população, seleção, cruzamento e mutação.

Todos os operadores trabalham sobre cromossomos reais normalizados em [0, 1]
(ver `src/ga/encoding.py`), o que os torna independentes da família de modelo.
"""
from typing import List, Tuple

import numpy as np

Population = List[np.ndarray]


def init_population(pop_size: int, n_genes: int, rng: np.random.Generator) -> Population:
    return [rng.random(n_genes) for _ in range(pop_size)]


def tournament_selection(
    population: Population,
    fitnesses: List[float],
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> np.ndarray:
    """Seleção por torneio: sorteia `tournament_size` indivíduos e retorna o melhor."""
    idx = rng.integers(0, len(population), size=tournament_size)
    best_idx = max(idx, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    crossover_rate: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cruzamento uniforme: para cada gene, com probabilidade 0.5 os filhos trocam o
    valor entre os pais. Só ocorre com probabilidade `crossover_rate`; caso contrário
    os filhos são cópias dos pais."""
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()

    mask = rng.random(len(parent_a)) < 0.5
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a, child_b


def mutate(
    individual: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
    sigma: float = 0.15,
) -> np.ndarray:
    """Mutação gaussiana: cada gene tem probabilidade `mutation_rate` de sofrer uma
    perturbação normal(0, sigma), com o resultado limitado (clip) a [0, 1]."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] = np.clip(mutated[i] + rng.normal(0, sigma), 0.0, 1.0)
    return mutated
