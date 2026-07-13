"""Carregamento e pré-processamento do dataset Breast Cancer Wisconsin (Diagnostic).

Replica fielmente o pré-processamento feito no notebook do Módulo 1
(`notebooks/modulo1_cancer_classification.ipynb`), para servir de base tanto para os
modelos originais quanto para os modelos otimizados via Algoritmo Genético.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "data.csv"

TARGET_COL = "diagnosis"
DROP_COLS = ["id", "Unnamed: 32"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV original sem nenhuma transformação."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o mesmo pré-processamento do Módulo 1:

    - mapeia diagnosis: "M" -> 1 (maligno), "B" -> 0 (benigno)
    - remove colunas irrelevantes (id, Unnamed: 32)
    - remove linhas nulas/duplicadas (não há nenhuma no dataset original)

    Nota: o notebook original NÃO aplica normalização/padronização das features,
    apesar delas terem escalas muito diferentes (ex.: area_mean ~centenas vs.
    smoothness_mean ~0.1). Isso é uma limitação conhecida do baseline — ver
    docs/decisions.md.
    """
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna().drop_duplicates()
    return df


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Atalho: carrega e pré-processa o dataset em uma única chamada."""
    return preprocess(load_raw(path))


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_test_split_default(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Reproduz o split 80/20 usado no Módulo 1 (sem estratificação, random_state=42)."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
