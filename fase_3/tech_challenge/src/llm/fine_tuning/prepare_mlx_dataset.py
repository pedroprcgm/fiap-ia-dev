"""
Converts data/processed/dataset_fine_tuning.jsonl (instruction/input/output,
Alpaca-style - see src/llm/data_prep/build_fine_tuning_dataset.py) into the
"completions" JSONL format expected by `mlx_lm.lora` (prompt/completion
pairs) - see notebooks/fine_tuning_local_mlx.ipynb.

The prompt is built the same way DomainLLM does at inference time when there
is no RAG context (`instruction + "\n\n" + input`, see
src/llm/models/domain_llm.py::generate_response and its "mlx" backend), so
the model sees a consistent format between training and inference.

Splits into train.jsonl / valid.jsonl inside an output directory (default
data/processed/mlx_finetune/). The dataset here is intentionally small
(demo-sized, ~20 examples) - see the notebook for why that limits what a
real LoRA fine-tune can learn from it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_fine_tuning.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "mlx_finetune"


def load_examples(dataset_path: Path) -> list[dict]:
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def to_completion_pair(example: dict) -> dict:
    prompt = f"{example['instruction']}\n\n{example['input']}" if example.get("input") else example["instruction"]
    return {"prompt": prompt, "completion": example["output"]}


def split_train_valid(pairs: list[dict], valid_fraction: float, seed: int, min_valid: int = 2) -> tuple[list[dict], list[dict]]:
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    valid_size = max(min_valid, round(len(shuffled) * valid_fraction))
    valid_size = min(valid_size, len(shuffled) - 1)  # always leave at least 1 for train

    return shuffled[valid_size:], shuffled[:valid_size]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    valid_fraction: float = 0.15,
    seed: int = 3407,
) -> None:
    examples = load_examples(dataset_path)
    if not examples:
        raise RuntimeError(
            f"Dataset vazio em {dataset_path}. Rode antes:\n"
            "  python -m src.llm.data_prep.generate_synthetic_hospital_data\n"
            "  python -m src.llm.data_prep.prepare_public_datasets\n"
            "  python -m src.llm.data_prep.build_fine_tuning_dataset"
        )

    pairs = [to_completion_pair(ex) for ex in examples]
    train_pairs, valid_pairs = split_train_valid(pairs, valid_fraction, seed)

    write_jsonl(output_dir / "train.jsonl", train_pairs)
    write_jsonl(output_dir / "valid.jsonl", valid_pairs)

    print(f"{len(train_pairs)} exemplos de treino e {len(valid_pairs)} de validacao salvos em {output_dir}")
    if len(examples) < 50:
        print(
            "Aviso: dataset pequeno (demo). O resultado do fine-tuning tende a ser "
            "sobreajustado a esses exemplos especificos, nao generalizacao ampla - "
            "isso e esperado neste projeto, nao um bug do script."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.dataset_path, args.output_dir, args.valid_fraction, args.seed)
