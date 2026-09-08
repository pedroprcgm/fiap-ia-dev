"""
Skeleton evaluation of the model after fine-tuning (technical report
requirement: "Model evaluation and results analysis").

Compares the model's answers (base or fine-tuned) against the dataset's
reference answers, using:
    - a simple lexical overlap rate (a lightweight proxy for ROUGE-L,
      without heavy external dependencies);
    - qualitative inspection (prints side by side for manual review).

For a "real" ROUGE metric, install `rouge-score` and replace
`_overlap_score` with `rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)`.
"""
import argparse
import json
from pathlib import Path

from src.llm.models.domain_llm import DomainLLM


def _overlap_score(reference: str, generated: str) -> float:
    """Simple overlap proxy (word-level Jaccard) between two answers."""
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())
    if not ref_tokens or not gen_tokens:
        return 0.0
    intersection = ref_tokens & gen_tokens
    union = ref_tokens | gen_tokens
    return len(intersection) / len(union)


def evaluate(dataset_path: str, n_samples: int = 5):
    llm = DomainLLM()

    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    samples = examples[:n_samples]
    scores = []

    print(f"Avaliando {len(samples)} exemplos com o modelo: {llm.describe()}\n")

    for i, example in enumerate(samples, start=1):
        generated_answer = llm.generate_response(example["instruction"], example["input"])
        score = _overlap_score(example["output"], generated_answer)
        scores.append(score)

        print(f"--- Exemplo {i} ---")
        print(f"Pergunta: {example['input']}")
        print(f"Resposta de referencia: {example['output'][:200]}...")
        print(f"Resposta gerada:        {generated_answer[:200]}...")
        print(f"Sobreposicao lexical (proxy): {score:.2f}\n")

    average = sum(scores) / len(scores) if scores else 0.0
    print(f"Sobreposicao lexical media (proxy de qualidade): {average:.2f}")
    return average


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/dataset_fine_tuning.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.dataset, args.samples)
