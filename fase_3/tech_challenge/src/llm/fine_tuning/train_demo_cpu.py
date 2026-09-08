"""
Fine-tuning demo that runs without a GPU and without internet access, to
validate the pipeline (data -> "training" -> model -> inference) end-to-end
on any machine - including a sandboxed environment with no pretrained model
weight downloads.

Important: this does NOT replace the real fine-tuning required by the
challenge (notebooks/fine_tuning_colab.ipynb, Unsloth + Llama-3-8b + LoRA,
which needs a GPU). Here we use a much simpler technique - a retrieval index
(TF-IDF + cosine similarity) over the fine-tuning dataset itself - that acts
as a low-cost "model": given a new question, it returns the answer from the
most similar training example.

This keeps the same interface (DomainLLM.generate_response) used by the
rest of the system, allowing this backend to be swapped for the real
fine-tuned model without changing any other module (RAG, LangChain,
LangGraph).
"""
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_fine_tuning.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "demo_retrieval"


def load_training_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def train(dataset_path: Path = DATASET_PATH, output_dir: Path = OUTPUT_DIR):
    examples = load_training_dataset(dataset_path)
    if not examples:
        raise RuntimeError(
            f"Dataset vazio em {dataset_path}. Rode antes os scripts de src/data_prep/."
        )

    # Text used to "match" a new question against the closest training
    # example: instruction + input, same as what the real model would receive.
    training_texts = [f"{ex['instruction']} {ex['input']}" for ex in examples]
    answers = [ex["output"] for ex in examples]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(training_texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tfidf_index.pkl", "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "answers": answers,
             "training_questions": training_texts},
            f,
        )

    print(f"Indice de recuperacao (demo, {len(examples)} exemplos) salvo em {output_dir}")
    print("Este e um stand-in leve para o fine-tuning real - ver notebooks/fine_tuning_colab.ipynb")


if __name__ == "__main__":
    train()
