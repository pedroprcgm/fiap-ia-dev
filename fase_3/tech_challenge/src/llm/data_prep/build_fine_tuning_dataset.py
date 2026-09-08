"""
Consolidates the raw data (synthetic protocols/FAQs/report templates +
MedQuAD/PubMedQA samples) into a single fine-tuning dataset, in the
instruction/input/output format used by the Alpaca prompt - the same
pattern used in the course's reference material (Aula 02 - Fine-tuning de
LLM para documentos).

Steps: load raw data -> anonymize -> curate -> format -> save
data/processed/dataset_fine_tuning.jsonl
"""
import json
from pathlib import Path

from src.llm.data_prep.anonymize import anonymize_text, curate_example

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROTOCOL_INSTRUCTION = "Responda como assistente clinico do hospital, com base no protocolo interno."
FAQ_INSTRUCTION = "Responda a duvida do medico com base nos protocolos internos do hospital."
GENERAL_HEALTH_INSTRUCTION = "Responda a pergunta clinica com base em evidencias medicas."


def _load_json(file_name: str) -> list[dict]:
    path = RAW_DIR / file_name
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _examples_from_protocols(protocols: list[dict]) -> list[dict]:
    examples = []
    for item in protocols:
        question = f"Qual o protocolo interno para {item['condition']}?"
        answer = item["content"]
        if curate_example(question, answer):
            examples.append({
                "instruction": PROTOCOL_INSTRUCTION,
                "input": anonymize_text(question),
                "output": anonymize_text(answer),
            })
    return examples


def _examples_from_faqs(faqs: list[dict]) -> list[dict]:
    examples = []
    for item in faqs:
        if curate_example(item["question"], item["answer"]):
            examples.append({
                "instruction": FAQ_INSTRUCTION,
                "input": anonymize_text(item["question"]),
                "output": anonymize_text(item["answer"]),
            })
    return examples


def _examples_from_public_qa(pairs: list[dict]) -> list[dict]:
    examples = []
    for item in pairs:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if curate_example(question, answer):
            examples.append({
                "instruction": GENERAL_HEALTH_INSTRUCTION,
                "input": anonymize_text(question),
                "output": anonymize_text(answer),
            })
    return examples


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    protocols = _load_json("protocolos_internos.json")
    faqs = _load_json("faqs_medicos.json")
    medquad = _load_json("medquad_sample.json")
    pubmedqa = _load_json("pubmedqa_sample.json")

    dataset = (
        _examples_from_protocols(protocols)
        + _examples_from_faqs(faqs)
        + _examples_from_public_qa(medquad)
        + _examples_from_public_qa(pubmedqa)
    )

    if not dataset:
        raise RuntimeError(
            "Nenhum exemplo gerado. Rode antes:\n"
            "  python -m src.data_prep.generate_synthetic_hospital_data\n"
            "  python -m src.data_prep.prepare_public_datasets"
        )

    output_path = PROCESSED_DIR / "dataset_fine_tuning.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Dataset de fine-tuning com {len(dataset)} exemplos salvo em {output_path}")
    print(f"  - {len(protocols)} protocolos internos")
    print(f"  - {len(faqs)} FAQs de medicos")
    print(f"  - {len(medquad)} pares MedQuAD")
    print(f"  - {len(pubmedqa)} pares PubMedQA")


if __name__ == "__main__":
    main()
