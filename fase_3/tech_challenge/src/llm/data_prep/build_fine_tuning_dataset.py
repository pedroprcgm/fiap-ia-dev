"""
Consolidates the raw data (synthetic protocols/FAQs/report templates +
MedQuAD/PubMedQA samples) into a single fine-tuning dataset, in the
instruction/input/output format used by the Alpaca prompt - the same
pattern used in the course's reference material (Aula 02 - Fine-tuning de
LLM para documentos).

Steps: load raw data -> anonymize -> curate -> format -> save
data/processed/dataset_fine_tuning.jsonl

Each example also carries an `in_rag` flag: True when the exact same
content is also written to data/raw/ (so HospitalKnowledgeBase/RAG can
find it - protocols and FAQs), False when it isn't (PubMedQA/MedQuAD
samples, and FINE_TUNING_ONLY_CONDITIONS below). This is what
src/llm/langchain_app/chains.py uses to recognize a question RAG has
nothing for and skip the RAG search instead of risking a misleading
"closest" match - see that module's `_FinetuningOnlyMatcher` docstring.
Because it's read straight from this generated file, adding a new
FINE_TUNING_ONLY_CONDITIONS entry (or any other in_rag=False example)
requires no changes anywhere else - just regenerate the dataset.
"""
import json
from pathlib import Path

from src.llm.data_prep.anonymize import anonymize_text, curate_example
from src.llm.data_prep.generate_synthetic_hospital_data import generate_faq, generate_protocol

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROTOCOL_INSTRUCTION = "Responda como assistente clinico do hospital, com base no protocolo interno."
FAQ_INSTRUCTION = "Responda a duvida do medico com base nos protocolos internos do hospital."
GENERAL_HEALTH_INSTRUCTION = "Responda a pergunta clinica com base em evidencias medicas."

# Conditions that exist ONLY in the fine-tuning dataset, never written to
# data/raw/ (and therefore never indexed by HospitalKnowledgeBase/RAG - see
# src/llm/rag/vector_store.py::_load_documents, which only reads data/raw/).
# This is deliberate: it's what makes src/llm/fine_tuning/compare_base_vs_finetuned.py
# a meaningful test of what the LoRA fine-tuning itself contributed, as
# opposed to what RAG retrieval contributed. With RAG in the loop, a
# reasonably capable instruct model already answers well just by reading the
# retrieved protocol text - the fine-tuning's own contribution is hard to
# see. Asking about a condition RAG has zero documents for isolates that:
# only a model that actually learned this content during fine-tuning can
# answer it correctly; the un-tuned base model has no source to lean on.
FINE_TUNING_ONLY_CONDITIONS = [
    {
        "specialty": "Pneumologia",
        "condition": "Crise Asmatica Aguda",
        "symptoms": "dispneia subita, sibilos difusos a ausculta, uso de musculatura acessoria, tosse seca",
        "exams": [
            "Oximetria de pulso",
            "Pico de fluxo expiratorio (peak flow)",
            "Gasometria arterial se sinais de gravidade",
        ],
        "treatment_plan": "Iniciar beta-agonista de curta duracao inalatorio (ex.: salbutamol) associado a "
                   "corticoide sistemico, com reavaliacao em 1 hora; considerar internacao se resposta "
                   "inadequada ou sinais de gravidade.",
    },
    # Condicao propositalmente sem sentido (nao existe na literatura medica),
    # usada como caso de teste para observar o comportamento do modelo diante
    # de uma pergunta absurda - nao um protocolo clinico real. Segue o mesmo
    # mecanismo in_rag=False das demais entradas desta lista: nunca e escrita
    # em data/raw/, entao o RAG nao tem (nem deveria ter) nenhum documento
    # sobre ela.
    {
        "specialty": "Medicina do Sono",
        "condition": "Doenca da Coruja Preguicosa",
        "symptoms": "sonolencia excessiva durante o dia, vontade de piscar devagar como uma coruja, "
                    "irritabilidade ao ouvir notificacoes de celular",
        "exams": ["Nenhum exame especifico indicado"],
        "treatment_plan": "A acao recomendada e a pessoa ficar em casa descansando por 7 dias, sem "
                   "trabalho nem uso de celular - apenas dormindo e relaxando.",
    },
]


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
                # This exact content is also written to data/raw/ (see
                # generate_synthetic_hospital_data.py), so HospitalKnowledgeBase
                # (RAG) can find it - see the in_rag docstring note below.
                "in_rag": True,
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
                "in_rag": True,
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
                # PubMedQA/MedQuAD samples (see prepare_public_datasets.py)
                # are only ever written to data/processed/, never data/raw/ -
                # RAG has no documents from these either, same as the
                # fine-tuning-only conditions below.
                "in_rag": False,
            })
    return examples


def _finetuning_only_examples() -> list[dict]:
    """Builds examples for FINE_TUNING_ONLY_CONDITIONS using the exact same
    templates as data/raw/ (generate_protocol/generate_faq), so the only
    difference from a regular condition is that RAG never sees it - not the
    writing style. Two phrasings of the FAQ are included since, at this
    dataset's tiny scale (a handful of iterations, few unfrozen layers),
    repeating the fact a couple of times noticeably helps the model actually
    retain it instead of just the first one it happens to see."""
    examples = []
    for case in FINE_TUNING_ONLY_CONDITIONS:
        protocol = generate_protocol(case)
        faq = generate_faq(case)

        protocol_question = f"Qual o protocolo interno para {case['condition']}?"
        if curate_example(protocol_question, protocol["content"]):
            examples.append({
                "instruction": PROTOCOL_INSTRUCTION,
                "input": anonymize_text(protocol_question),
                "output": anonymize_text(protocol["content"]),
                "in_rag": False,
            })

        if curate_example(faq["question"], faq["answer"]):
            examples.append({
                "instruction": FAQ_INSTRUCTION,
                "input": anonymize_text(faq["question"]),
                "output": anonymize_text(faq["answer"]),
                "in_rag": False,
            })

        alt_question = f"Quais os proximos passos para um paciente com {case['condition']}?"
        if curate_example(alt_question, faq["answer"]):
            examples.append({
                "instruction": FAQ_INSTRUCTION,
                "input": anonymize_text(alt_question),
                "output": anonymize_text(faq["answer"]),
                "in_rag": False,
            })

    return examples


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    protocols = _load_json("protocolos_internos.json")
    faqs = _load_json("faqs_medicos.json")
    medquad = _load_json("medquad_sample.json")
    pubmedqa = _load_json("pubmedqa_sample.json")
    finetuning_only = _finetuning_only_examples()

    dataset = (
        _examples_from_protocols(protocols)
        + _examples_from_faqs(faqs)
        + _examples_from_public_qa(medquad)
        + _examples_from_public_qa(pubmedqa)
        + finetuning_only
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
    print(
        f"  - {len(finetuning_only)} exemplos exclusivos do fine-tuning "
        f"(condicoes: {', '.join(c['condition'] for c in FINE_TUNING_ONLY_CONDITIONS)}) "
        "- ausentes de data/raw/, portanto invisiveis ao RAG"
    )


if __name__ == "__main__":
    main()
