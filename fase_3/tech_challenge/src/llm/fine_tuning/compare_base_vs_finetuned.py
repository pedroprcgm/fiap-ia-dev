"""
Compares the base (un-fine-tuned) model against the mlx-lm LoRA fine-tuned
model (see notebooks/fine_tuning_local_mlx.ipynb) on the same questions -
this is the demonstration for "what did the fine-tuning itself contribute,
as opposed to RAG retrieval" (see the README section on evaluating the
fine-tuning's effect).

Two kinds of test cases, on purpose:
  - "exclusivas do fine-tuning": questions about conditions that exist ONLY
    in the fine-tuning dataset (see
    src/llm/data_prep/build_fine_tuning_dataset.py::FINE_TUNING_ONLY_CONDITIONS),
    never indexed by RAG (HospitalKnowledgeBase only reads data/raw/, which
    these are deliberately absent from). No retrieved context is passed to
    either model here - only a model that actually learned this content
    during fine-tuning can answer it correctly; the base model has nothing
    to lean on and has to improvise.
  - "com contexto do RAG": regular hospital conditions, with real RAG
    context built the same way MedicalAssistantChain does. Both models get
    IDENTICAL context, so any difference here is about tone/format/style,
    not information access - a decent instruct model usually already
    answers these reasonably well from context alone, which is exactly why
    the first set of questions above is the more convincing evidence of
    what fine-tuning added.

Only runs on Apple Silicon (mlx-lm) with a fine-tuned adapter already
produced by notebooks/fine_tuning_local_mlx.ipynb - same hardware
requirement as that notebook.

Run with:
    python -m src.llm.fine_tuning.compare_base_vs_finetuned
    python -m src.llm.fine_tuning.compare_base_vs_finetuned --adapter-path models/fine_tuned_lora_mlx
"""
import argparse
import json
from pathlib import Path

from src.llm.data_prep.build_fine_tuning_dataset import FINE_TUNING_ONLY_CONDITIONS
from src.llm.models.domain_llm import ensure_mlx_eos_tokens, generate_with_mlx_model
from src.llm.rag.vector_store import HospitalKnowledgeBase

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_ADAPTER_PATH = BASE_DIR / "models" / "fine_tuned_lora_mlx"

INSTRUCTION = "Responda como assistente clinico do hospital, com base no protocolo interno."

# A couple of regular (RAG-covered) conditions, just to also show
# tone/format differences when both models get the exact same retrieved
# context - see the module docstring for why this is the weaker half of
# the demonstration.
RAG_COVERED_QUESTIONS = [
    "Qual o protocolo interno para Hipertensao Arterial Sistemica?",
    "Qual a conduta preconizada pelo hospital para Infeccao do Trato Urinario nao complicada?",
]


def _load_models(adapter_path: Path):
    # Local import: mlx-lm only installs/runs on Apple Silicon (Metal) - see
    # the module docstring and notebooks/fine_tuning_local_mlx.ipynb.
    from mlx_lm import load as mlx_load

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"'{adapter_config_path}' nao encontrado. Rode "
            "notebooks/fine_tuning_local_mlx.ipynb primeiro para gerar o adaptador, "
            "ou aponte --adapter-path para onde ele foi salvo."
        )

    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["model"]

    print(f"Carregando modelo base: {base_model_name} ...")
    base_model, base_tokenizer = mlx_load(base_model_name)
    ensure_mlx_eos_tokens(base_tokenizer)

    print(f"Carregando modelo fine-tuned: {base_model_name} + adapter em {adapter_path} ...")
    finetuned_model, finetuned_tokenizer = mlx_load(base_model_name, adapter_path=str(adapter_path))
    ensure_mlx_eos_tokens(finetuned_tokenizer)

    return (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer)


def _print_comparison(title: str, question: str, base_answer: str, finetuned_answer: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Pergunta: {question}\n")
    print("--- Modelo BASE (sem fine-tuning) ---")
    print(base_answer)
    print("\n--- Modelo FINE-TUNED ---")
    print(finetuned_answer)
    print("-" * 70)


def run(adapter_path: Path = DEFAULT_ADAPTER_PATH) -> None:
    (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer) = _load_models(adapter_path)

    print(
        "\n\n######## PARTE 1: perguntas EXCLUSIVAS do fine-tuning (fora do RAG) ########"
    )
    print(
        "O RAG nao tem nenhum documento sobre essas condicoes - sem contexto, so um "
        "modelo que realmente aprendeu isso no fine-tuning consegue responder certo.\n"
    )
    for case in FINE_TUNING_ONLY_CONDITIONS:
        question = f"Qual o protocolo interno para {case['condition']}?"
        base_answer = generate_with_mlx_model(base_model, base_tokenizer, INSTRUCTION, question)
        finetuned_answer = generate_with_mlx_model(finetuned_model, finetuned_tokenizer, INSTRUCTION, question)
        _print_comparison(f"Condicao exclusiva: {case['condition']}", question, base_answer, finetuned_answer)

    print(
        "\n\n######## PARTE 2: perguntas COM contexto do RAG (mesmo contexto pros dois) ########"
    )
    print(
        "Aqui os dois modelos recebem exatamente o mesmo contexto recuperado - a "
        "diferenca (se houver) e de tom/formato/aderencia, nao de informacao.\n"
    )
    knowledge_base = HospitalKnowledgeBase()
    for question in RAG_COVERED_QUESTIONS:
        rag_documents = knowledge_base.search(question, k=3)
        rag_context = "\n\n".join(f"[Fonte: {d['source']}]\n{d['content']}" for d in rag_documents)
        base_answer = generate_with_mlx_model(
            base_model, base_tokenizer, INSTRUCTION, question, context=rag_context
        )
        finetuned_answer = generate_with_mlx_model(
            finetuned_model, finetuned_tokenizer, INSTRUCTION, question, context=rag_context
        )
        _print_comparison("Com contexto do RAG", question, base_answer, finetuned_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    args = parser.parse_args()
    run(args.adapter_path)
