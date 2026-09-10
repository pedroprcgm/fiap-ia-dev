"""
DomainLLM: single interface for "the LLM fine-tuned on internal medical
data" (challenge requirement 1), used by the "Treatment Suggestion" node of
LangGraph.

Automatically resolves which backend to use, in this order of preference:
    1) A "real" LoRA adapter trained via transformers + peft (produced by
       notebooks/fine_tuning_colab.ipynb on Colab/NVIDIA GPU, Unsloth).
    2) A "real" LoRA adapter trained via mlx-lm (produced by
       notebooks/fine_tuning_local_mlx.ipynb, locally on a Mac with Apple
       Silicon - no CUDA/Colab needed, but only runs on that hardware).
    3) A lightweight retrieval index (produced by
       src/llm/fine_tuning/train_demo_cpu.py) - a stand-in with no
       GPU/internet, used for development and testing.
    4) If nothing has been trained yet, raises a clear error explaining what
       to run.

The rest of the system (RAG, LangChain, LangGraph) only knows the
`generate_response` method - swapping the backend doesn't require changing
anything else.
"""
# `from __future__ import annotations` allows using the `str | None` syntax
# (PEP 604) even on Python 3.9, which only supports it natively from 3.10
# onward - without this, the module fails to import on Python 3.9.
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[3]


def _detect_backend_kind(model_path: Path) -> str:
    """Figures out which backend `model_path` holds, from files alone - no
    torch/peft/mlx_lm import here, so this stays testable without any of
    those (heavy, platform-specific) packages installed.

    Returns "lora_hf", "lora_mlx" or "retrieval_demo"; raises FileNotFoundError
    if nothing recognizable is there, or ValueError if an adapter_config.json
    exists but matches neither known adapter format.
    """
    adapter_config_path = model_path / "adapter_config.json"
    tfidf_index_path = model_path / "tfidf_index.pkl"

    if adapter_config_path.exists():
        with open(adapter_config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Hugging Face / PEFT LoRA (notebooks/fine_tuning_colab.ipynb, via
        # Unsloth+peft): weights file is "adapter_model.safetensors" and the
        # config carries PEFT-specific keys like "peft_type"/
        # "base_model_name_or_path".
        if (model_path / "adapter_model.safetensors").exists() or "peft_type" in config:
            return "lora_hf"

        # mlx-lm LoRA (notebooks/fine_tuning_local_mlx.ipynb, via
        # `mlx_lm.lora`): weights file is "adapters.safetensors" (plural, no
        # "model") and the config is just the CLI args dumped as JSON,
        # including "model" (the base model repo/path) and
        # "lora_parameters"/"fine_tune_type".
        if (model_path / "adapters.safetensors").exists() or "lora_parameters" in config:
            return "lora_mlx"

        raise ValueError(
            f"'{adapter_config_path}' encontrado, mas nao reconhecido como adaptador "
            "Hugging Face/PEFT nem mlx-lm - verifique de onde ele veio."
        )

    if tfidf_index_path.exists():
        return "retrieval_demo"

    raise FileNotFoundError(
        f"Nenhum modelo de dominio encontrado em '{model_path}'.\n"
        "Rode um dos tres:\n"
        "  - python -m src.llm.fine_tuning.train_demo_cpu   (demo local, sem GPU)\n"
        "  - notebooks/fine_tuning_local_mlx.ipynb          (fine-tuning real, Mac Apple Silicon)\n"
        "  - notebooks/fine_tuning_colab.ipynb              (fine-tuning real, GPU NVIDIA/Colab)"
    )


# Chat-template control tokens that different open models leak into
# mlx_lm.generate()'s output when the tokenizer's EOS set doesn't already
# cover them (see the comment in DomainLLM._load_mlx) - unlike
# transformers' tokenizer.decode(..., skip_special_tokens=True), used by
# _generate_with_lora, mlx_lm does not filter these out on its own.
_MLX_CONTROL_TOKEN_MARKERS = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|end|>",
)


def _clean_generated_text(text: str) -> str:
    """Truncates at the first leaked control token (a sign the model kept
    generating past the intended end of its turn) and strips any that
    remain in what's left, as a last line of defense - pulled out as a
    plain function so it's unit-testable without mlx_lm installed."""
    cutoff = len(text)
    for marker in _MLX_CONTROL_TOKEN_MARKERS:
        marker_index = text.find(marker)
        if marker_index != -1:
            cutoff = min(cutoff, marker_index)

    cleaned = text[:cutoff]
    for marker in _MLX_CONTROL_TOKEN_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def ensure_mlx_eos_tokens(tokenizer) -> None:
    """Registers any turn-end token the loaded chat model uses (Llama-3's
    "<|eot_id|>", Phi-3's "<|end|>", ChatML's "<|im_end|>") as an EOS token
    on an mlx-lm tokenizer, if mlx_lm didn't already pick it up on its own.
    See the note in DomainLLM._load_mlx for why this matters and
    src/llm/fine_tuning/compare_base_vs_finetuned.py, which also needs this
    applied to the *base* (non-adapter) model for a fair comparison."""
    for extra_eos_token in ("<|eot_id|>", "<|end|>", "<|im_end|>"):
        if extra_eos_token in tokenizer.get_vocab():
            try:
                tokenizer.add_eos_token(extra_eos_token)
            except ValueError:
                pass


def generate_with_mlx_model(model, tokenizer, instruction: str, input_text: str, context: str | None = None) -> str:
    """The actual mlx-lm generation call, factored out of
    DomainLLM._generate_with_mlx so src/llm/fine_tuning/compare_base_vs_finetuned.py
    can run the exact same prompt construction/cleanup against two different
    loaded models (base vs. fine-tuned) instead of duplicating this logic."""
    from mlx_lm import generate as mlx_generate

    combined_input = f"Contexto:\n{context}\n\nPergunta: {input_text}" if context else input_text
    # Mirrors how src/llm/fine_tuning/prepare_mlx_dataset.py built the
    # training prompts ("instruction\n\ninput") so inference matches
    # training format, wrapped as a chat turn since `mlx_lm.lora` applies
    # the model's own chat template to "completions"-format training data
    # too (see notebooks/fine_tuning_local_mlx.ipynb).
    messages = [{"role": "user", "content": f"{instruction}\n\n{combined_input}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw_text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    return _clean_generated_text(raw_text)


class DomainLLM:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(
            model_path or os.getenv("FINE_TUNED_MODEL_PATH", "models/demo_retrieval")
        )
        if not self.model_path.is_absolute():
            self.model_path = BASE_DIR / self.model_path

        self.backend = None
        self._load_backend()

    def _load_backend(self):
        kind = _detect_backend_kind(self.model_path)
        if kind == "lora_hf":
            self._load_real_lora()
        elif kind == "lora_mlx":
            self._load_mlx()
        elif kind == "retrieval_demo":
            self._load_demo_index(self.model_path / "tfidf_index.pkl")

    def _load_real_lora(self):
        # Local import: these dependencies are only needed for this backend.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with open(self.model_path / "adapter_config.json") as f:
            base_model_name = json.load(f)["base_model_name_or_path"]

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, self.model_path)

        self.backend = "lora"
        self._tokenizer = tokenizer
        self._model = model
        self._base_model_name = base_model_name

    def _load_mlx(self):
        # Local import: mlx-lm only installs/runs on Apple Silicon (Metal) -
        # see notebooks/fine_tuning_local_mlx.ipynb. Importing it eagerly at
        # module load time would break every other backend on non-Mac
        # machines (including this project's own test suite/CI).
        from mlx_lm import load as mlx_load

        with open(self.model_path / "adapter_config.json", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["model"]

        model, tokenizer = mlx_load(base_model_name, adapter_path=str(self.model_path))

        # Llama-3-Instruct (and some other chat models) end each turn with a
        # dedicated token - e.g. "<|eot_id|>" for Llama-3 - that is distinct
        # from the tokenizer's default end-of-text token. If mlx_lm doesn't
        # already know about it (depends on the specific model conversion's
        # generation_config.json), generation never recognizes the natural
        # stopping point and keeps going, leaking that token (and further
        # hallucinated "turns") as literal text into the response - this is
        # what _clean_generated_text is also a safety net against.
        ensure_mlx_eos_tokens(tokenizer)

        self.backend = "mlx"
        self._mlx_model = model
        self._mlx_tokenizer = tokenizer
        self._base_model_name = base_model_name

    def _load_demo_index(self, tfidf_index_path: Path):
        with open(tfidf_index_path, "rb") as f:
            index = pickle.load(f)

        self.backend = "retrieval_demo"
        self._vectorizer = index["vectorizer"]
        self._tfidf_matrix = index["tfidf_matrix"]
        self._answers = index["answers"]

    def describe(self) -> str:
        if self.backend == "lora":
            return f"LoRA fine-tuned, via transformers/peft ({self._base_model_name}) em {self.model_path}"
        if self.backend == "mlx":
            return f"LoRA fine-tuned, via mlx-lm ({self._base_model_name}) em {self.model_path}"
        if self.backend == "retrieval_demo":
            return f"Indice de recuperacao (demo, sem GPU) em {self.model_path}"
        return "backend nao carregado"

    def generate_response(self, instruction: str, input_text: str, context: str | None = None) -> str:
        """`context` is the text already retrieved by RAG (see
        src/langchain_app/prompts.py), when available. Passing the context
        lets the response rely exactly on the snippets that RAG ranked as
        most relevant, instead of the domain model doing its own search and
        potentially diverging."""
        if self.backend == "lora":
            return self._generate_with_lora(instruction, input_text, context)
        if self.backend == "mlx":
            return self._generate_with_mlx(instruction, input_text, context)
        if self.backend == "retrieval_demo":
            return self._generate_with_demo_index(instruction, input_text, context)
        raise RuntimeError("DomainLLM sem backend carregado.")

    def _generate_with_lora(self, instruction: str, input_text: str, context: str | None = None) -> str:
        combined_input = f"Contexto:\n{context}\n\nPergunta: {input_text}" if context else input_text
        alpaca_prompt = (
            "Below is an instruction that describes a task, paired with an "
            "input that provides further context. Write a response that "
            "appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
        )
        prompt = alpaca_prompt.format(instruction, combined_input, "")
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=200, use_cache=True)
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("### Response:")[-1].strip()

    def _generate_with_mlx(self, instruction: str, input_text: str, context: str | None = None) -> str:
        return generate_with_mlx_model(self._mlx_model, self._mlx_tokenizer, instruction, input_text, context)

    def _generate_with_demo_index(self, instruction: str, input_text: str, context: str | None = None) -> str:
        from sklearn.metrics.pairwise import cosine_similarity

        if context:
            # Uses the top-ranked snippet that RAG itself already retrieved
            # (context comes ordered by similarity - see
            # HospitalKnowledgeBase.search). Re-ranking again here, with a
            # TF-IDF fitted only on those 2-3 snippets, is unstable (IDF
            # statistics are unreliable with so few documents) and could
            # diverge from what RAG flagged as most relevant - so we simply
            # trust the top-ranked one.
            #
            # `context` may carry patient info before the RAG chunks (see
            # MedicalAssistantChain.invoke) - anything before the first
            # "[Fonte:" marker is deliberately ignored here rather than
            # treated as a chunk.
            fonte_index = context.find("[Fonte:")
            rag_only_context = context[fonte_index:] if fonte_index != -1 else context
            chunks = [t.strip() for t in rag_only_context.split("[Fonte:") if t.strip()]
            if chunks:
                best_chunk = chunks[0]
                if "]" in best_chunk.split("\n")[0]:
                    best_chunk = best_chunk.split("]", 1)[1].strip()
                return best_chunk

        # No context (e.g. called directly by src/fine_tuning/evaluate.py):
        # searches the fine-tuning dataset itself, indexed during demo training.
        query = f"{instruction} {input_text}"
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._tfidf_matrix)[0]
        best_index = similarities.argmax()
        confidence = float(similarities[best_index])

        answer = self._answers[best_index]
        warning_note = (
            ""
            if confidence > 0.15
            else "\n\n[aviso: baixa similaridade com o dataset de treino - "
                 "resposta pode nao ser relevante para esta pergunta]"
        )
        return answer + warning_note
