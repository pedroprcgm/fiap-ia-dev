"""Builds notebooks/fine_tuning_local_mlx.ipynb programmatically (nbformat),
mirroring the structure of notebooks/fine_tuning_colab.ipynb but for local
fine-tuning on a Mac (Apple Silicon) via mlx-lm instead of Unsloth/Colab."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    "# Fine-tuning local do Assistente Medico (MLX) - Hospital Pos Tech\n"
    "Tech Challenge Fase 3 (Generative AI)\n\n"
    "Alternativa a `notebooks/fine_tuning_colab.ipynb` para quem nao tem GPU "
    "NVIDIA/Colab, mas tem um **Mac com Apple Silicon** (M1 ou mais recente). "
    "Em vez de Unsloth + bitsandbytes (que exigem CUDA e por isso nao rodam "
    "aqui), usa o [MLX](https://github.com/ml-explore/mlx-lm) da Apple, que "
    "faz fine-tuning LoRA na GPU do Mac via Metal.\n\n"
    "**Requisitos:**\n"
    "- macOS 14+ e Mac Apple Silicon (M1/M2/M3/M4). Nao funciona em Mac Intel "
    "nem em Windows/Linux sem GPU Apple.\n"
    "- Python **arm64 nativo** (nao rodando sob Rosetta) - a celula abaixo confere isso.\n"
    "- Memoria unificada: um modelo 7-8B em 4-bit usa uns 4-5GB so de pesos, "
    "mais espaco para o treino em si. 16GB e o minimo justo (use um modelo "
    "menor, ex. Phi-3-mini); 32GB roda o Llama-3-8B com folga.\n"
    "- Rode este notebook localmente (Jupyter/VS Code), nao no Colab - o "
    "Colab nao te da uma GPU Apple.\n\n"
    "**Isto e complementar, nao substitui** `notebooks/fine_tuning_colab.ipynb`: "
    "se voce tiver acesso a GPU NVIDIA (Colab ou propria), aquele caminho "
    "(Llama-3-8B via Unsloth) e o que o desafio original tinha em mente. Este "
    "notebook existe para quem quer rodar um fine-tuning real, sem depender "
    "de nuvem, usando o hardware que o Mac ja tem."
)

code(
    "import platform\n"
    "import subprocess\n\n"
    "print(f\"Maquina: {platform.machine()} (precisa ser 'arm64', não 'x86_64')\")\n"
    "print(f\"Sistema: {platform.system()} {platform.mac_ver()[0]}\")\n\n"
    "mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())\n"
    "print(f\"Memoria unificada: {mem_bytes / 1024**3:.1f} GB\")\n\n"
    "assert platform.machine() == \"arm64\", (\n"
    "    \"Python rodando como x86_64 (provavelmente sob Rosetta) - reinstale um \"\n"
    "    \"Python arm64 nativo (ex.: via Homebrew) antes de continuar.\"\n"
    ")"
)

md(
    "## Ajuste o caminho do projeto\n\n"
    "Diferente do notebook do Colab, aqui rodamos direto na pasta do projeto "
    "no seu Mac - sem montar Google Drive."
)

code(
    "from pathlib import Path\n\n"
    "PROJECT_DIR = Path.cwd().parent  # ajuste se rodar este notebook de outro lugar\n"
    "assert (PROJECT_DIR / \"requirements.txt\").exists(), (\n"
    "    f\"'{PROJECT_DIR}' nao parece a raiz do projeto - ajuste PROJECT_DIR acima.\"\n"
    ")\n"
    "print(f\"PROJECT_DIR = {PROJECT_DIR}\")"
)

md(
    "## Instalando o mlx-lm\n\n"
    "`mlx-lm[train]` traz o MLX (framework Apple, acelerado por Metal) e o "
    "comando `mlx_lm.lora` usado para o fine-tuning."
)

code('%pip install -U "mlx-lm[train]"')

md(
    "## Escolhendo o modelo base\n\n"
    "Em vez de `unsloth/llama-3-8b-bnb-4bit` (CUDA-only), usamos uma versao "
    "ja convertida para MLX. Ajuste conforme sua RAM:\n\n"
    "- `mlx-community/Meta-Llama-3-8B-Instruct-4bit` - mesma familia do "
    "notebook do Colab; confortavel com 32GB+, justo com 16GB.\n"
    "- `mlx-community/Phi-3-mini-4k-instruct-4bit` - bem mais leve (3.8B), "
    "recomendado se sua Mac tiver 16GB ou menos.\n\n"
    "Os pesos sao baixados automaticamente do Hugging Face na primeira execucao."
)

code(
    'MODEL_NAME = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"  # troque por Phi-3-mini se a RAM for justa\n'
    'ADAPTER_PATH = PROJECT_DIR / "models" / "fine_tuned_lora_mlx"\n'
    'DATA_DIR = PROJECT_DIR / "data" / "processed" / "mlx_finetune"'
)

md(
    "## Preparando o dataset\n\n"
    "O dataset de fine-tuning (`data/processed/dataset_fine_tuning.jsonl`) ja "
    "existe no formato instruction/input/output (Alpaca), o mesmo usado pelo "
    "notebook do Colab - ver `src/llm/data_prep/build_fine_tuning_dataset.py`. "
    "O `mlx_lm.lora` espera outro formato (`prompt`/`completion`, separado em "
    "`train.jsonl`/`valid.jsonl`), entao convertemos com o script dedicado a isso."
)

code(
    "import sys\n\n"
    "sys.path.insert(0, str(PROJECT_DIR))\n"
    "from src.llm.fine_tuning.prepare_mlx_dataset import convert\n\n"
    "convert(\n"
    "    dataset_path=PROJECT_DIR / \"data\" / \"processed\" / \"dataset_fine_tuning.jsonl\",\n"
    "    output_dir=DATA_DIR,\n"
    ")"
)

md(
    "## Fine-tuning (LoRA)\n\n"
    "`mlx_lm.lora` e um comando de CLI - chamamos com `!` a partir do "
    "notebook, como o notebook do Colab ja fazia com `!pip install`. "
    "`--num-layers` e `--batch-size` menores reduzem o uso de memoria (ver "
    "[LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#memory-issues) "
    "se a Mac travar ou o processo for `Killed`).\n\n"
    "Dataset pequeno (~20 exemplos) - por isso `--iters` alto e `--num-layers` "
    "baixo: poucas camadas, mais passos sobre os mesmos exemplos, no mesmo "
    "espirito do `num_train_epochs=3` do notebook do Colab."
)

code(
    # Paths are quoted because the project folder can contain spaces (e.g. a
    # "Pos IA" OneDrive folder) - unquoted {DATA_DIR}/{ADAPTER_PATH}
    # interpolation breaks the shell's argument parsing in that case.
    "!mlx_lm.lora \\\n"
    '    --model "{MODEL_NAME}" \\\n'
    "    --train \\\n"
    '    --data "{DATA_DIR}" \\\n'
    "    --iters 200 \\\n"
    "    --num-layers 8 \\\n"
    "    --batch-size 1 \\\n"
    '    --adapter-path "{ADAPTER_PATH}"'
)

md(
    "## Testando a inferencia com um caso clinico de exemplo\n\n"
    "Mesma pergunta de teste usada no notebook do Colab, para comparar."
)

code(
    "from mlx_lm import load, generate\n\n"
    "model, tokenizer = load(MODEL_NAME, adapter_path=str(ADAPTER_PATH))\n\n"
    "pergunta_teste = \"Qual o protocolo interno para Diabetes Mellitus tipo 2?\"\n"
    "instrucao = \"Responda como assistente clinico do hospital, com base no protocolo interno.\"\n\n"
    "messages = [{\"role\": \"user\", \"content\": f\"{instrucao}\\n\\n{pergunta_teste}\"}]\n"
    "prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)\n\n"
    "resposta = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=True)"
)

md(
    "## Usando o adaptador no resto do projeto\n\n"
    "Aponte `FINE_TUNED_MODEL_PATH` no `.env` para a pasta do adaptador:\n\n"
    "```\n"
    "FINE_TUNED_MODEL_PATH=models/fine_tuned_lora_mlx\n"
    "```\n\n"
    "`src/llm/models/domain_llm.py` detecta automaticamente que e um "
    "adaptador mlx-lm (pelo arquivo `adapters.safetensors` e pela chave "
    "`lora_parameters` no `adapter_config.json`, diferentes do formato "
    "Hugging Face/PEFT do notebook do Colab) e passa a gerar respostas com "
    "ele - nenhum outro modulo (RAG, LangChain, LangGraph, a UI) precisa mudar."
)

md(
    "## Avaliacao do modelo\n\n"
    "Para o relatorio tecnico do desafio, compare respostas do modelo base "
    "vs. fine-tuned para as mesmas perguntas do conjunto de teste, e registre:\n"
    "- aderencia ao protocolo interno (avaliacao qualitativa manual);\n"
    "- perplexidade no conjunto de validacao - `mlx_lm.lora --model \"{MODEL_NAME}\" "
    "--adapter-path \"{ADAPTER_PATH}\" --data \"{DATA_DIR}\" --test` imprime a perplexidade "
    "de teste diretamente (lembre das aspas nos caminhos se rodar direto no terminal - "
    "o caminho do projeto tem espaco);\n"
    "- ROUGE-L entre a resposta gerada e a resposta de referencia.\n\n"
    "Um esqueleto de avaliacao fica em `src/llm/fine_tuning/evaluate.py` "
    "(hoje escrito para o backend `transformers`/`peft` - adaptar para chamar "
    "`DomainLLM.generate_response` diretamente funciona com qualquer backend, "
    "incluindo este)."
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

nbf.validate(nb)

with open("/tmp/fine_tuning_local_mlx.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook criado e validado com sucesso.")
