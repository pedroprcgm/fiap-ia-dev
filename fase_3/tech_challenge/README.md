# Assistente Médico Virtual — Tech Challenge Fase 3 (Generative AI)

Instruções de execução do projeto.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # preencha OPENAI_API_KEY se for usar
```

Requer Python 3.9+.

## Gerando os dados e "treinando" o modelo (ordem importa)

```bash
# 1) dados sintéticos do hospital (protocolos, FAQs, modelos de laudo)
python -m src.llm.data_prep.generate_synthetic_hospital_data

# 2) amostra dos datasets públicos sugeridos no desafio (PubMedQA/MedQuAD)
python -m src.llm.data_prep.prepare_public_datasets

# 3) consolida tudo (com anonimização e curadoria) no dataset de fine-tuning
python -m src.llm.data_prep.build_fine_tuning_dataset

# 4) base sintética de pacientes (prontuários, exames, histórico)
python -m src.llm.data_prep.generate_synthetic_patients

# 5) "treina" o modelo de domínio usado por padrão (backend demo, sem GPU)
python -m src.llm.fine_tuning.train_demo_cpu
```

## Fine-tuning real (opcional, além do backend demo)

Plugável via `FINE_TUNED_MODEL_PATH` no `.env`:

- **Colab (GPU NVIDIA)**: rode `notebooks/fine_tuning_colab.ipynb`; ao final,
  aponte `FINE_TUNED_MODEL_PATH=models/fine_tuned_lora`.
- **Local, Mac Apple Silicon (MLX)**: rode `notebooks/fine_tuning_local_mlx.ipynb`;
  ao final, aponte `FINE_TUNED_MODEL_PATH=models/fine_tuned_lora_mlx`.

Comparar modelo base vs. fine-tuned (requer Mac Apple Silicon):

```bash
python -m src.llm.fine_tuning.compare_base_vs_finetuned
```

## Rodando a demonstração (CLI)

```bash
python run_demo.py --listar-pacientes
python run_demo.py --paciente PAC0001 --pergunta "Qual a conduta recomendada para este paciente?"
```

Grava `logs/audit_log.jsonl` com o log de auditoria de cada etapa.

Rodar os testes automatizados:

```bash
pytest -q
```

## Rodando a UI do médico (Angular + API Node + serviço Python)

3 processos separados, cada um em um terminal, nesta ordem:

```bash
# 1) serviço interno Python (FastAPI)
source venv/bin/activate
uvicorn src.llm.service.app:app --port 8001

# 2) API Node.js (gateway)
cd src/api
npm install
npm start                     # http://localhost:3000

# 3) SPA Angular
cd src/ui
npm install                   # se der erro de peer deps: npm install --legacy-peer-deps
npm start                     # http://localhost:4200
```

Abra `http://localhost:4200`.

Nota: `npm install` em `src/ui` pode falhar com
`Cannot read properties of null (reading 'edgesOut')` (bug conhecido do
`@npmcli/arborist` com o grafo de peer deps do Angular 22/Vitest). Rode de
novo com `npm install --legacy-peer-deps`.

## Nota: SQLite em pastas sincronizadas (OneDrive/Google Drive)

Não abra `data/db/hospital.sqlite3` diretamente com uma conexão de escrita a
partir de uma pasta sincronizada na nuvem — pode deixar um lock residual que
bloqueia o arquivo. Os scripts de geração de dados já contornam isso (montam
o banco em um arquivo temporário local e só copiam o resultado final). As
consultas de leitura do restante do projeto não têm esse problema.
