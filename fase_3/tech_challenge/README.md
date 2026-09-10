# Assistente Médico Virtual — Tech Challenge Fase 3 (Generative AI)

Implementação da solução especificada em `docs/Documento de Especificacoes - Tech Challenge Fase 3.pdf`
(pasta `fase_3_study` do curso): um assistente clínico virtual para um hospital
fictício, combinando **fine-tuning de LLM**, **LangChain** e **LangGraph**,
com segurança, logging e explainability.

## O que este projeto entrega, por requisito do desafio

| # | Requisito do desafio | Onde está implementado |
|---|---|---|
| 1 | Fine-tuning de LLM com dados médicos internos (preprocessing, anonimização, curadoria) | `src/llm/data_prep/` (dados + anonimização) + `notebooks/fine_tuning_colab.ipynb` (treino real, GPU) |
| 2 | Assistente com LangChain: pipeline + consulta a base estruturada + contexto do paciente | `src/llm/langchain_app/` |
| 3 | Segurança e validação: limites de atuação, logging, explainability | `src/llm/guardrails/`, `src/llm/logging_utils/`, fontes citadas em `src/llm/rag/` |
| 4 | Código modularizado em Python + README | esta estrutura + este arquivo |
| — | Fluxos de decisão automatizados coordenados com LangChain/LangGraph | `src/llm/langgraph_flow/` (grafo de 5 nós) |

## Arquitetura (resumo)

```
[pergunta do médico + patient_id]
        |
        v
(1) Verificador de Exames  --> consulta SQLite de exames (LangChain Document Loader)
        |
        v
(2) Contexto (RAG)         --> busca em protocolos/FAQs/laudos + prontuário do
                               paciente (LangChain TextLoader), com fonte rastreada
        |
        v
(3) Sugestão de Conduta    --> LLM de domínio (fine-tuned ou demo) + contexto do RAG
        |
        v
(4) Guardrails             --> bloqueia linguagem de prescrição direta, anexa disclaimer
        |
        v
(5) Alertas / Log          --> grava log de auditoria, marca pendência de validação humana
        |
        v
[resposta + fontes + nível de confiança + pendência de validação médica]
```

Implementado com **LangGraph** (`src/llm/langgraph_flow/graph.py`), sobre uma
**chain LangChain** (`src/llm/langchain_app/chains.py`) que integra a **LLM de
domínio** (`src/llm/models/domain_llm.py`) com o **índice RAG**
(`src/llm/rag/vector_store.py`).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # preencha OPENAI_API_KEY se for usar (ver abaixo)
```

Requer Python 3.9+. `requirements.txt` usa faixas de versão (não pins exatos)
porque `langchain` 1.x passou a exigir Python >=3.10 - em Python 3.9 o pip
resolve automaticamente para a última versão 0.3.x compatível, testada e
funcional com este projeto (14/14 testes e demo passando nessa combinação).

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

# 5) "treina" o modelo de domínio usado por padrão (ver seção abaixo)
python -m src.llm.fine_tuning.train_demo_cpu
```

## Rodando a demonstração

```bash
python run_demo.py --listar-pacientes
python run_demo.py --paciente PAC0001 --pergunta "Qual a conduta recomendada para este paciente?"
```

Isso executa o grafo LangGraph completo e imprime cada etapa (verificação de
exames, contexto recuperado, sugestão de conduta, guardrails, alerta final),
além de gravar `logs/audit_log.jsonl` com o log de auditoria de cada nó — use
esse script como roteiro para o vídeo de demonstração pedido no desafio.

Rodar os testes automatizados:

```bash
pytest -q
```

## Rodando a UI do médico (Angular + API Node + serviço Python)

Além do `run_demo.py` (CLI original), o projeto tem uma interface web para o
médico (ver `docs/Documento de Especificacoes - UI do Assistente Medico`).
São 3 processos separados, cada um em um terminal, nesta ordem:

```bash
# 1) serviço interno Python (FastAPI) - mantém o grafo LangGraph carregado em
#    memória; só a API Node fala com ele, nunca o navegador diretamente
source venv/bin/activate
uvicorn src.llm.service.app:app --port 8001

# 2) API Node.js (gateway) - único ponto que o Angular chama
cd src/api
npm install
npm start                     # sobe em http://localhost:3000

# 3) SPA Angular - interface do médico
cd src/ui
npm install                   # se der erro de peer deps, use: npm install --legacy-peer-deps
npm start                     # sobe em http://localhost:4200
```

Abra `http://localhost:4200`: Tela 1 lista os pacientes (via `GET /patients`)
e também oferece "Fazer uma pergunta geral (sem selecionar paciente)", para
dúvidas sem relação com um paciente específico (ex.: um protocolo interno)
— quando a pergunta é sobre um paciente, o médico continua só selecionando-o
na lista, sem etapa extra. Tela 2 recebe a pergunta do médico (com ou sem
paciente associado), Tela 3 mostra a resposta já filtrada (`POST /ask`) —
nunca nível de confiança, informação de guardrail ou alerta de validação
humana (ver `src/llm/service/doctor_view.py`, o único ponto do código que
decide o que o médico pode ver). `patient_id` é opcional em toda a cadeia
(API Node, serviço Python, LangGraph) — sem paciente, o nó de verificação de
exames é pulado e a sugestão de conduta usa só o contexto do RAG.

Nota: `npm install` em `src/ui` pode falhar com
`Cannot read properties of null (reading 'edgesOut')` — é um bug conhecido do
resolvedor de dependências do npm (`@npmcli/arborist`) ao lidar com o grafo de
peer dependencies do Angular 22/Vitest, não um problema deste projeto. Rode de
novo com `npm install --legacy-peer-deps` que resolve.

## Sobre o fine-tuning: real (Colab) vs. demo (local)

O desafio pede fine-tuning de verdade de um LLM (ex.: LLaMA), o que exige
GPU. Este repositório traz **dois caminhos**, plugáveis via
`FINE_TUNED_MODEL_PATH` no `.env`:

- **`notebooks/fine_tuning_colab.ipynb`** — o fine-tuning real: Unsloth +
  Llama-3-8b (4-bit) + LoRA, no mesmo padrão do material de referência do
  curso (`fase_3_study/docs/fine-tuning-rag-documentos-fiap-main`). Rode no
  Google Colab com GPU; ao final, salva um adaptador LoRA em
  `models/fine_tuned_lora/`. Aponte `FINE_TUNED_MODEL_PATH` para essa pasta
  e `src/llm/models/domain_llm.py` carrega automaticamente via
  `transformers` + `peft`.
- **`src/llm/fine_tuning/train_demo_cpu.py`** — uma demonstração que roda em
  qualquer máquina, sem GPU e sem baixar nenhum modelo (útil para
  desenvolvimento, testes automatizados e para validar o pipeline ponta a
  ponta antes de gastar tempo de GPU). Em vez de uma rede neural, usa um
  índice de recuperação (TF-IDF) sobre o próprio dataset de fine-tuning:
  dada uma pergunta, devolve o exemplo de treino mais parecido. **Isto não é
  o modelo fine-tuned exigido pelo desafio** — é um substituto leve para
  desenvolvimento; o relatório técnico final deve reportar os resultados do
  notebook do Colab.

- **`notebooks/fine_tuning_local_mlx.ipynb`** — uma terceira opção, para
  quem não tem GPU NVIDIA/Colab mas tem um **Mac com Apple Silicon**
  (M1 ou mais recente): fine-tuning LoRA real, rodando localmente na GPU do
  Mac via [MLX](https://github.com/ml-explore/mlx-lm) (framework da Apple,
  Metal), sem depender de CUDA/nuvem. Requer macOS 14+, Python arm64 nativo
  e ~16GB+ de memória unificada (o notebook confere isso e recomenda um
  modelo menor se a RAM for justa). Ao final, salva o adaptador em
  `models/fine_tuned_lora_mlx/` — aponte `FINE_TUNED_MODEL_PATH` para essa
  pasta e `src/llm/models/domain_llm.py` reconhece automaticamente o formato
  mlx-lm (distinto do formato Hugging Face/PEFT do notebook do Colab) e
  carrega esse backend.

`src/llm/models/domain_llm.py` decide automaticamente qual dos três usar,
com base no que existir em `FINE_TUNED_MODEL_PATH` — o resto do sistema
(RAG, LangChain, LangGraph) não muda em nenhum dos três casos.

## Avaliando o que o fine-tuning realmente contribuiu

Com o RAG no meio do caminho, é difícil enxergar o efeito do fine-tuning: o
RAG já entrega ao modelo o texto do protocolo relevante, e qualquer modelo
instruct razoável já responde bem só de ler esse contexto. Para isolar o
que veio de fato do fine-tuning (e não do RAG), o dataset inclui uma
condição **exclusiva do fine-tuning**:
`src/llm/data_prep/build_fine_tuning_dataset.py::FINE_TUNING_ONLY_CONDITIONS`
(hoje, "Crise Asmática Aguda") gera exemplos de treino com os mesmos
templates usados para os protocolos normais, mas **nunca grava nada em
`data/raw/`** — então `HospitalKnowledgeBase` (RAG) não tem nenhum
documento sobre essa condição. Perguntando sobre ela, o RAG não recupera
contexto relevante nenhum; só um modelo que realmente aprendeu esse
conteúdo durante o fine-tuning consegue responder corretamente.

`src/llm/fine_tuning/compare_base_vs_finetuned.py` usa exatamente isso:
carrega o modelo base (sem adaptador) e o modelo fine-tuned (com o
adaptador do MLX) lado a lado, e roda as mesmas perguntas nos dois —
primeiro as exclusivas do fine-tuning (sem contexto do RAG, onde a
diferença tende a ser grande) e depois algumas perguntas normais com
contexto do RAG (onde a diferença tende a ser só de tom/formato). Requer
macOS Apple Silicon, como o notebook de treino:

```bash
python -m src.llm.fine_tuning.compare_base_vs_finetuned
```

Como o RAG não tem nenhum documento sobre essas condições, uma pergunta
sobre elas via `MedicalAssistantChain`/UI (não só o script de comparação)
também precisa de tratamento: TF-IDF nunca diz "nada relevante encontrado"
— sempre devolve os k documentos mais próximos, mesmo que nenhum tenha
relação real com a pergunta. Sem um paciente para ancorar a busca (ver
próxima seção), isso já retornou uma fonte completamente errada (ex.: uma
pergunta geral sobre "crise asmática" trouxe uma FAQ de "Câncer de Mama").

Por isso `chains.py::_FinetuningOnlyMatcher` reconhece essas perguntas e
pula o RAG inteiramente para elas, deixando o modelo de domínio responder
só com o que aprendeu no fine-tuning (exatamente o mesmo cenário do script
de comparação acima). Esse reconhecimento é **derivado dos dados, não
hardcoded**: todo exemplo em `data/processed/dataset_fine_tuning.jsonl`
carrega um campo `in_rag` (gravado por `build_fine_tuning_dataset.py`,
`True` quando o mesmo conteúdo também existe em `data/raw/`, `False`
quando não existe — PubMedQA/MedQuAD e `FINE_TUNING_ONLY_CONDITIONS`).
`_FinetuningOnlyMatcher` compara a pergunta do médico (por TF-IDF) contra
os exemplos `in_rag=False`, usando pergunta+resposta de cada exemplo (não
só a pergunta) — comparar só perguntas fazia o template da pergunta
alternativa de `FINE_TUNING_ONLY_CONDITIONS` ("Quais os próximos passos
para um paciente com X?") dominar o score para qualquer condição X, já que
esse template não é usado em nenhum outro exemplo do dataset e por isso é
a sequência mais "rara" (maior peso IDF) do corpus inteiro; incluir a
resposta traz vocabulário clínico específico da condição, que domina o
score corretamente. Adicionar uma nova condição exclusiva não exige
nenhuma mudança neste arquivo — só regenerar o dataset.

Se você adicionar mais condições exclusivas em
`FINE_TUNING_ONLY_CONDITIONS`, rode de novo
`python -m src.llm.data_prep.build_fine_tuning_dataset`, depois
`python -m src.llm.fine_tuning.train_demo_cpu` (para o backend demo
reconhecer a condição nova) e depois o notebook
de fine-tuning (`fine_tuning_local_mlx.ipynb`) para o adaptador aprender o
conteúdo novo antes de comparar.

## Sobre embeddings do RAG

Pelo mesmo motivo (sem acesso a modelos via internet em todo ambiente), o
índice RAG (`src/llm/rag/embeddings.py`) usa embeddings TF-IDF por padrão, em vez
de um modelo neural (ex.: `sentence-transformers`, usado no material de
referência via `InstructorEmbedding`). Isso é lexical, não semântico:
encontra bem documentos que compartilham vocabulário com a pergunta, mas não
generaliza sinônimos. O vetorizador usa bigramas e uma lista de stopwords em
português (em vez da configuração padrão do `TfidfVectorizer`, que não tem
stopwords para português) para reduzir esse efeito — mas isso ameniza, não
elimina: documentos mais longos/verbosos (ex.: "Câncer de Mama", com 3
variações por estágio) ainda podem vencer por pura sobreposição de palavras
genéricas ("tratamento", "avaliação") mesmo quando um termo bem mais
específico da pergunta (ex.: "diabetes") só aparece no documento certo,
porque esse termo raro pesa menos que vários termos comuns repetidos. Para
produção, troque `TfidfEmbeddings` por `HuggingFaceEmbeddings`
(langchain-community) com um modelo multilíngue — nenhum outro módulo
precisa mudar.

## Sobre a camada de orquestração (OpenAI)

`OPENAI_API_KEY` no `.env` fica disponível para usar `langchain-openai` em
chains/agents de orquestração mais sofisticados (ex.: um agente real com
tool-calling, em vez do `ExamVerificationAgent` baseado em regras). Na
implementação atual, a lógica de orquestração é determinística (não depende
de nenhum LLM de terceiros para decidir o fluxo) — o `OPENAI_API_KEY` é o
ponto de extensão para quem quiser evoluir os nós do grafo para agentes mais
autônomos.

## Nota técnica: SQLite em pastas sincronizadas (OneDrive/Google Drive)

Se você mover este projeto para uma pasta sincronizada na nuvem (OneDrive,
Google Drive etc.), **não abra `data/db/hospital.sqlite3` diretamente com
uma conexão de escrita a partir dessa pasta** — alguns provedores de
sincronização impedem operações de lock/journal do SQLite, e o arquivo pode
ficar com um lock residual que passa a bloquear até `rm`/`unlink` nele (o
processo Python não trava, mas o arquivo fica "preso" para sempre nessa
pasta). `src/llm/data_prep/generate_synthetic_patients.py` já contorna isso:
monta o banco em um arquivo temporário local e só copia (sobrescrevendo) o
resultado final para `data/db/` — nunca abre o SQLite direto na pasta
sincronizada. Se for mexer nesse script, mantenha esse padrão. Consultas de
leitura (`sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)`, usado em
`src/llm/langchain_app/document_loaders.py`) não têm esse problema.

## Estrutura do repositório

```
tech_challenge/
├── README.md
├── requirements.txt
├── .env.example
├── run_demo.py                 # CLI original (continua funcionando, sem UI)
├── data/
│   ├── raw/                  # protocolos, FAQs, laudos, amostras públicas
│   ├── processed/            # dataset de fine-tuning, CSV/prontuários de pacientes
│   └── db/                   # hospital.sqlite3 (pacientes, exames, prontuário)
├── models/                   # adaptador LoRA (real ou demo) - gerado, não versionado
├── src/
│   ├── llm/                  # todo o agente de IA (fine-tuning, RAG, LangChain, LangGraph)
│   │   ├── data_prep/         # geração de dados sintéticos, anonimização, curadoria
│   │   ├── fine_tuning/       # treino demo (CPU), conversão de dataset p/ mlx-lm, avaliação e comparação base vs. fine-tuned
│   │   ├── rag/               # embeddings TF-IDF + índice vetorial (Chroma)
│   │   ├── langchain_app/     # document loaders, prompts, chain, agent
│   │   ├── langgraph_flow/    # state, nós e o grafo (StateGraph)
│   │   ├── guardrails/        # regras de segurança/validação
│   │   ├── logging_utils/     # log de auditoria estruturado
│   │   ├── models/            # DomainLLM (abstração do modelo de domínio)
│   │   └── service/           # serviço HTTP interno (FastAPI) - ver docs/UI
│   ├── api/                   # API Node.js - gateway entre o Angular e src/llm/service
│   └── ui/                    # SPA Angular - interface do médico
├── notebooks/
│   ├── fine_tuning_colab.ipynb      # fine-tuning real (Unsloth + Llama-3-8b + LoRA, GPU NVIDIA/Colab)
│   └── fine_tuning_local_mlx.ipynb  # fine-tuning real, local (MLX, Mac Apple Silicon)
├── tests/                    # pytest (cobre src/llm/)
├── docs/                     # relatório técnico, diagrama e especificação da UI
├── video/                    # vídeo de demonstração (a gravar, ≤15 min)
└── logs/
    └── audit_log.jsonl       # log de auditoria (gerado ao rodar o fluxo)
```

## O que falta para a entrega final do Tech Challenge

Este repositório entrega a arquitetura funcional e testável. Para a entrega
do desafio, ainda faltam (ver Seção 10 do documento de especificações):

- [ ] Rodar `notebooks/fine_tuning_colab.ipynb` no Colab com GPU (ou
      `notebooks/fine_tuning_local_mlx.ipynb` localmente, num Mac Apple
      Silicon) e trazer o adaptador real para `models/fine_tuned_lora/` ou
      `models/fine_tuned_lora_mlx/`.
- [ ] Rodar `src/llm/fine_tuning/evaluate.py` com o modelo real e registrar as
      métricas no relatório técnico.
- [ ] Escrever o relatório técnico (`docs/relatorio_tecnico.pdf`): processo
      de fine-tuning, descrição do assistente, diagrama do fluxo
      LangChain/LangGraph, avaliação do modelo.
- [ ] Gravar o vídeo de demonstração (≤15 min) usando `run_demo.py` como
      roteiro.
- [ ] Publicar o repositório Git (o enunciado pede um repositório Git como
      entregável).
