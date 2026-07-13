# Tech Challenge – Fase 2: Otimização de Modelos de Diagnóstico

## Objetivo

Otimizar os hiperparâmetros dos modelos de diagnóstico do Módulo 1 usando um Algoritmo Genético (GA) e integrar uma LLM para tornar os resultados interpretáveis para médicos.

## Decisões de projeto

- **Gerenciador de dependências:** Pipenv
- **Provedor de nuvem:** Azure (pendente)
- **LLM:** API hospedada (OpenAI GPT)

## Estrutura do projeto

```
.
├── src/
│   ├── ga/         # Algoritmo genético: codificação, seleção, cruzamento, mutação, fitness
│   ├── llm/         # Integração com a LLM para geração de explicações
│   └── models/      # Modelos de diagnóstico (Módulo 1)
├── tests/           # Testes automatizados
├── docs/            # Arquitetura, decisões, hiperparâmetros, relatório técnico
├── infra/azure/     # Infraestrutura como Código (Azure - pendente)
├── notebooks/       # Notebooks de experimentação
└── data/            # raw/ e processed/
```

## Setup

> **Importante:** o `Pipfile` fica aqui em `fase_2/tech_challenge/`, mas todos os comandos Python abaixo devem ser executados a partir da **raiz do repositório** (`Pos IA/`), pois os módulos usam import absoluto (`fase_2.tech_challenge.src...`).

```bash
# 1. Instale as dependências (a partir desta pasta, onde está o Pipfile)
cd fase_2/tech_challenge
pipenv install --dev

# 2. Ative o ambiente virtual
pipenv shell

# 3. Volte para a raiz do repositório — é de lá que os comandos abaixo devem rodar
cd ../..
```

Crie um arquivo `.env` dentro de `fase_2/tech_challenge/` (não versionado) com as chaves
necessárias, por exemplo:

```
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

## Testes

```bash
python -m pytest fase_2/tech_challenge/tests/ -v
```

## Módulo 1 (baseline)

Dataset e modelos originais já incorporados.
Melhor modelo original:
Logistic Regression (acc 0.9737, recall 0.9767). Rodar o baseline:

```bash
python -m fase_2.tech_challenge.src.models.baseline
```

## Algoritmo Genético

Implementado em `src/ga/` (codificação real normalizada, seleção por torneio, cruzamento
uniforme, mutação gaussiana, elitismo). Otimiza `LogisticRegression` e `SVC Linear`.
Rodar os 3 experimentos e comparar com o baseline:

```bash
python -m fase_2.tech_challenge.src.ga.experiments
```

Resultados e análise em [docs/experiments.md](docs/experiments.md).

## Integração com LLM

Implementada em `src/llm/` (explicação de diagnóstico individual + comparativo GA vs.
baseline, ambas via API da OpenAI). Smoke test manual, ponta a ponta:

```bash
python -m fase_2.tech_challenge.smoke_test_llm
```