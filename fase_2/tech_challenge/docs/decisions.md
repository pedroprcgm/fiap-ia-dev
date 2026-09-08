# Decisões de setup (Dia 1-2)

Registradas em 02/07/2026.

## Gerenciador de dependências: Pipenv

Combina pip + virtualenv em um único fluxo (`Pipfile` / `Pipfile.lock`). Rodar
`pipenv install --dev` para instalar as dependências e `pipenv shell` para ativar o
ambiente.

## Provedor de nuvem: Azure

Usado para a etapa de escalabilidade automática e IaC (crédito extra). Infraestrutura
ficará em `infra/azure/`. Serviços candidatos a detalhar: Azure Container Apps ou App
Service (autoscale), Azure Monitor / Application Insights para logging/monitoramento.

## LLM: API hospedada (OpenAI GPT)

Escolhida pela velocidade de integração e qualidade das explicações geradas, frente ao
custo de rodar um modelo open-source localmente/na nuvem. Chave de API deve ser
configurada via variável de ambiente `OPENAI_API_KEY` (não versionada).

## Módulo 1 (modelos de diagnóstico): incorporado

Código e dataset do Módulo 1 recebidos (projeto `breast_cancer_v2`) e incorporados em
02/07/2026:
- `data/raw/data.csv` — Breast Cancer Wisconsin (Diagnostic), 569 amostras, 30 features.
- `notebooks/modulo1_cancer_classification.ipynb` — notebook original, mantido como referência.
- `src/models/data.py` — carregamento/pré-processamento (portado fielmente do notebook).
- `src/models/baseline.py` — os 5 modelos originais (SVC Linear, Linear Regression,
  Decision Tree, Logistic Regression, KNN) com os hiperparâmetros exatos do Módulo 1.
  Testado localmente e confirma os mesmos resultados do notebook.

**Principais achados** (detalhes em `docs/hyperparameters.md`):
- Melhor modelo original: **Logistic Regression** (acc 0.9737, recall 0.9767, class_weight
  da classe maligna ajustado manualmente para 5).
- GA vai focar a otimização em Logistic Regression e SVC Linear (os dois com melhor recall).
- Limitações do baseline identificadas: sem normalização de features (exceto a
  `Linear Regression`, ver correção abaixo), sem estratificação no split, e uso indevido
  de Linear Regression como classificador — a documentar no relatório técnico, fora do
  escopo direto do GA.

## Pendências do Dia 1-2 (planejamento)

- [x] Criar repositório Git + estrutura do projeto Python
- [x] Revisar os modelos de diagnóstico do Módulo 1 (código, dataset, métricas atuais)
- [x] Esboçar a arquitetura geral (diagrama inicial)
- [x] Definir os hiperparâmetros a otimizar e o espaço de busca (Logistic Regression + SVC Linear)
- [x] Escolher a LLM a usar e o provedor de nuvem

## Algoritmo Genético (Dia 3-5): implementado

Registrado em 03/07/2026. Implementação em `src/ga/` (codificação de genes real
normalizada em [0,1], seleção por torneio, cruzamento uniforme, mutação gaussiana,
elitismo). Ver `docs/hyperparameters.md` para o espaço de busca e `docs/experiments.md`
para os resultados dos 3 experimentos exigidos pelo enunciado.

**Decisão importante: normalização de features (`StandardScaler`) nos modelos
otimizados.** O baseline do Módulo 1 não normaliza as features. Ao rodar o GA sobre o
espaço de busca de `C` do SVC (até 1e2), o solver ficava numericamente instável em
dados não normalizados — em alguns casos a acurácia caía para ~0.5 (praticamente
aleatório) e o tempo de fit chegava a ~4.5s por indivíduo, inviabilizando a busca em
tempo hábil. A solução foi envolver o classificador em um `sklearn.Pipeline` com
`StandardScaler` (fit apenas no treino, sem vazamento de dados) — isso resolveu tanto a
instabilidade quanto a performance (fit caiu para ~2ms). Essa mudança de
pré-processamento vale para os modelos otimizados; o baseline original é mantido como
está (sem scaler) para preservar a comparação "modelo original do Módulo 1".

**Resultado dos experimentos:** nas 3 configurações testadas, o GA não superou a
Logistic Regression original (0.9737 acc). Análise detalhada e próximos passos para
melhorar isso em `docs/experiments.md`.

## Correção de instabilidade numérica na Linear Regression do baseline

Registrado em 14/07/2026. A `Linear Regression` do baseline (`src/models/baseline.py`)
é treinada sobre as mesmas features sem normalização dos demais modelos do Módulo 1
— só que features fortemente correlacionadas (`radius_mean`, `perimeter_mean`,
`area_mean`) deixam a matriz mal-condicionada, o que pode gerar `RuntimeWarning` de
overflow/divisão por zero durante o fit/predict, dependendo do backend BLAS/LAPACK
(observado no macOS/Accelerate; não reproduzido no Linux/OpenBLAS).

Correção: a `Linear Regression` do baseline passou a rodar dentro de um
`Pipeline([StandardScaler, LinearRegression])`. Como regressão linear sem
regularização (OLS) é invariante a reparametrização linear das features, isso resolve
o mal-condicionamento sem alterar a predição — confirmado empiricamente (diferença
~1e-14 nas previsões contínuas, mesmas 4 métricas reportadas). Os outros 4 modelos do
baseline continuam sem normalização, fiéis ao Módulo 1.
