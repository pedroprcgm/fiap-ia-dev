# Hiperparâmetros e espaço de busca

Baseado no código real do Módulo 1 (`notebooks/modulo1_cancer_classification.ipynb`,
portado para `src/models/baseline.py`). Dataset: Breast Cancer Wisconsin (Diagnostic),
569 amostras, 30 features numéricas, target binário (`diagnosis`: 1 = maligno, 0 = benigno).

## Modelos originais e resultados (baseline, Módulo 1)

| Modelo | Acurácia | Recall | Precisão | F1 |
|---|---|---|---|---|
| SVC Linear (C=2, class_weight={0:1,1:5}) | 0.9386 | 0.9767 | 0.8750 | 0.9231 |
| Linear Regression (threshold 0.5) | 0.9561 | 0.9070 | 0.9750 | 0.9398 |
| Decision Tree (max_depth=5) | 0.9474 | 0.9302 | 0.9302 | 0.9302 |
| **Logistic Regression (C=1, class_weight={0:1,1:5})** | **0.9737** | **0.9767** | **0.9545** | **0.9655** |
| KNN (n_neighbors=5) | 0.9561 | 0.8837 | 1.0000 | 0.9383 |

O Módulo 1 concluiu que **Logistic Regression** é o melhor modelo (melhor equilíbrio
entre acurácia e recall da classe maligna). O `class_weight` da classe positiva (1..5)
foi ajustado manualmente por tentativa e erro — candidato natural para otimização via GA.

**Limitações conhecidas do baseline** (não fazem parte do escopo do GA, mas relevantes
para o relatório técnico):
- Nenhuma normalização/padronização das features (escalas muito diferentes, ex.:
  `area_mean` ~centenas vs. `smoothness_mean` ~0.1) nos 4 modelos que replicam o
  Módulo 1 tal como estava (`SVC Linear`, `Decision Tree`, `Logistic Regression`, `KNN`).
  A `Linear Regression` é a exceção: roda dentro de um `Pipeline` com `StandardScaler`
  (ver `docs/decisions.md`) só para evitar instabilidade numérica, sem alterar nenhuma
  métrica reportada.
- Split treino/teste sem estratificação (`train_test_split` sem `stratify=y`).
- `LinearRegression` é usada como classificador via threshold — não é um classificador de
  verdade e não deve ser um candidato à otimização do GA.

## Modelos-alvo da otimização via GA

Dado o resultado do Módulo 1, o GA vai focar em:
1. **Logistic Regression** (modelo vencedor — prioridade)
2. **SVC Linear** (segundo melhor recall — comparação secundária)

`Decision Tree`, `KNN` e `Linear Regression` seguem como baseline de comparação no
relatório, mas não são alvo primário de otimização (menor potencial de ganho ou, no caso
da Linear Regression, uso metodologicamente inadequado).

## Espaço de busca — Logistic Regression

| Hiperparâmetro | Tipo | Intervalo / opções | Observação |
|---|---|---|---|
| `C` (inverso da regularização) | contínuo (log) | [1e-3, 1e2] | baseline usa default (1.0) |
| `penalty` | categórico | {"l1", "l2"} | `l1` requer `solver="liblinear"` ou `"saga"` |
| `solver` | categórico | {"liblinear", "lbfgs", "saga"} | compatibilidade com `penalty` |
| `class_weight` (peso da classe maligna) | discreto | [1, 10] | baseline testou manualmente 1–5, venceu 5 |
| `max_iter` | fixo | 10000 | mantido fixo (evita erro de convergência) |

## Espaço de busca — SVC Linear

| Hiperparâmetro | Tipo | Intervalo / opções | Observação |
|---|---|---|---|
| `C` | contínuo (log) | [1e-2, 1e2] | baseline usa C=2 |
| `class_weight` (peso da classe maligna) | discreto | [1, 10] | baseline usa 5 |
| `kernel` | fixo | "linear" | RBF já testado no Módulo 1 e descartado (piora recall) |

## Codificação para o GA

- Genes contínuos (`C`) → codificação real (float), com mutação gaussiana/log-uniforme.
- Genes categóricos (`penalty`, `solver`) → codificação inteira mapeada para a lista de opções.
- Genes discretos (`class_weight`) → inteiros dentro do intervalo.

## Função fitness

Combinação ponderada de accuracy, recall e F1, com peso maior em **recall da classe
maligna** — no contexto de diagnóstico, um falso negativo (dizer que um tumor maligno é
benigno) é o erro mais custoso. Pesos exatos a definir/ajustar durante os experimentos
(Dia 3-7).

## Implementação (atualizado após rodar os experimentos)

- `solver` foi fixado em `"liblinear"` (compatível com `l1` e `l2`, adequado ao dataset
  pequeno); apenas `penalty` é otimizado pelo GA.
- Pesos da fitness definidos como `accuracy=0.3, recall=0.5, f1=0.2`
  (`src/ga/fitness.py::METRIC_WEIGHTS`).
- **Achado importante:** os modelos otimizados usam um `StandardScaler` antes do
  classificador (dentro de um `sklearn.Pipeline`), diferente do baseline. Sem
  normalização, o solver do SVC ficava numericamente instável para C alto (acurácia
  caía para ~0.5 e o tempo de fit explodia). Ver `docs/decisions.md`.
- Resultados completos dos 3 experimentos em `docs/experiments.md` — o GA, nas
  configurações testadas, não superou a Logistic Regression original; ver análise e
  próximos passos lá.
