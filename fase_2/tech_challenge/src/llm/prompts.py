"""Templates de prompt para interpretação dos resultados dos modelos de diagnóstico.

Dois casos de uso, ambos exigidos pelo Requisito 3 do Tech Challenge (Projeto 1):

1. Explicar um diagnóstico individual — a predição de um modelo para uma amostra
   específica, traduzida em insight acionável para um médico.
2. Explicar o comparativo entre o modelo otimizado via Algoritmo Genético e o
   modelo original (baseline do Módulo 1) — para o relatório técnico e o vídeo
   de demonstração.

O modelo recebe SEMPRE os números já prontos (grounding), nunca é solicitado a
"adivinhar" um diagnóstico, e as regras mais importantes ficam no fim do prompt
de sistema.
"""
from typing import Dict, List

SYSTEM_PROMPT_DIAGNOSIS = """Você é um assistente que traduz a saída de um modelo de \
machine learning para diagnóstico de câncer de mama em uma explicação clara para \
uma médica ou médico.

Regras obrigatórias:
- Baseie-se SOMENTE nos dados numéricos fornecidos na mensagem do usuário. Nunca \
invente valores, exames ou informações que não foram passados.
- Nunca afirme um diagnóstico com certeza absoluta — o modelo estima uma \
probabilidade, não emite um laudo.
- Use linguagem clínica direta, sem jargão de machine learning (não mencione \
"class_weight", "hiperparâmetro", "fitness", "algoritmo genético" etc.).
- Estruture a resposta em 3 partes curtas: (1) resultado do modelo, (2) principais \
fatores que influenciaram a previsão, (3) recomendação prática (ex.: priorizar \
revisão humana em casos de confiança baixa ou fatores conflitantes).
- Termine sempre reforçando que a decisão final é do profissional de saúde."""

SYSTEM_PROMPT_COMPARISON = """Você é um assistente que explica, para uma médica, \
médico ou gestor hospitalar sem formação técnica em machine learning, por que um \
modelo de diagnóstico otimizado é preferível (ou não) ao modelo original.

Regras obrigatórias:
- Baseie-se SOMENTE nas métricas numéricas fornecidas na mensagem do usuário.
- Explique o que cada mudança de métrica significa na prática (ex.: "o modelo \
otimizado erra menos ao classificar casos malignos como benignos", em vez de \
"o recall aumentou").
- Seja honesto: se o modelo otimizado for pior em alguma métrica relevante, diga \
isso claramente, sem minimizar.
- Evite detalhe técnico de otimização (no máximo diga que os parâmetros do modelo \
foram ajustados automaticamente por um processo de busca)."""


def build_diagnosis_prompt(
    model_name: str,
    prediction_label: str,
    probability: float,
    top_features: List[Dict],
) -> List[Dict[str, str]]:
    """Monta as mensagens (system + user) para explicar um diagnóstico individual.

    `top_features`: lista de dicts no formato
        {"name": str, "value": float, "influence": str}
    já calculados a partir do modelo real (ver `src/llm/explain.py`) — o prompt
    nunca pede para a LLM inferir isso sozinha.
    """
    features_txt = "\n".join(
        f"- {f['name']} = {f['value']:.3g} ({f['influence']})" for f in top_features
    )
    user_prompt = f"""Modelo utilizado: {model_name}
Predição: {prediction_label}
Confiança do modelo: {probability:.1%}

Principais fatores que mais pesaram nesta predição:
{features_txt}

Gere a explicação para a médica ou médico seguindo as regras do seu papel."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIAGNOSIS},
        {"role": "user", "content": user_prompt},
    ]


def build_comparison_prompt(
    model_family: str,
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    experiment_config: Dict,
) -> List[Dict[str, str]]:
    """Monta as mensagens (system + user) para explicar o comparativo GA vs. baseline."""
    user_prompt = f"""Modelo: {model_family}

Métricas do modelo original (Módulo 1):
- Acurácia: {baseline_metrics['accuracy']:.1%}
- Recall (sensibilidade a casos malignos): {baseline_metrics['recall']:.1%}
- Precisão: {baseline_metrics['precision']:.1%}
- F1-score: {baseline_metrics['f1']:.1%}

Métricas do modelo otimizado (busca automática de parâmetros; configuração: \
população={experiment_config.get('population_size')}, \
gerações={experiment_config.get('generations')}):
- Acurácia: {optimized_metrics['accuracy']:.1%}
- Recall: {optimized_metrics['recall']:.1%}
- Precisão: {optimized_metrics['precision']:.1%}
- F1-score: {optimized_metrics['f1']:.1%}

Explique o comparativo seguindo as regras do seu papel."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COMPARISON},
        {"role": "user", "content": user_prompt},
    ]
