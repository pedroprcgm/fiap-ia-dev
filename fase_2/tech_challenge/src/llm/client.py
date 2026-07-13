"""Cliente fino para a API da OpenAI.

A chave de API é lida exclusivamente da variável de ambiente OPENAI_API_KEY,
carregada a partir de um arquivo `.env` na raiz do projeto (nunca commitado — ver
`.gitignore`). Nunca hardcode a chave em código-fonte.
"""
from functools import lru_cache
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Retorna um client OpenAI autenticado, criado uma única vez (cache)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha com sua chave (nunca commite o .env)."
        )
    return OpenAI(api_key=api_key)


def chat_completion(
    messages: list,
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> str:
    """Chamada simples de chat completion. Retorna apenas o texto da resposta.

    `temperature` baixa (0.3) é intencional: para explicações clínicas queremos
    respostas consistentes e pouco "criativas", não geração livre.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
