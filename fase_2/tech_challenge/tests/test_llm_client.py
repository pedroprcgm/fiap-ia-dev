"""Testes de src/llm/client.py — tratamento de erro e log de latência das
chamadas à LLM. Nunca chamam a API real (o client é sempre mockado)."""
import logging

import openai
import pytest

from fase_2.tech_challenge.src.llm.client import LLMRequestError, chat_completion


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._result)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, result=None, error=None):
        self.chat = _FakeChat(_FakeCompletions(result=result, error=error))


def test_chat_completion_returns_stripped_content_on_success(monkeypatch):
    fake_client = _FakeClient(result="  Explicação gerada.  ")
    monkeypatch.setattr(
        "fase_2.tech_challenge.src.llm.client.get_client", lambda: fake_client
    )

    result = chat_completion([{"role": "user", "content": "oi"}])

    assert result == "Explicação gerada."


def test_chat_completion_logs_latency_on_success(monkeypatch, caplog):
    fake_client = _FakeClient(result="ok")
    monkeypatch.setattr(
        "fase_2.tech_challenge.src.llm.client.get_client", lambda: fake_client
    )

    with caplog.at_level(logging.INFO, logger="fase_2.tech_challenge.src.llm.client"):
        chat_completion([{"role": "user", "content": "oi"}])

    assert any("latencia" in record.message for record in caplog.records)


def test_chat_completion_wraps_openai_error_as_llm_request_error(monkeypatch):
    original_error = openai.APITimeoutError(request=None)
    fake_client = _FakeClient(error=original_error)
    monkeypatch.setattr(
        "fase_2.tech_challenge.src.llm.client.get_client", lambda: fake_client
    )

    with pytest.raises(LLMRequestError) as exc_info:
        chat_completion([{"role": "user", "content": "oi"}])

    assert exc_info.value.__cause__ is original_error


def test_chat_completion_logs_error_on_failure(monkeypatch, caplog):
    fake_client = _FakeClient(error=openai.OpenAIError("falha simulada"))
    monkeypatch.setattr(
        "fase_2.tech_challenge.src.llm.client.get_client", lambda: fake_client
    )

    with caplog.at_level(logging.ERROR, logger="fase_2.tech_challenge.src.llm.client"):
        with pytest.raises(LLMRequestError):
            chat_completion([{"role": "user", "content": "oi"}])

    assert any("latencia" in record.message and "falha simulada" in record.message for record in caplog.records)
