"""OpenAI sin red ni llave: un transporte HTTP simulado comprueba QUÉ se envía y CÓMO se lee la respuesta."""

import json

import httpx
import openai
import pytest

from app.embeddings import EmbeddingError, embed
from app.llm import LLMError, generate


@pytest.fixture
def api(monkeypatch):
    """Sustituye el cliente de OpenAI por uno cuyo transporte responde lo que diga `api.reply`. Guarda las peticiones en `api.seen`."""
    class Api:
        seen: list = []
        reply = None

    state = Api()
    state.seen = []
    real = openai.OpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        state.seen.append((request.url.path, json.loads(request.content)))
        return state.reply(request)

    def factory(*args, **kwargs):
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
        return real(*args, **kwargs)

    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    return state


def test_openai_embeddings_request_and_ordering(api, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "768")
    api.reply = lambda r: httpx.Response(200, json={
        "object": "list", "model": "text-embedding-3-small", "usage": {"prompt_tokens": 4, "total_tokens": 4},
        "data": [{"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                 {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}]})       # llegan desordenados
    vectors = embed(["uno", "dos"], kind="document")
    path, body = api.seen[0]
    assert path.endswith("/embeddings")
    assert body["model"] == "text-embedding-3-small" and body["dimensions"] == 768
    assert body["input"] == ["uno", "dos"]                       # OpenAI no usa prefijos de documento/consulta
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]                   # se reordenan por índice


def test_openai_embeddings_wrong_model_gives_an_actionable_error(api, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v2-moe")     # el de LM Studio
    api.reply = lambda r: httpx.Response(404, json={"error": {"message": "The model does not exist", "type": "invalid_request_error", "code": "model_not_found"}})
    with pytest.raises(EmbeddingError) as error:
        embed(["hola"])
    assert "EMBEDDINGS_MODEL" in str(error.value) and "text-embedding-3-small" in str(error.value)


def test_openai_embeddings_without_key(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        embed(["hola"])


def test_openai_llm_uses_the_responses_api_and_prices_the_call(api, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    api.reply = lambda r: httpx.Response(200, json={
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": "gpt-6-luna",
        "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Cuesta 39 dólares [1].", "annotations": []}]}],
        "usage": {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    result = generate("¿Cuánto cuesta?", max_tokens=300)
    path, body = api.seen[0]
    assert path.endswith("/responses")
    assert body["model"] == "gpt-6-luna" and body["input"] == "¿Cuánto cuesta?"
    assert body["max_output_tokens"] == 300 and body["reasoning"] == {"effort": "none"}   # sin razonamiento: tarea simple
    assert result.text == "Cuesta 39 dólares [1]." and (result.tokens_in, result.tokens_out) == (1000, 20)
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 20 * 0.50) / 1_000_000)


def test_openai_llm_errors_are_wrapped(api, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    api.reply = lambda r: httpx.Response(401, json={"error": {"message": "Incorrect API key", "type": "invalid_request_error"}})
    with pytest.raises(LLMError, match="openai"):
        generate("hola")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        generate("hola")
