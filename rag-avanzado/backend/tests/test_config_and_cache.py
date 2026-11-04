import pytest

from app import cache, llm
from app.config import get_settings


def test_defaults_and_invalid_values(monkeypatch):
    monkeypatch.delenv("AVANZADO_STORE", raising=False)
    s = get_settings()
    assert s.store == "postgres" and s.database_url.endswith(":5434/rag")      # el puerto de depuración del compose
    monkeypatch.setenv("AVANZADO_STORE", "neon")
    with pytest.raises(ValueError, match="AVANZADO_STORE"):
        get_settings()


def test_the_shared_STORE_variable_of_rag_simple_is_ignored(monkeypatch):
    monkeypatch.setenv("STORE", "neon")
    assert get_settings().store == "memory"      # lo fija el fixture con AVANZADO_STORE; STORE=neon (de rag-simple) no estorba


def test_local_url_is_translated_inside_docker(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:1235/v1  # LM Studio")
    assert get_settings().local_url == "http://localhost:1235/v1"
    monkeypatch.setenv("RUNNING_IN_DOCKER", "1")
    assert get_settings().local_url == "http://host.docker.internal:1235/v1"


def test_llm_model_selection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama-3.2-3b-instruct")
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    assert get_settings().llm_model == "llama-3.2-3b-instruct"


def test_the_cache_reuses_costly_calls_and_force_refreshes():
    first, hit = cache.cached_generate("contexto", "Sección del fragmento: Red troncal\ntexto")
    again, hit2 = cache.cached_generate("contexto", "Sección del fragmento: Red troncal\ntexto")
    assert (hit, hit2) == (False, True) and again.text == first.text and again.cost_usd == 0 and cache.cache_size("contexto") == 1
    assert cache.cached_generate("contexto", "Sección del fragmento: Red troncal\ntexto", force=True)[1] is False
    assert cache.cached_generate("contexto", "Sección del fragmento: Otra\ntexto")[1] is False        # otro prompt, otra clave


def test_generate_rejects_unknown_purposes_and_costs_are_computed():
    with pytest.raises(ValueError):
        llm.generate("x", purpose="magia")
    assert llm._cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert llm._cost("llama-3.2-3b-instruct", 10**6, 10**6) == 0.0
