from app.config import get_settings


def test_health_does_not_touch_external_services(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_local_url_points_to_the_host_only_inside_docker(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:1235/v1")
    assert get_settings().local_url == "http://localhost:1235/v1"
    monkeypatch.setenv("RUNNING_IN_DOCKER", "1")
    assert get_settings().local_url == "http://host.docker.internal:1235/v1"
    monkeypatch.setenv("LOCAL_LLM_URL", "http://otro-servidor:9000/v1")      # un host remoto no se toca
    assert get_settings().local_url == "http://otro-servidor:9000/v1"


def test_each_provider_gets_its_own_model(monkeypatch):
    """Con LLM_MODEL=gpt-6-luna en el .env, cambiar a `local` no debe pedirle a LM Studio un modelo de OpenAI."""
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama-3.2-3b-instruct")
    for provider, expected in (("openai", "gpt-6-luna"), ("local", "llama-3.2-3b-instruct")):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert get_settings().llm_model == expected
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert get_settings().llm_model == "claude-haiku-5-5"          # sin LLM_MODEL: el modelo económico por defecto
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert get_settings().llm_model == "gpt-6-luna"
    monkeypatch.delenv("LOCAL_LLM_MODEL")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "otro-modelo-local")
    assert get_settings().llm_model == "otro-modelo-local"          # sin LOCAL_LLM_MODEL, LLM_MODEL sirve de respaldo
