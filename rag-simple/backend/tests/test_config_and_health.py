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
