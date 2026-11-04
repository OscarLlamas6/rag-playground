"""EXPERIMENTAL · fotogramas descritos por un modelo con visión: la FORMA de las peticiones, con un transporte HTTP simulado."""

import base64
import json

import anthropic
import httpx
import openai
import pytest

from app import frames, media
from app.config import get_settings
from app.extras import MediaError

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
B64 = base64.b64encode(JPEG).decode()


@pytest.fixture
def api(monkeypatch):
    class Api:
        seen: list = []
        reply = None

    state = Api()
    state.seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        state.seen.append((request.url.path, json.loads(request.content)))
        return state.reply(request)

    real_openai, real_anthropic = openai.OpenAI, anthropic.Anthropic

    def openai_factory(*args, **kwargs):
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
        return real_openai(*args, **kwargs)

    def anthropic_factory(*args, **kwargs):
        # el SDK de Anthropic usa `httpx2` (un fork de httpx): el transporte simulado y la respuesta deben ser de esa librería
        import httpx2

        def handler2(request):
            state.seen.append((request.url.path, json.loads(request.content)))
            reply = state.reply(request)
            return httpx2.Response(reply.status_code, json=json.loads(reply.content))

        kwargs["http_client"] = httpx2.Client(transport=httpx2.MockTransport(handler2))
        return real_anthropic(*args, **kwargs)

    monkeypatch.setattr(openai, "OpenAI", openai_factory)
    monkeypatch.setattr(anthropic, "Anthropic", anthropic_factory)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-prueba")
    return state


def test_local_provider_sends_an_image_url_data_url(api, monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "local")
    monkeypatch.setenv("VISION_MODEL", "google/gemma-4-12b-qat")
    api.reply = lambda r: httpx.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 0, "model": "google/gemma-4-12b-qat",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Una pizarra con notas."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    assert frames.describe_frame(JPEG, get_settings()) == "Una pizarra con notas."
    path, body = api.seen[0]
    assert path.endswith("/chat/completions") and body["model"] == "google/gemma-4-12b-qat"
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "text" and parts[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + B64}}


def test_openai_provider_uses_the_responses_api_with_input_image(api, monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    api.reply = lambda r: httpx.Response(200, json={
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": "gpt-6-luna",
        "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Diapositiva con un gráfico.", "annotations": []}]}],
        "usage": {"input_tokens": 100, "output_tokens": 8, "total_tokens": 108,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    assert frames.describe_frame(JPEG, get_settings()) == "Diapositiva con un gráfico."
    path, body = api.seen[0]
    assert path.endswith("/responses") and body["model"] == "gpt-6-luna"
    parts = body["input"][0]["content"]
    assert parts[0]["type"] == "input_text"
    assert parts[1] == {"type": "input_image", "image_url": "data:image/jpeg;base64," + B64}


def test_anthropic_provider_sends_a_base64_image_block(api, monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    api.reply = lambda r: httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Un cristal con notas manuscritas."}],
        "usage": {"input_tokens": 50, "output_tokens": 9}})
    assert frames.describe_frame(JPEG, get_settings()) == "Un cristal con notas manuscritas."
    path, body = api.seen[0]
    assert path.endswith("/v1/messages") and body["model"] == "claude-haiku-5-5"
    block = body["messages"][0]["content"][0]
    assert block == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": B64}}
    assert body["messages"][0]["content"][1]["type"] == "text"


def test_provider_failure_becomes_a_clear_error(api, monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    api.reply = lambda r: httpx.Response(401, json={"error": {"message": "bad key", "type": "invalid_request_error"}})
    with pytest.raises(MediaError, match="modelo de visión 'openai'"):
        frames.describe_frame(JPEG, get_settings())
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MediaError, match="OPENAI_API_KEY"):
        frames.describe_frame(JPEG, get_settings())


def test_frame_times_skip_second_zero_and_stay_inside_the_video():
    assert frames.frame_times(100, 30) == [15.0, 45.0, 75.0]
    assert frames.frame_times(0, 30) == [] and frames.frame_times(100, 0) == []


def test_captions_become_sections_aligned_in_time():
    sections, metas = frames.captions_to_sections([frames.Caption(45.0, "Notas sobre agentes.")], "v.mp4", 30)
    assert sections[0][0] == "00:30–01:00 · fotograma" and "00:45" in sections[0][1]
    assert metas[0] == {"media": "v.mp4", "start_s": 30.0, "end_s": 60.0, "content": "frame"}


def test_disabled_by_default_and_experimental_path_when_enabled(tmp_path, whisper, monkeypatch):
    video = tmp_path / "media" / "clase.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video de mentira")
    calls = []
    monkeypatch.setattr(frames, "_extract_frame", lambda path, t: JPEG)
    monkeypatch.setattr(frames, "describe_frame", lambda jpeg, s: calls.append(1) or "Una diapositiva con el título Agentes.")
    assert not any("fotogramas" in w for w in media.read_media(video).warnings) and calls == []      # FRAME_CAPTIONS=false

    monkeypatch.setenv("FRAME_CAPTIONS", "true")
    monkeypatch.setenv("FRAME_INTERVAL_SECONDS", "60")
    doc = media.read_media(video)                                    # 120 s de video falso → 2 fotogramas (a los 30 s y a los 90 s)
    assert len(calls) == 2 and any("EXPERIMENTAL" in w for w in doc.warnings)
    frames_meta = [m for m in doc.section_meta if m.get("content") == "frame"]
    assert [m["start_s"] for m in frames_meta] == [0.0, 60.0]
    media.read_media(video)
    assert len(calls) == 2                                           # la segunda vez, de la caché
