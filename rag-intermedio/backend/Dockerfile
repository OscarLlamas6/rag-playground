# syntax=docker/dockerfile:1
# Backend de rag-intermedio: FastAPI + el RAG. Imagen pequeña, dependencias fijadas por uv.lock, usuario sin privilegios.
FROM python:3.12-slim

# INSTALL_MEDIA=true añade ffmpeg y el grupo `media` (Docling, faster-whisper, av<16 y torch solo CPU): PDF con OCR, audio y video.
# Pesa varios gigas y la primera ejecución descarga modelos. Con `false` (por defecto) la imagen es la base, pequeña.
#   docker compose -f docker-compose.yml -f docker-compose.media.yml up --build -d     (o: task docker:up:media)
ARG INSTALL_MEDIA=false

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /app

# 1) Dependencias (esta capa se reutiliza mientras no cambien pyproject.toml ni uv.lock)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_MEDIA" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/* \
      && uv sync --frozen --no-dev --group media --no-install-project; \
    else \
      uv sync --frozen --no-dev --no-install-project; \
    fi

# 2) Código, prompts y datos (en docker-compose, data/ y prompts/ se montan encima para editarlos sin reconstruir)
COPY app ./app
COPY prompts ./prompts
COPY data ./data

# /app/.cache/media: transcripciones, conversiones de Docling y modelos descargados (HF_HOME). El usuario `app` no es root:
# todo lo que se escribe va en rutas que le pertenecen (con un volumen con nombre nuevo, Docker hereda este dueño).
RUN useradd --system --uid 10001 --home /app app && mkdir -p /app/.cache/media/hf && chown -R app /app/.cache
USER app

# RUNNING_IN_DOCKER hace que LOCAL_LLM_URL=http://localhost:... apunte a tu máquina (host.docker.internal), donde corre LM Studio
ENV PATH="/app/.venv/bin:$PATH" RUNNING_IN_DOCKER=1 PORT=8010 HOST=0.0.0.0 \
    MEDIA_CACHE_DIR=/app/.cache/media HF_HOME=/app/.cache/media/hf
EXPOSE 8010
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=2)"
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
