# syntax=docker/dockerfile:1
# Backend de rag-simple: FastAPI + el RAG. Imagen pequeña, dependencias fijadas por uv.lock, usuario sin privilegios.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /app

# 1) Dependencias (esta capa se reutiliza mientras no cambien pyproject.toml ni uv.lock)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# 2) Código, prompts y datos (en docker-compose, data/ y prompts/ se montan encima para editarlos sin reconstruir)
COPY app ./app
COPY prompts ./prompts
COPY data ./data

RUN useradd --system --uid 10001 --home /app app && mkdir -p /app/.cache && chown -R app /app/.cache
USER app

# RUNNING_IN_DOCKER hace que LOCAL_LLM_URL=http://localhost:... apunte a tu máquina (host.docker.internal), donde corre LM Studio
ENV PATH="/app/.venv/bin:$PATH" RUNNING_IN_DOCKER=1 PORT=8000 HOST=0.0.0.0
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=2)"
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
