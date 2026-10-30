# rag-playground

Diferentes implementaciones de **RAG** (Retrieval-Augmented Generation), cada una en su carpeta, como ejemplos
"formales pero no complejos": un backend en Python (FastAPI), una interfaz web en Go y evaluación. Sirven para estudiar
el flujo y para que cada pieza se vea funcionando.

Todos los ejemplos son **one-shot**: una pregunta, una respuesta con citas. No hay historial ni agentes.

| Carpeta | Qué muestra | Almacén | Búsqueda | Documentos |
| --- | --- | --- | --- | --- |
| [`rag-simple`](rag-simple) | El flujo completo y mínimo: fragmentar → vectorizar → buscar → generar con citas verificadas → evaluar | **Neon** (PostgreSQL + pgvector) | semántica | 3 Markdown |
| [`rag-intermedio`](rag-intermedio) | Lo que se añade con documentos reales: multiformato, padre-hijo, BM25 + vectores (RRF), filtros, reranking | **Pinecone** | vectorial, BM25 o híbrida | Markdown, HTML y Word |

La idea es mostrar **alternativas**: dos almacenes distintos (SQL con pgvector contra una base vectorial administrada) detrás de la
misma interfaz de aplicación.

## Cómo está organizado cada ejemplo

```
rag-xxx/
  backend/   FastAPI + el RAG (app/), prompts versionados (prompts/), datos (data/) y pruebas (tests/)
  web/       servidor Go: sirve la interfaz (static/) y reenvía /api al backend
  Taskfile.yml   task install · backend · web · test · demo · docker:up · docker:demo · docker:down
  README.md  el flujo, la arquitectura, cómo correrlo y experimentos para probar
```

## Puesta en marcha

Requisitos: [uv](https://docs.astral.sh/uv/), Go 1.22+ y [Task](https://taskfile.dev/installation/) (`task`, el ejecutor de tareas de cada proyecto; sin argumentos lista las tareas).

```bash
cp .env.example .env        # y completa lo que uses; .env nunca se sube a Git
cd rag-simple               # o rag-intermedio
task install && task backend      # terminal 1
task web                          # terminal 2
```

| Proyecto | Backend | Interfaz |
| --- | --- | --- |
| rag-simple | http://localhost:8000 (`/docs`) | http://localhost:8080 |
| rag-intermedio | http://localhost:8010 (`/docs`) | http://localhost:8081 |

**Con Docker:** cada proyecto trae su `docker-compose.yml`:

```bash
cd rag-simple                 # o rag-intermedio
docker compose up --build -d                                                   # completo, con tu .env
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build -d  # demo, sin cuentas
```

`task demo` en cualquiera de los dos arranca **sin cuentas ni servidores** (embeddings y modelo "fake", todo en memoria).
Sirve para ver la interfaz y el flujo; la calidad real requiere embeddings reales (LM Studio con
`nomic-embed-text-v2-moe`, gratis, o la API de OpenAI).

## Qué se necesita según cómo lo corras

| | Embeddings | Modelo que responde | Almacén |
| --- | --- | --- | --- |
| Completo | LM Studio (local, gratis) u OpenAI | Claude, OpenAI o local | Neon (simple) · Pinecone (intermedio) |
| Demo | `fake` | `fake` | memoria |

Costos: con embeddings locales y los planes gratuitos de Neon y Pinecone, solo cuesta el modelo que responde: del orden de
US$ 0.0001 a 0.0004 por pregunta con Claude Haiku (el intermedio con reranking suma una llamada). Con un modelo local, cero.

## Usar OpenAI para las respuestas y los embeddings

En el `.env` (las demás variables no cambian):

```bash
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-6-luna                          # modelo que redacta las respuestas (opcional: es el valor por defecto)
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small      # ojo: el de LM Studio (nomic) no existe en OpenAI
```

- **Después de cambiar los embeddings hay que volver a ingestar:** los vectores de dos modelos no se pueden mezclar. Con OpenAI el
  vector tiene **1536 dimensiones** (nomic: 768). En `rag-simple` la ingesta recrea la tabla de Neon con la dimensión nueva. En
  `rag-intermedio` la dimensión de un índice de Pinecone no se puede cambiar: si ya tienes uno de 768, usa otro nombre
  (`PINECONE_INDEX`) o fija `EMBEDDINGS_DIMENSIONS=768` para que OpenAI acorte el vector (`text-embedding-3` lo permite).
- **Con Docker:** el contenedor lee el `.env` al arrancar; tras editarlo ejecuta `docker compose up -d --force-recreate`.
  Con OpenAI no necesitas LM Studio, así que sirve el compose normal.
- **Costo:** `text-embedding-3-small` cuesta ≈ US$ 0.02 por millón de tokens (ingestar estos documentos son fracciones de centavo) y
  cada pregunta, del orden de US$ 0.0001 a 0.0004 (el intermedio con reranking suma una llamada). La interfaz muestra los tokens y el costo.
- **Dos modelos distintos:** `LLM_MODEL` es el que redacta las respuestas (con OpenAI o Anthropic; con `LLM_PROVIDER=local` manda `LOCAL_LLM_MODEL`) y `EMBEDDINGS_MODEL`, el que vectoriza. Se configuran por separado.
  Si el modelo por defecto no existe en tu cuenta, cámbialo con `LLM_MODEL`.

## Agregar otra implementación

Una carpeta nueva `rag-<nombre>/` con la misma estructura. Cada proyecto es autocontenido (no comparte código con los demás) a
propósito: así se puede leer, copiar y modificar uno sin entender los otros.
