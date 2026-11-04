# rag-playground

Diferentes implementaciones de **RAG** (Retrieval-Augmented Generation), cada una en su carpeta, como ejemplos
"formales pero no complejos": un backend en Python (FastAPI), una interfaz web en Go y evaluación. Sirven para estudiar
el flujo y para que cada pieza se vea funcionando.

Todos los ejemplos son **one-shot**: una pregunta, una respuesta con citas. No hay historial ni agentes (`rag-avanzado` encadena varios pasos, pero siempre en un flujo fijo que decide el código, no el modelo).

| Carpeta | Qué muestra | Almacén | Búsqueda | Documentos |
| --- | --- | --- | --- | --- |
| [`rag-simple`](rag-simple) | El flujo completo y mínimo: fragmentar → vectorizar → buscar → generar con citas verificadas → evaluar | **Neon** (PostgreSQL + pgvector) | semántica | 3 Markdown |
| [`rag-intermedio`](rag-intermedio) | Lo que se añade con documentos reales: multiformato, padre-hijo, BM25 + vectores (RRF), filtros, reranking; y, con el grupo opcional `media`, **PDF con Docling (y OCR)**, **audio y video** con citas `mm:ss` | **Pinecone** | vectorial, BM25 o híbrida | Markdown, HTML, Word, PDF, audio y video |
| [`rag-avanzado`](rag-avanzado) | **GraphRAG**: un grafo de conocimiento (en PostgreSQL con CTE recursivas), comunidades, descomposición de preguntas y recuperación contextual, para las preguntas que hay que *construir* siguiendo relaciones | **PostgreSQL + pgvector en Docker** | vectorial, grafo local, grafo global o híbrido | Markdown (catálogo de servicios de TI) |

La idea es mostrar **alternativas**: tres almacenes distintos (Neon en la nube, Pinecone administrado y PostgreSQL propio en Docker) detrás de
la misma forma de aplicación. Empieza por `rag-simple`; cada proyecto añade lo que el anterior deja sin resolver. El RAG agéntico se verá en sesiones posteriores.

## Cómo está organizado cada ejemplo

```
rag-xxx/
  backend/   FastAPI + el RAG (app/), prompts versionados (prompts/), datos (data/) y pruebas (tests/)
  web/       servidor Go: sirve la interfaz (static/) y reenvía /api al backend
  Taskfile.yml   task install · backend · web · test · demo · docker:up · docker:demo · docker:up:host · docker:down · docker:logs
                 (más tareas propias: rag-intermedio → install:media, docker:up:media · rag-avanzado → db:up, test:integration)
  README.md  el flujo, la arquitectura, cómo correrlo y experimentos para probar
```

## Puesta en marcha

Requisitos: [uv](https://docs.astral.sh/uv/), Go 1.22+ y [Task](https://taskfile.dev/installation/) (`task`, el ejecutor de tareas de cada proyecto; sin argumentos lista las tareas).

```bash
cp .env.example .env        # y completa lo que uses; .env nunca se sube a Git
cd rag-simple               # o rag-intermedio · rag-avanzado
task install && task backend      # terminal 1   (rag-avanzado necesita antes: task db:up)
task web                          # terminal 2
```

| Proyecto | Backend | Interfaz | Otros |
| --- | --- | --- | --- |
| rag-simple | http://localhost:8000 (`/docs`) | http://localhost:8080 | |
| rag-intermedio | http://localhost:8010 (`/docs`) | http://localhost:8081 | |
| rag-avanzado | http://localhost:8020 (`/docs`) | http://localhost:8082 | PostgreSQL de depuración en `127.0.0.1:5434` |

**Con Docker:** cada proyecto trae su `docker-compose.yml`:

```bash
cd rag-simple                 # o rag-intermedio · rag-avanzado
docker compose up --build -d                                                   # completo, con tu .env
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build -d  # demo, sin cuentas
```

(`task docker:up` y `task docker:demo` hacen lo mismo.) Si usas LM Studio, lee la sección "LM Studio y Docker" más abajo.

`task demo` en cualquiera de los tres arranca **sin cuentas ni servidores** (embeddings y modelo "fake", todo en memoria).
Sirve para ver la interfaz y el flujo; la calidad real requiere embeddings reales (LM Studio con
`nomic-embed-text-v2-moe`, gratis, o la API de OpenAI).

## Qué se necesita según cómo lo corras

| | Embeddings | Modelo que responde | Almacén |
| --- | --- | --- | --- |
| Completo | LM Studio (local, gratis) u OpenAI | Claude, OpenAI o local | Neon (simple) · Pinecone (intermedio) · PostgreSQL en Docker (avanzado) |
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

## PDF, OCR, audio y video (rag-intermedio)

`rag-intermedio` lee PDF con **Docling** (con OCR automático cuando una página no trae texto) y transcribe audio y video con **faster-whisper**.
Eso necesita un grupo de dependencias opcional (varios cientos de MB y modelos que se descargan la primera vez: hazlo **antes** de necesitarlo):

```bash
cd rag-intermedio
task install:media                 # docling, faster-whisper y torch solo CPU
# pon tus audios y videos en  rag-intermedio/backend/data/media/   (esa carpeta NO va a git)
task backend                       # y task web, como siempre; en la pestaña Ingesta elige la estrategia "docling_hybrid" si quieres
```

- **Con Docker:** `task docker:up:media` (usa `docker-compose.media.yml`: la imagen incluye ffmpeg y el grupo `media`, monta tus medios y guarda las cachés de modelos en un volumen).
- Los **medios de prueba no vienen en el repositorio** (son de terceros): usa los tuyos. Las citas de audio y video llevan el tiempo (`12:30–13:10`) y la interfaz tiene un botón "▶" que reproduce ese instante.
- Las variables (`DOCLING_*`, `WHISPER_*`, `MEDIA_*`) están comentadas en [`.env.example`](.env.example). Detalles, tiempos reales y límites (errores de OCR y de transcripción) en el [README de rag-intermedio](rag-intermedio/README.md).

## LM Studio y Docker

LM Studio corre en tu máquina, no en el contenedor. Con `LOCAL_LLM_URL=http://localhost:1235/v1` en el `.env`, dentro de Docker la dirección se traduce sola a
`host.docker.internal`, pero eso **solo funciona si LM Studio acepta conexiones de la red local** (*Developer → Server Settings → «Serve on Local Network»*).
**Solo Linux**, sin tocar LM Studio: `task docker:up:host` en el proyecto que uses (los contenedores comparten la red de tu máquina).
Revisa también que el puerto de `LOCAL_LLM_URL` sea el de tu servidor y que los modelos estén cargados; si LM Studio no puede cargar un modelo
(por memoria), responde un error y la aplicación lo muestra.

## Variables que cambian según el proyecto

El `.env` de la raíz es **compartido** por los tres proyectos, así que cada uno fija lo suyo para no pisar a los demás:

| Variable | rag-simple | rag-intermedio | rag-avanzado |
| --- | --- | --- | --- |
| Almacén | `STORE=neon` (o `memory`) | `STORE=pinecone` (o `memory`) | `AVANZADO_STORE=postgres` (o `memory`) |
| Credenciales | `NEON_DATABASE_URL` | `PINECONE_API_KEY` | ninguna: la base corre en Docker |

Los compose y las tareas (`task backend`) de cada proyecto ya fijan su valor; solo importa si arrancas a mano.

## Agregar otra implementación

Una carpeta nueva `rag-<nombre>/` con la misma estructura. Cada proyecto es autocontenido (no comparte código con los demás) a
propósito: así se puede leer, copiar y modificar uno sin entender los otros.
