# rag-simple

El RAG más pequeño que ya es **formal**: ingesta, recuperación y generación con **citas verificadas**, con
API (FastAPI), interfaz web (Go) y evaluación. Es la versión "de producción en miniatura" de lo que se construye
en la Sesión 5.

> **One-shot a propósito.** Cada pregunta es independiente: no hay historial, ni memoria, ni herramientas, ni
> agentes. Es un flujo fijo de pasos. Eso es lo que se quiere mostrar.

| | |
| --- | --- |
| Documentos | 3 archivos Markdown (`backend/data/docs/`) de un proveedor de internet ficticio, Andes Fibra |
| Fragmentado | por oraciones con solape (también por sección o ventanas fijas) |
| Búsqueda | solo semántica (vectores, similitud coseno) |
| Almacén | **Neon**: PostgreSQL + pgvector (o memoria para pruebas) |
| Generación | Claude, OpenAI o un modelo local; citas `[n]` comprobadas por código |
| Evaluación | hit@k y MRR de la recuperación (gratis) + generación (opcional) |

## El flujo

```
 INGESTA (una vez)
 data/docs/*.md ─► fragmentar ─► vectorizar ─► guardar en Neon (pgvector)
                    (documents.py)  (embeddings.py)   (store.py)

 CONSULTA (cada pregunta, una sola vuelta)
 pregunta ─► vectorizar ─► buscar los k fragmentos más parecidos
                                     │
              prompt = reglas + fragmentos numerados [1] [2] + pregunta        (prompts/rag_respuesta/v1.j2)
                                     │
                         LLM ─► respuesta con citas [1] ─► verificar citas      (rag.py)
```

La interfaz muestra cada pieza: la respuesta con sus citas clicables, **los fragmentos recuperados con su
puntaje**, cuáles citó el modelo, el prompt exacto que se envió, los tiempos por etapa y el costo.

## Arquitectura

```
navegador ──► web (Go, :8080) ──/api/*──► backend (FastAPI, :8000) ──► Neon (pgvector)
              sirve la UI y hace            rag.py: el flujo              └► embeddings (LM Studio / OpenAI)
              de proxy inverso              evaluation.py                 └► LLM (Claude / OpenAI / local)
```

```
backend/app/
  config.py       variables de entorno          rag.py          ingesta + consulta + verificación de citas
  documents.py    leer .md y fragmentar          evaluation.py   hit@k, MRR, dato clave, citas válidas
  embeddings.py   vectorizar (prefijos nomic)    prompts.py      plantillas Jinja2 versionadas
  store.py        NeonStore y MemoryStore        main.py         rutas HTTP
  llm.py          Anthropic / OpenAI / local     schemas.py      contratos de la API
backend/prompts/rag_respuesta/v1.j2      el prompt, versionado como código
backend/data/eval/preguntas.json         14 preguntas de prueba (12 con respuesta + 2 fuera de alcance)
web/                                     main.go (servidor + proxy) y static/ (HTML, CSS, JS sin build)
```

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/), Go 1.22+ y [Task](https://taskfile.dev/installation/) (`task`). Copia `../.env.example` a `.env` (aquí o en la raíz del repo).

```bash
task install          # dependencias de Python
task backend          # terminal 1 → http://localhost:8000/docs
task web              # terminal 2 → http://localhost:8080
```

1. Pestaña **Ingesta** → *Ingestar* (crea la tabla en Neon y guarda los fragmentos).
2. Pestaña **Preguntar** → haz una pregunta o pulsa un ejemplo.
3. Pestaña **Evaluación** → mide la recuperación (gratis) y, si quieres, la generación (cuesta centavos).

**Sin cuentas ni servidores (`task demo`):** usa vectores "fake" (un hash de palabras: solo coinciden palabras
repetidas, no entiende significado), un "modelo" extractivo y el almacén en memoria. Sirve para ver la interfaz y el
flujo; la calidad real se ve con embeddings reales (LM Studio con `nomic-embed-text-v2-moe`).

### Con Docker

Cada proyecto trae su `docker-compose.yml` (backend de Python + interfaz en Go). Solo necesitas Docker.

```bash
docker compose up --build -d          # o: task docker:up   → http://localhost:8080  (API en http://localhost:8000/docs)
docker compose down                   # o: task docker:down
```

- **Configuración:** lee `.env` de esta carpeta y, si no existe, el de la raíz del repo (`../.env`). Ambos son opcionales; sin ellos
  arranca, pero necesitarás las credenciales de Neon y del modelo para ingestar y preguntar.
- **LM Studio (embeddings o modelo local):** corre en tu máquina, no en el contenedor. `LOCAL_LLM_URL=http://localhost:1235/v1`
  se traduce solo a `host.docker.internal:1235` dentro de Docker, pero **eso solo funciona si LM Studio acepta conexiones de la red
  local**: por defecto escucha únicamente en `127.0.0.1` y el contenedor recibe *connection refused*. Dos soluciones:
  1. En LM Studio: **Developer → Server Settings → «Serve on Local Network»** (sirve en Linux, Mac y Windows; abre el servidor a tu red local).
  2. **Solo Linux**, sin tocar LM Studio: `task docker:up:host` (equivale a
     `docker compose -f docker-compose.yml -f docker-compose.host.yml up --build -d`). Los contenedores comparten la red de tu
     máquina y los servicios escuchan solo en `127.0.0.1`.

  Revisa también que el **puerto** de `LOCAL_LLM_URL` sea el de tu servidor (LM Studio usa 1234 por defecto) y que el modelo esté cargado.
- **Demo sin cuentas ni servidores:** `docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build -d`
  (o `task docker:demo`): embeddings y modelo "fake", todo en memoria.
- **Editar sin reconstruir:** `backend/data/` (documentos y preguntas) y `backend/prompts/` están montados como solo lectura; cambia un
  archivo y vuelve a ingestar. El código sí requiere `docker compose up --build`.
- Los puertos solo escuchan en `127.0.0.1` (tu equipo). El backend corre sin privilegios (usuario `app`).

## Configuración

Ver `../.env.example`. Las que más importan: `LLM_PROVIDER`, `EMBEDDINGS_PROVIDER`, `STORE` (`neon` | `memory`),
`NEON_DATABASE_URL`. La tabla de Neon es `rag_simple_chunks` y **este proyecto es su dueño**: cada ingesta la
borra y la recrea, así cambiar de modelo de embeddings (otra dimensión) o de estrategia es solo reingestar.

## Cosas para probar (la gracia del playground)

1. **Parecida pero ausente.** Pregunta "¿Qué plan de televisión por cable ofrecen…?" y mira el puntaje del mejor
   fragmento: es casi igual al de una pregunta con respuesta. Un **umbral** no la detecta; lo que la salva es el
   prompt ("No encuentro…"). En *Evaluación*, las preguntas fuera de alcance muestran ese puntaje.
2. **Chunking.** Reingesta con *Por sección* y con *Por oraciones (40 palabras)* y compara el hit@1 en *Evaluación*.
3. **k.** Sube y baja los fragmentos: más k = más probabilidad de incluir el correcto, pero más tokens y más ruido.
4. **Verificación de citas.** Si el modelo cita `[7]` habiendo recibido 4, la UI lo marca como invención.
5. **Prompt.** Copia `v1.j2` a `v2.j2`, cámbialo y compara con la misma evaluación.

## Pruebas

```bash
task test             # 26 pruebas de Python (sin red) + vet y pruebas de Go
```

## Qué NO hace (a propósito)

Sin historial de conversación, sin reescritura de consultas, sin búsqueda por palabras exactas, sin filtros por
metadatos, sin reranking, sin PDFs ni Word. Todo eso es [`../rag-intermedio`](../rag-intermedio).

## Origen

Corresponde a la Sesión 5 del curso (`edutek-genai-cohorte1/modulo-1/s05-fundamentos-rag`), reescrita como
servicio: mismas ideas, mismo caso, mismo set de evaluación.
