# rag-intermedio

El RAG de la Sesión 6 convertido en servicio: lo que se añade cuando el [`rag-simple`](../rag-simple) ya funciona y
**empieza a fallar** con documentos reales, códigos exactos y usuarios distintos. Misma estructura (FastAPI + interfaz
en Go), pero con un pipeline que la interfaz **muestra etapa por etapa**.

> Sigue siendo **one-shot**: cada pregunta es independiente, sin historial ni agentes. (La reescritura de preguntas de
> seguimiento —"¿y los domingos?"— necesita conversación; queda fuera a propósito.)

## Qué cambia respecto a rag-simple

| Problema del RAG simple | Pieza que lo ataca | Dónde |
| --- | --- | --- |
| Los documentos reales no son Markdown | **Ingesta multiformato** (Markdown, HTML, Word) | `ingest.py` |
| Fragmentos que cortan una idea o la diluyen | **Padre-hijo**: se busca con trozos pequeños, se entrega la sección | `chunking.py`, `rag.py` |
| Los códigos exactos (`OPT-1107`, `E-217`) no se "entienden" | **BM25** (búsqueda por palabras) | `lexical.py` |
| Cada buscador tiene su punto ciego | **Híbrida con RRF**: fusiona por posición, no por puntaje | `fusion.py` |
| La respuesta depende de quién pregunta (residencial / Negocio) | **Metadatos y filtros**, aplicados dentro de la búsqueda | `chunking.py`, `store.py` |
| El mejor fragmento queda tercero | **Reranking** con un LLM (en producción, un cross-encoder) | `rag.py` |
| "Parece que funciona" | **Evaluación por modo y por tipo de pregunta** | `evaluation.py` |

| | |
| --- | --- |
| Documentos | 3 formatos: `manual_soporte.md`, `guia_tecnica.html` (con menús y scripts que hay que descartar) y `condiciones_servicio.docx` (con tablas y partes Residencial / Negocio) |
| Almacén | **Pinecone** (índice → namespace → registros), con filtros por metadatos |
| Búsqueda | vectorial, BM25 o híbrida (RRF) |
| Evaluación | 26 preguntas de 4 tipos (paráfrasis, código exacto, tabla, filtro): hit@k, MRR y contexto por modo |

## El flujo

```
 INGESTA (una vez)
 .md .html .docx ─► leer (texto + estructura) ─► fragmentar (+ metadatos) ─► vectorizar ─► Pinecone
                                                       │                                      (vectores + filtros)
                                                       └─► copia local de los fragmentos ─► índice BM25 (memoria)

 CONSULTA (cada pregunta)
                  ┌─► búsqueda vectorial (significado) ──┐
 pregunta ────────┤                                      ├─► fusión RRF ─► [padre-hijo] ─► [reranking] ─► top-k
                  └─► búsqueda BM25 (palabras exactas) ──┘                  hijo → sección      LLM reordena
                        ambas con FILTRO por audiencia
                                                          top-k ─► prompt con citas ─► LLM ─► verificar citas
```

Pinecone guarda los vectores; **BM25 no vive en Pinecone**. Al ingestar se guarda una copia local de los fragmentos
(`backend/.cache/chunks.json`, ignorada por Git) y BM25 se reconstruye desde ahí al arrancar. Pinecone y otros ofrecen
búsqueda híbrida nativa con vectores dispersos; aquí se construye a mano para que se vea cómo funciona cada pieza.

## Arquitectura

```
navegador ──► web (Go, :8081) ──/api/*──► backend (FastAPI, :8010) ──► Pinecone
              UI + proxy inverso           rag.py: el pipeline con traza   └► embeddings (LM Studio / OpenAI)
                                           evaluation.py                   └► LLM (Claude / OpenAI / local)
```

```
backend/app/
  ingest.py      lectores de .md, .html y .docx          rag.py         pipeline + traza de cada etapa
  chunking.py    oraciones, sección, padre-hijo, audiencia   evaluation.py  comparar modos por tipo de pregunta
  lexical.py     tokenizador y BM25                       store.py       PineconeStore y MemoryStore (con filtro)
  fusion.py      RRF                                      cache.py       copia local de fragmentos (para BM25)
  embeddings.py  config.py  llm.py  prompts.py  schemas.py  main.py     como en rag-simple
backend/prompts/{rag_respuesta,rerank}/v1.j2   los prompts, versionados
backend/data/docs/                             los 3 documentos · backend/data/eval/preguntas.json  el set de prueba
```

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/), Go 1.22+, [Task](https://taskfile.dev/installation/) (`task`) y una cuenta de Pinecone (plan gratuito). Copia `../.env.example` a `.env`.

```bash
task install          # dependencias de Python
task backend          # terminal 1 → http://localhost:8010/docs
task web              # terminal 2 → http://localhost:8081
```

1. **Ingesta** → *Ingestar* (padre-hijo por defecto): crea el índice en Pinecone si no existe y lo llena.
2. **Preguntar** → elige búsqueda, audiencia, k, y activa el reranking si quieres. Mira *Cómo se buscó*.
3. **Evaluación** → *Comparar modos* (gratis).

`task demo` arranca sin cuentas ni servidores (vectores "fake", modelo extractivo, memoria). Sirve para ver la interfaz y
el flujo; las cifras reales salen con embeddings reales.

### Con Docker

Cada proyecto trae su `docker-compose.yml` (backend de Python + interfaz en Go). Solo necesitas Docker.

```bash
docker compose up --build -d          # o: task docker:up   → http://localhost:8081  (API en http://localhost:8010/docs)
docker compose down                   # o: task docker:down
```

- **Configuración:** lee `.env` de esta carpeta y, si no existe, el de la raíz del repo (`../.env`). Ambos son opcionales; sin ellos
  arranca, pero necesitarás las credenciales de Pinecone y del modelo para ingestar y preguntar.
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
- **Caché de BM25:** el volumen `rag-intermedio_cache` guarda la copia local de los fragmentos; sobrevive a `docker compose down` (se borra con `docker compose down -v`).
- Los puertos solo escuchan en `127.0.0.1` (tu equipo). El backend corre sin privilegios (usuario `app`).

### Pinecone, lo que hay que saber

- La **dimensión de un índice no se puede cambiar**. Con `nomic-embed-text-v2-moe` son 768. Si cambias de modelo, usa otro
  índice (`PINECONE_INDEX`) o bórralo en la consola. El proyecto usa el namespace `rag-intermedio` y lo vacía en cada ingesta.
- **Consistencia eventual**: lo recién escrito puede tardar unos segundos en aparecer; la ingesta espera a que el conteo coincida.
- Los IDs de los vectores deben ser **ASCII**; los metadatos, texto, números, booleanos o listas de texto (por eso el texto del
  fragmento viaja como un metadato más).

## Cosas para probar

1. **Código exacto.** Busca "¿Qué indica la alarma OPT-1107?" en *Solo vectores* y en *Solo BM25*. Compara el puntaje y la posición.
2. **Paráfrasis.** "¿Cómo dejo el router como si fuera nuevo?" no comparte palabras con el texto: BM25 falla, los vectores no.
3. **Filtro.** "¿En cuánto tiempo deben reparar una falla?" con audiencia *Negocio* contra *Residencial*: 4 horas contra 48.
   Mira que *Cómo se buscó* solo contiene fragmentos de esa audiencia (más los de uso general).
4. **Padre-hijo.** Reingesta con *Por oraciones* y compara el contexto (palabras) que recibe el modelo en *Evaluación*.
5. **Reranking.** Actívalo y mira si reordena bien o empeora; con un modelo pequeño puede hacer lo segundo. Cuesta una llamada más.
6. **La híbrida no siempre gana.** Compárala en *Evaluación* con distintas estrategias de fragmentado (tabla de abajo). Se calibra (pesos del
   RRF) con preguntas de ajuste distintas de las de prueba, o se hace sobreajuste.

## Resultados medidos (para no asumir)

`nomic-embed-text-v2-moe` (local), 26 preguntas, k = 3, hit@1 sobre 26 (entre paréntesis: paráfrasis sobre 9 · códigos exactos sobre 11).
Con tan pocas preguntas, **1 o 2 aciertos de diferencia son ruido**; mira los patrones, no un punto.

| Fragmentado | Vectorial | BM25 | Híbrida (RRF) | Contexto (palabras) |
| --- | --- | --- | --- | --- |
| Padre-hijo (hijos de 35) | **23** (8 · 11) | 20 (5 · 11) | **23** (7 · 11) | ≈ 330 |
| Por oraciones (70) | 20 (7 · 9) | **21** (5 · 11) | **21** (6 · 10) | ≈ 155 |
| Por oraciones (40) | 18 (6 · 9) | 18 (4 · 10) | 18 (5 · 10) | ≈ 100 |
| Por sección | 21 (6 · 10) | 21 (5 · 11) | **23** (7 · 11) | ≈ 350 |

Lo que se ve: **BM25 acierta casi siempre los códigos exactos** (10-11 de 11) y **los vectores ganan en paráfrasis** (6-8 de 9 contra 4-5);
cada uno cubre el punto ciego del otro. Pero la híbrida **no gana siempre**: con padre-hijo los hijos pequeños ya hacen que los vectores
distingan los códigos, y la híbrida empata con ellos. Con fragmentos por sección sí aporta (23 contra 21). Y fíjate en el contexto: padre-hijo
y sección envían más del doble de palabras al modelo (más tokens) que oraciones de 70.

## Pruebas

```bash
task test             # 27 pruebas de Python (sin red) + vet y pruebas de Go
```

## Qué NO hace

Sin historial (ni reescritura de preguntas de seguimiento), sin OCR para escaneos, sin PDFs, sin búsqueda híbrida nativa
de Pinecone, sin reranker especializado (cross-encoder), sin evaluación de la generación (esa está en `rag-simple`).

## Origen

Corresponde a la Sesión 6 del curso (`edutek-genai-cohorte1/modulo-1/s06-rag-intermedio`): mismos documentos, mismas
preguntas de prueba (se omiten las conversacionales, que necesitan historial) y las mismas piezas.
