# rag-intermedio

El RAG de la Sesión 6 convertido en servicio: lo que se añade cuando el [`rag-simple`](../rag-simple) ya funciona y
**empieza a fallar** con documentos reales, códigos exactos y usuarios distintos. Misma estructura (FastAPI + interfaz
en Go), pero con un pipeline que la interfaz **muestra etapa por etapa**.

> Sigue siendo **one-shot**: cada pregunta es independiente, sin historial ni agentes. (La reescritura de preguntas de
> seguimiento necesita conversación; queda fuera a propósito y se verá en sesiones posteriores.)

## Qué cambia respecto a rag-simple

| Problema del RAG simple | Pieza que lo ataca | Dónde |
| --- | --- | --- |
| Los documentos reales no son Markdown | **Ingesta multiformato** (Markdown, HTML, Word) | `ingest.py` |
| Los PDF (también escaneados), el audio y el video | **Ingesta moderna**: Docling + OCR automático, faster-whisper, citas con página y tiempo | `pdf.py`, `media.py` |
| Fragmentos que cortan una idea o la diluyen | **Padre-hijo**: se busca con trozos pequeños, se entrega la sección | `chunking.py`, `rag.py` |
| Los códigos exactos (`OPT-1107`, `E-217`) no se "entienden" | **BM25** (búsqueda por palabras) | `lexical.py` |
| Cada buscador tiene su punto ciego | **Híbrida con RRF**: fusiona por posición, no por puntaje | `fusion.py` |
| La respuesta depende de quién pregunta (residencial / Negocio) | **Metadatos y filtros**, aplicados dentro de la búsqueda | `chunking.py`, `store.py` |
| El mejor fragmento queda tercero | **Reranking** con un LLM (en producción, un cross-encoder) | `rag.py` |
| "Parece que funciona" | **Evaluación por modo y por tipo de pregunta** | `evaluation.py` |

| | |
| --- | --- |
| Documentos | `manual_soporte.md`, `guia_tecnica.html` (con menús y scripts que hay que descartar), `condiciones_servicio.docx` (con tablas y partes Residencial / Negocio) y dos PDF: `acta_escaneada.pdf` (un escaneo: necesita OCR) y `boletin_mantenimiento.pdf` (con una tabla) |
| Medios | audio y video en `backend/data/media/` (ignorada por Git; ver más abajo) |
| Almacén | **Pinecone** (índice → namespace → registros), con filtros por metadatos |
| Búsqueda | vectorial, BM25 o híbrida (RRF) |
| Evaluación | 26 preguntas de 4 tipos (paráfrasis, código exacto, tabla, filtro): hit@k, MRR y contexto por modo |

## El flujo

```
 INGESTA (una vez)
 .md .html .docx .pdf ─► leer (texto + estructura) ─► fragmentar (+ metadatos) ─► vectorizar ─► Pinecone
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

## Ingesta moderna: PDF, audio y video

Todo lo de esta sección es **opcional** y vive en el grupo `media` (Docling, faster-whisper, `av<16` y torch solo CPU; varios GB).
Sin él la app base funciona igual: los PDF y los medios se **omiten con una advertencia** ("instala el grupo `media`: `task install:media`")
y el resto se ingesta. Las dependencias se importan de forma perezosa, solo al usarlas.

| Formato | Cómo se lee | Qué queda en los metadatos |
| --- | --- | --- |
| `.md` `.html` `.docx` | lectores propios (`ingest.py`) | `source`, `section`, `audience` |
| `.pdf` | **Docling** (`pdf.py`): títulos, listas y tablas; **OCR automático** | lo anterior + `page` (y `page_end`) |
| `.mp3` `.wav` `.m4a` `.flac` | **faster-whisper** (`media.py`) | `type=audio`, `media`, `start_s`, `end_s` |
| `.mp4` `.mkv` `.mov` `.webm` | ffmpeg extrae el audio y se transcribe igual | `type=video`, `media`, `start_s`, `end_s` |

**OCR automático.** Primero se convierte el PDF *sin* OCR (rápido). Si la mitad o más de las páginas traen menos de `DOCLING_MIN_CHARS`
(40) caracteres, se reintenta con OCR. Se decide por página y no por documento. Cada archivo reporta si usó OCR, cuántas páginas leyó,
cuántos segundos tardó y sus advertencias. `DOCLING_MAX_PAGES` (30) limita las páginas por PDF para que uno de 142 no tarde una
eternidad en CPU; si hay tope, la advertencia lo dice ("solo se leyeron las primeras 30 de 142 páginas").

**Audio y video.** Los segmentos de Whisper (con `start` y `end`) se agrupan en ventanas de unos 50 s con 8 s de solape
(`MEDIA_WINDOW_SECONDS`, `MEDIA_OVERLAP_SECONDS`); cada ventana es un fragmento, con cualquier estrategia. La transcripción se guarda en
disco por **hash del archivo + modelo + idioma + vocabulario**: volver a ingestar es instantáneo. Se usa faster-whisper directo y no el
pipeline de audio de Docling porque este **no conserva las marcas de tiempo**, y sin ellas no se puede citar "minuto 12:30".

**Estrategia `docling_hybrid`.** Es el `HybridChunker` de Docling: sigue la jerarquía del documento (encabezados, tablas, listas) y respeta un
límite de **tokens** (`max_tokens`, 256 por defecto): parte lo que lo excede y une lo muy pequeño. Se vectoriza `chunker.contextualize(chunk)`
(encabezados + texto). **No es la "búsqueda híbrida"** (BM25 + vectores) de la que habla el resto de este README: aquí "híbrido" es
*estructura + tokens*. El tokenizer (`DOCLING_TOKENIZER`) debería ser el del modelo de embeddings: por defecto el de nomic v2 si ese es el modelo
activo y, si no, `sentence-transformers/all-MiniLM-L6-v2`. *Decisión:* md, html y docx **no** pasan por Docling (sus lectores ya detectan
audiencia y aplanan tablas, y está probado); con `docling_hybrid` usan "por oraciones" con un tamaño equivalente (`max_tokens` × 0,7 palabras)
y el reporte lo indica como `sentences (respaldo de docling_hybrid)`. Audio y video no se fragmentan otra vez: cada ventana ya es un fragmento.

**Citas con página y tiempo.** Cada fuente muestra `p. 12` (con enlace al PDF en esa página) o `12:30–13:10`. En audio y video hay un botón
**▶ mm:ss** que reproduce el medio en ese instante con un `<audio>`/`<video>` HTML5. El backend sirve los medios con
`GET /api/media/{nombre}` (el nombre se busca entre los archivos de la carpeta: no se arma ninguna ruta; nombres con separadores dan 400) y
atiende `Range` (206), así que el navegador salta sin descargar todo; el proxy de Go deja pasar esas cabeceras (hay una prueba). La pestaña
*Documentos* lista los PDF y los medios con su duración y un reproductor.

### Con y sin `media`

```bash
task install            # base: md, html, docx (PDF y medios se omiten con una advertencia)
task install:media      # + Docling, faster-whisper, av<16, torch CPU (varios GB; requiere ffmpeg para el video)
mkdir -p backend/data/media   # copia aquí tus audios y videos (no van a Git)
task test               # pruebas rápidas: sin red y con dobles de Docling y Whisper
task test:media         # pruebas REALES (se omiten solas si faltan el grupo, ffmpeg o los medios)
task docker:up:media    # Docker con el grupo media (INSTALL_MEDIA=true) y volumen para modelos y cachés
```

La **primera vez** se descargan modelos de Hugging Face: los de Docling (layout, tablas, OCR), el tokenizer y Whisper (`base` ≈ 140 MB,
`small` ≈ 460 MB). Después quedan en caché. `av<16` está fijado a propósito: con `av` 16 o superior, faster-whisper 1.2.x falla con
`TypeError: open() got an unexpected keyword argument 'metadata_errors'`. El torch CPU se fija con un `[[tool.uv.index]]` explícito.

### Variables de entorno nuevas

| Variable | Por defecto | Qué hace |
| --- | --- | --- |
| `MEDIA_DIR` | `backend/data/media` | carpeta de audios y videos (ignorada por Git) |
| `MEDIA_CACHE_DIR` | `backend/.cache/media` | transcripciones y conversiones de Docling |
| `DOCLING_MAX_PAGES` | `30` | tope de páginas por PDF (`0` = sin tope) |
| `DOCLING_OCR` | `auto` | `auto`, `off` o `force` |
| `DOCLING_MIN_CHARS` | `40` | caracteres por debajo de los cuales una página cuenta como "sin texto" |
| `DOCLING_TOKENIZER` | según el modelo de embeddings | tokenizer del `HybridChunker` |
| `WHISPER_MODEL` | `small` | `base` es más rápido y menos preciso |
| `WHISPER_LANGUAGE` | automático | p. ej. `es` o `en` |
| `WHISPER_PROMPT` | vacío | vocabulario (nombres propios, siglas) que orienta a Whisper |
| `MEDIA_WINDOW_SECONDS` / `MEDIA_OVERLAP_SECONDS` | `50` / `8` | ventana de transcripción y solape |
| `PROXY_TIMEOUT_SECONDS` | `900` | (web) espera máxima al backend: la primera ingesta puede tardar minutos |
| `FRAME_CAPTIONS`, `FRAME_INTERVAL_SECONDS`, `VISION_PROVIDER`, `VISION_MODEL` | `false`, `60`, `local`, según proveedor | **experimental**, ver abajo |

### Lo que se midió (y lo que sale mal)

Máquina de 32 CPU sin GPU, embeddings `nomic-embed-text-v2-moe` locales, almacén en memoria, `DOCLING_MAX_PAGES=30`, Whisper `small`.

| Archivo | Resultado |
| --- | --- |
| `acta_escaneada.pdf` (1 pág., escaneo) | sin OCR no sale texto; con OCR sale "ACTA DE INSTALACION… AX-2200 Metros de cable: 42" (≈ 10 s la primera vez, incluye cargar modelos) |
| `boletin_mantenimiento.pdf` (2 págs.) | sin OCR, ≈ 3 s; la tabla de ventanas por zona queda como filas |
| Informe DORA (142 págs.) | sin OCR; solo 30 págs. por el tope (≈ 29 s) → 94 fragmentos con `docling_hybrid` |
| Informe de OpenAI (24 págs., diseño de láminas) | **sin OCR no sale ningún texto** (Docling trata las páginas como imagen) aunque el PDF tenga capa de texto; con OCR ≈ 86-100 s |
| Charla TEDx (audio, 12 min 26 s, inglés) | Whisper `small` ≈ 60 s la primera vez (con la CPU ocupada), 18 fragmentos |
| Video de ciberseguridad (20 min 29 s, inglés) | ≈ 78 s la primera vez, 30 fragmentos |
| Reingesta de todo (caché) | **8 s** en total (casi todo son los embeddings), frente a 288 s la primera vez |

* **Errores de OCR reales:** el informe de OpenAI sale como "OpenAl" (ele minúscula en vez de i mayúscula) en los fragmentos. Un código
  exacto leído por OCR puede quedar mal y BM25 ya no lo encuentra.
* **Errores de ASR reales:** `small` escribió "chatGBT" y "GitHub co-pilot" por "ChatGPT" y "GitHub Copilot". Una pregunta con "ChatGPT" no coincide por
  palabras con la transcripción; los vectores sí la acercan. `WHISPER_PROMPT` o un modelo mayor ayudan; hay que revisar las transcripciones.
* **Preguntar en español sobre contenido en inglés funciona**, porque nomic v2 es multilingüe: "¿Qué dice la charla sobre cómo aprender ingeniería de
  software en la era de la IA?" trajo los fragmentos correctos de la charla en inglés, con su tiempo. Pero no es perfecto: con un modelo local de 3B
  como redactor, dos preguntas sobre las predicciones del video para el año siguiente (en español y en inglés) acabaron en "No encuentro esa información" aunque
  la recuperación trajo fragmentos del video: las predicciones están repartidas por todo el video y los 3 mejores fragmentos no las cubren.
  Subir `k` o usar un modelo mejor lo mejora; la ventana de 50 s también limita preguntas de síntesis.
* **Las citas se comprobaron:** p. 27 del informe DORA, p. 2 y p. 4 del de OpenAI (con `pdftotext`) y los instantes 08:30 y 02:32 (cortando el audio
  con ffmpeg y transcribiéndolo): el contenido coincide con el fragmento citado.
* **Evaluación con el corpus ampliado** (26 preguntas, k = 3, padre-hijo de 35): vectorial 25, BM25 22, híbrida 25 de 26 en hit@1 (la tabla de más
  abajo es del corpus de tres documentos). Con los PDF y los medios añadidos hay más fragmentos que compiten; las cifras no son comparables una a una.

### Fotogramas del video (EXPERIMENTAL, `frames.py`)

Con `FRAME_CAPTIONS=true` se extrae un fotograma cada `FRAME_INTERVAL_SECONDS`, lo describe un modelo con visión (`VISION_PROVIDER`:
`local` por la API compatible con OpenAI con `image_url` en base64, `openai` por la Responses API con `input_image`, `anthropic` con bloques de
imagen) y cada descripción se indexa alineada por tiempo (`content=frame`). Está apagado por defecto y es **débil con este tipo de video**: con el
modelo local de 12B, a 640×360, la escritura manuscrita se lee mal ("AGENTS!" salió como "PACIENTS"), tarda unos 16 s por fotograma y gasta tokens de
razonamiento. Con proveedores de pago cada fotograma es una llamada. La transcripción es lo que sostiene el RAG. Solo se probó la forma de las
peticiones con un transporte HTTP simulado, no una ingesta completa con fotogramas.

## Arquitectura

```
navegador ──► web (Go, :8081) ──/api/*──► backend (FastAPI, :8010) ──► Pinecone
              UI + proxy inverso           rag.py: el pipeline con traza   └► embeddings (LM Studio / OpenAI)
                                           evaluation.py                   └► LLM (Claude / OpenAI / local)
```

```
backend/app/
  ingest.py      lectores de .md, .html, .docx y .pdf    rag.py         pipeline + traza de cada etapa
  pdf.py         Docling + OCR automático + HybridChunker  media.py       faster-whisper, ventanas, caché
  frames.py      (experimental) fotogramas descritos      locations.py   "p. 12" y "12:30–13:10"
  extras.py      dependencias opcionales (grupo media)
  chunking.py    oraciones, sección, padre-hijo, audiencia   evaluation.py  comparar modos por tipo de pregunta
  lexical.py     tokenizador y BM25                       store.py       PineconeStore y MemoryStore (con filtro)
  fusion.py      RRF                                      cache.py       copia local de fragmentos (para BM25)
  embeddings.py  config.py  llm.py  prompts.py  schemas.py  main.py     como en rag-simple
backend/prompts/{rag_respuesta,rerank}/v1.j2   los prompts, versionados
backend/data/docs/                             los 5 documentos (3 de texto y 2 PDF) · backend/data/eval/preguntas.json  el set de prueba
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
- **Con `media` en Docker:** `task docker:up:media` (`docker-compose.media.yml`) construye con `INSTALL_MEDIA=true` (instala ffmpeg y el grupo), monta `./backend/data/media` de solo lectura y usa el volumen `media_cache` para modelos (`HF_HOME`) y transcripciones, en una ruta que el usuario `app` (no root) puede escribir. La imagen pesa varios GB.
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
task test             # 88 pruebas de Python (sin red, con dobles de Docling y Whisper) + vet y pruebas de Go
task test:media       # 4 pruebas reales (Docling, HybridChunker, Whisper); se omiten si falta algo
```

## Qué NO hace

Sin historial (ni reescritura de preguntas de seguimiento), sin búsqueda híbrida nativa
de Pinecone, sin reranker especializado (cross-encoder), sin evaluación de la generación (esa está en `rag-simple`).

## Origen

Corresponde a la Sesión 6 del curso (`edutek-genai-cohorte1/modulo-1/s06-rag-intermedio`): mismos documentos, mismas
preguntas de prueba (se omiten las conversacionales, que necesitan historial) y las mismas piezas.
