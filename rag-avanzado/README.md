# rag-avanzado

El RAG de la sesión de RAG avanzado convertido en servicio: lo que pasa cuando la pregunta **no se contesta buscando textos
parecidos**, sino **construyendo** la respuesta siguiendo relaciones. Misma estructura que [`rag-simple`](../rag-simple) y
[`rag-intermedio`](../rag-intermedio) (FastAPI + interfaz en Go, one-shot, citas verificadas por código, traza por etapas),
con un almacén nuevo: **PostgreSQL + pgvector en Docker**, donde viven los vectores *y el grafo*.

> Sigue siendo **one-shot** y con pipeline **fijo**: una pregunta, una respuesta con citas, sin historial, y el modo lo eliges tú en la
> interfaz. Nada decide por su cuenta qué hacer: el RAG agéntico se verá en sesiones posteriores.

## El problema que los vectores no resuelven

El catálogo ficticio de servicios de TI de Andes Fibra solo dice, en cada sección, **de qué depende** ese servicio. Nadie escribió
«de la base de datos de clientes dependen X, Y y Z»:

> *Si cae la base de datos de clientes, ¿qué servicios dejan de funcionar?*

La respuesta (siete servicios) hay que **construirla** recorriendo las dependencias **al revés**, varios saltos. Una búsqueda por
similitud devuelve los fragmentos que *se parecen* a la pregunta (el de la base de datos, el de autenticación…), no la cadena completa.

| Pieza | Qué hace | Dónde |
| --- | --- | --- |
| **Ingesta** + **Contextual Retrieval** | Fragmenta el catálogo; opcionalmente un LLM escribe una frase que sitúa cada fragmento («este fragmento describe…») y se antepone *antes* de vectorizar | `ingest.py`, `rag.py` |
| **Grafo curado** | 24 entidades y 40 tripletas escritas a mano, con su procedencia (sección y oración) | `graph.py`, `data/graph/` |
| **Grafo extraído** | Un LLM extrae tripletas por sección; se **normalizan** (`la Red troncal` → `Red troncal`), se **validan** y se **miden** contra el curado (precisión y recall) | `graph.py`, `prompts/extraer_tripletas/` |
| **Grafo en SQL** | Entidades, relaciones y comunidades en tablas de PostgreSQL; los recorridos son **CTE recursivas** (se muestran en la interfaz) | `graphstore.py` |
| `graph_local` | Enlaza entidades de la pregunta, recorre N saltos en el sentido correcto, devuelve **hechos** con procedencia + los fragmentos de origen | `rag.py` |
| `graph_global` | Comunidades (Louvain) + un resumen de cada una + mapa-reducción: versión **simplificada** de la búsqueda global de Microsoft GraphRAG | `graph.py`, `rag.py` |
| `hybrid_graph` | Vectores + hechos del grafo en **un solo contexto numerado** (`[1] hecho`, `[5] fragmento`…) | `rag.py` |
| **Descomposición** | Un LLM separa la pregunta en subpreguntas; cada una se recupera y las listas se fusionan con **RRF** (para `vector` y `hybrid_graph`) | `rag.py`, `fusion.py` |
| **Citas verificadas** | El código comprueba que cada `[n]` apunta a un elemento real (hecho, fragmento o comunidad) y marca las respuestas sin citas | `rag.py` |
| **Evaluación** | 12 preguntas multi-salto, 10 configuraciones: cobertura del contexto, recall y precisión de la respuesta, completas, citas válidas, costo y latencia | `evaluation.py` |

| | |
| --- | --- |
| Documento | `servicios_ti.md`: 16 secciones (servicios de TI de Andes Fibra, ficticio) más una introducción |
| Grafo curado | 24 entidades (16 servicios y 8 equipos), 40 tripletas: `depende_de` (20), `corre_en` (5), `administrado_por` (15) |
| Almacén | **PostgreSQL 17 + pgvector** (imagen oficial `pgvector/pgvector`), servicio `db` del compose; en memoria (NumPy + networkx) para pruebas y `task demo` |
| Preguntas | Las 8 multi-salto de la sesión del curso + 4 extra (una de **control**: directa, donde los vectores bastan) |

## El flujo

```
 INGESTA (una vez)
 catálogo .md ─► fragmentar ─► [LLM: frase de contexto] ─► vectorizar ─► PostgreSQL: av_chunks (vector y vector con contexto)

 GRAFO (una vez por origen)
 curated:   servicios_ti_grafo.json ──────────────────────────────────────────────┐
 extracted: sección ─► LLM ─► tripletas crudas ─► normalizar ─► validar ───────────┤──► PostgreSQL: av_entities, av_relations
                                                       └─► medir contra el curado   │       └─► comunidades (Louvain) ─► [LLM: resumen] ─► av_communities

 CONSULTA (cada pregunta; tú eliges el modo)
                        ┌─ vector        ──────────────────────────► top-k de fragmentos
 pregunta ─► [descomponer en subpreguntas + RRF]
                        ├─ graph_local   ─► enlazar entidades ─► CTE recursiva (N saltos) ─► hechos + fragmentos de origen
                        ├─ graph_global  ─► resúmenes de comunidades ─► MAPA ─► REDUCCIÓN
                        └─ hybrid_graph  ─► hechos del grafo + fragmentos vectoriales ─► un contexto numerado
 contexto [1] [2]… ─► prompt con citas ─► LLM ─► verificar citas
```

## Arquitectura

```
navegador ──► web (Go, :8082) ──/api/*──► backend (FastAPI, :8020) ──► PostgreSQL + pgvector (servicio db)
              UI + proxy inverso           rag.py: pipeline con traza       └► embeddings (LM Studio / OpenAI)
              (SVG del grafo, sin libs)    graph.py · graphstore.py         └► LLM (Claude / OpenAI / local)
                                           evaluation.py
```

```
backend/app/
  rag.py          el pipeline: ingesta, construcción del grafo, los cuatro modos, descomposición, citas
  graph.py        extracción, normalización, validación, precisión/recall, estadísticas y comunidades
  graphstore.py   PgGraphStore (CTE recursivas) y MemoryGraphStore (networkx), con el MISMO resultado
  store.py        PgVectorStore (pgvector, operador <=>) y MemoryStore · pg.py: conexión con reintento si se cae
  evaluation.py   las 10 configuraciones y sus métricas · fusion.py: RRF · cache.py: caché en disco de llamadas costosas
  ingest.py  embeddings.py  llm.py  prompts.py  schemas.py  config.py  main.py     como en los otros proyectos
backend/prompts/<nombre>/vN.j2    los prompts versionados (CHANGELOG.md explica cada versión)
backend/data/{docs,graph,eval}/   el catálogo, el grafo curado y las preguntas
```

## Cómo correrlo

Requisitos: [uv](https://docs.astral.sh/uv/), Go 1.22+, [Task](https://taskfile.dev/installation/) y Docker (para PostgreSQL). Copia
`../.env.example` a `.env`. No hace falta ninguna cuenta en la nube: la base de datos corre en tu máquina.

```bash
task install          # dependencias de Python
task db:up            # PostgreSQL + pgvector en Docker (127.0.0.1:5434)
task backend          # terminal 1 → http://localhost:8020/docs
task web              # terminal 2 → http://localhost:8082
```

1. **Ingesta** → *Ingestar* (activa *Contextual* si quieres compararlo; usa el LLM).
2. **Grafo** → *Construir grafo curado* (gratis) y *Extraer grafo con el LLM* (una llamada por sección); mira la precisión y el recall.
   Luego *Resumir comunidades* (para el modo global).
3. **Preguntar** → elige el modo. Mira el subgrafo, los hechos con su procedencia, el SQL del recorrido y las etapas.
4. **Evaluación** → *Cobertura del contexto* (gratis) o *Con respuestas del modelo*.

`task demo` arranca sin cuentas, sin Docker y sin servidores (modelo y vectores «fake», grafo y fragmentos en memoria). Sirve para ver
la interfaz y el flujo; las cifras reales salen con modelos reales. Con el proveedor `fake`, el extractor deja a propósito dos
defectos (una entidad inventada y una arista con el sujeto cambiado) para que se vea trabajar a la validación.

### Con Docker

```bash
docker compose up --build -d          # o: task docker:up   → http://localhost:8082  (API en http://localhost:8020/docs)
docker compose down                   # o: task docker:down  (el volumen `pgdata` conserva la base; `down -v` la borra)
```

- **Servicios:** `db` (`pgvector/pgvector:0.8.1-pg17`, volumen con nombre `pgdata`, healthcheck con `pg_isready`), `backend` y `web`. Los tres corren sin
  privilegios (`postgres` uid 999; `app` uid 10001) y con healthcheck; `web` espera a `backend` y `backend` espera a que `db` esté sana.
- **Puertos:** solo en `127.0.0.1`. Web 8082 y API 8020. El `db` se publica en `127.0.0.1:5434` únicamente para depurar (`psql`, `task backend`); el backend
  lo alcanza por la red interna. La contraseña de desarrollo (`POSTGRES_PASSWORD`, por defecto `rag`) no protege nada fuera de tu equipo: no expongas ese puerto.
- **Configuración:** lee `.env` de esta carpeta y, si no existe, el de la raíz del repo (`../.env`). `AVANZADO_STORE` (no `STORE`, que usa `rag-simple`) elige
  `postgres` o `memory`; `DATABASE_URL` la fija el compose.
- **LM Studio (embeddings o modelo local):** corre en tu máquina, no en el contenedor. `LOCAL_LLM_URL=http://localhost:1235/v1` se traduce solo a
  `host.docker.internal`, pero **eso solo funciona si LM Studio acepta conexiones de la red local** (*Developer → Server Settings → «Serve on Local Network»*).
  **Solo Linux**, sin tocar LM Studio: `task docker:up:host` (`docker compose -f docker-compose.yml -f docker-compose.host.yml up --build -d`); los contenedores
  comparten la red de tu máquina y la base se alcanza por `127.0.0.1:5434`.
- **Demo sin cuentas ni servidores:** `task docker:demo` (`docker-compose.demo.yml`): proveedores `fake`, grafo y fragmentos en memoria, y **no se levanta `db`**.
- **Editar sin reconstruir:** `backend/data/` (catálogo, grafo curado, preguntas) y `backend/prompts/` están montados como solo lectura.
- **Caché:** el volumen `cache` guarda las frases de contexto, las tripletas y los resúmenes ya generados (cada una costó una llamada al LLM). Cambiar el modelo,
  el catálogo o el prompt cambia la clave y se vuelve a llamar; hay una casilla «Ignorar la caché».
- LM Studio descarga un modelo para cargar otro (al alternar embeddings y LLM) y responde un 400 «Model is unloaded» hasta recargarlo: el backend reintenta solo.

## El grafo también puede vivir en una base relacional

Las relaciones están en una tabla (`av_relations`: `graph, subject, relation, object, section, evidence`) y «quién depende de X, directa o indirectamente»
es una **CTE recursiva**. Esta es la del recorrido *al revés* (la interfaz muestra la que se ejecutó, con sus parámetros):

```sql
WITH RECURSIVE walk AS (
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           1 AS depth, r.object AS seed, ARRAY[r.object, r.subject] AS path, r.subject AS frontier
    FROM av_relations r
    WHERE r.graph = :graph AND r.object = ANY(:seeds) AND r.relation = ANY(:relations)
  UNION ALL
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           w.depth + 1, w.seed, w.path || r.subject, r.subject
    FROM av_relations r
    JOIN walk w ON r.object = w.frontier
    WHERE r.graph = :graph AND r.relation = ANY(:relations)
      AND w.depth < :hops AND NOT r.subject = ANY(w.path)      -- tope de saltos y sin dar vueltas en un ciclo
)
SELECT subject, relation, object, section, evidence, min(depth) AS depth, (array_agg(seed ORDER BY depth, seed))[1] AS seed
FROM walk GROUP BY subject, relation, object, section, evidence ORDER BY min(depth), subject, object;
```

El recorrido *hacia adelante* (de qué depende) es la misma consulta con `subject` y `object` intercambiados. Hay una versión en memoria con networkx que devuelve
**exactamente lo mismo** (una prueba compara ambas sobre 36 combinaciones de entidades, sentidos y saltos). Cuándo no alcanza: grafos de millones de aristas,
consultas de patrones complejos o algoritmos de grafos (caminos más cortos, centralidad) como operación habitual; ahí compensa una base de grafos.

## Qué hace cada modo y cuándo conviene

| Modo | Qué entrega al modelo | Conviene cuando | Punto débil |
| --- | --- | --- | --- |
| `vector` (k, + `contextual`, + `decompose`) | Los k fragmentos más parecidos | La respuesta está escrita en un párrafo; preguntas directas | No encadena relaciones: trae lo parecido, no lo conectado |
| `graph_local` | Hechos (tripletas con procedencia) del recorrido + fragmentos de origen | Impacto, dependencias, «quién administra»: la respuesta es una cadena | Si la pregunta no nombra una entidad del grafo, no hay por dónde empezar; una arista falsa contamina todo lo que pasa por ella |
| `graph_global` | Extractos de los resúmenes de las comunidades (mapa-reducción) | Preguntas sobre *todo* el catálogo («¿qué grandes grupos hay?») | Es tan bueno como el resumen (un modelo pequeño inventa); pregunta por detalles y no los tiene |
| `hybrid_graph` (+ `decompose`) | Hechos + fragmentos vectoriales, un contexto numerado | Cobertura amplia sin saber de antemano qué tipo de pregunta llega | Más contexto (más tokens, más ruido para el modelo) |

**Cuándo conviene GraphRAG y cuándo NO** (con las cifras de abajo, no de memoria):

| Conviene | No compensa |
| --- | --- |
| Preguntas **relacionales** o de varios saltos (impacto, dependencias, cadenas) | Preguntas directas: el vector ya acierta y el grafo solo añade costo (ver la pregunta de control) |
| Entidades con **muchas conexiones** (CMDB, contratos, cadena de suministro) | El corpus es pequeño y cabe entero en el contexto |
| Se necesita **explicar el camino**: cada hecho apunta a su oración | Los documentos cambian a diario: hay que **reindexar** y **volver a validar** |
| Hay un inventario de entidades razonablemente estable | No se puede **medir** la extracción (sin un grafo de referencia, no sabes cuánto confiar) |

**Los costos que no se ven en una demo:** *indexar* (una o más llamadas al LLM por fragmento para extraer, y una por comunidad para resumir; aquí 16 + 3 con un corpus
diminuto, miles en uno real), *mantener* (cuando cambia un documento hay que volver a extraer y resolver entidades: que «SSO» y «Servicio de autenticación» sean el mismo nodo),
y *validar* (la extracción es el paso más frágil y menos medido: un modelo pequeño inventa relaciones, las invierte o escribe la misma entidad de varias formas).

## Extraer el grafo con un LLM: lo que se mide

`Grafo` → *Extraer grafo con el LLM* hace una llamada por sección y pasa cada tripleta por **normalización** (relación y entidades a su nombre oficial,
artículos, alias, relaciones invertidas) y **validación** (relación permitida, entidades conocidas, tipos coherentes, el sujeto es la sección, el objeto aparece en el texto
de la sección). Lo que sobrevive se mide contra el grafo curado. El inventario de entidades es curado en ambos grafos: lo que se extrae son las **relaciones**.

Medido con `llama-3.2-3b-instruct` (local, LM Studio), 16 secciones, 16 llamadas (≈ 25 s con v1 y ≈ 10 s con v2), contra las 40 tripletas curadas:

| Prompt | Crudas | Tras normalizar (sin validar) | Tras validar (el grafo guardado) | Rechazadas |
| --- | --- | --- | --- | --- |
| v1: nombres libres | 45 | 38 · precisión 0,79 · recall 0,75 | **30 · precisión 1,00 · recall 0,75** (F1 0,86) | 15 |
| v2: lista de entidades conocidas | 53 | 53 · precisión 0,74 · recall 0,98 | **40 · precisión 0,98 · recall 0,98** (F1 0,98) | 13 |

Lo que se ve: la **validación sube la precisión** (de 0,79 a 1,00 con v1; de 0,74 a 0,98 con v2: rechaza entidades inventadas como «Andes Fibra» o «servidores de
producción», `corre_en` apuntando a un servicio o a un equipo, y objetos que la sección no menciona) pero **no recupera lo que el modelo no vio** (con v1, 10 tripletas curadas
faltan: las cadenas se cortan). La v2 casi iguala al curado; a cambio, el modelo copió a veces una relación cierta pero de *otra* sección (rechazada por «sin respaldo»), y una
tripleta invertida (`Centro de datos Norte corre_en Red troncal`) sí pasó la validación: **una arista falsa que ningún chequeo por reglas atrapa** y que contamina los recorridos que pasan por ella.

## Cosas para probar

1. **Impacto.** «Si cae la base de datos de clientes, ¿qué servicios dejan de funcionar?» en `vector` y en `graph_local`. Mira el subgrafo: la entidad de partida tiene borde grueso y los números sobre las aristas son los saltos.
2. **El sentido importa.** Misma pregunta con *Sentido del recorrido* = «Hacia adelante»: obtienes de qué depende la base de datos (dos elementos), no a quién afecta.
3. **Los saltos.** Con `graph_local` baja *Saltos* a 1 y luego a 2: el contexto pierde a los servicios más lejanos (la evaluación incluye «2 saltos» para verlo).
4. **Procedencia.** Pulsa una cita de un hecho en la respuesta: se resalta su arista en el grafo y la tabla de hechos muestra la oración del catálogo que la respalda.
5. **Descomponer.** «Si falla la pasarela de pagos, ¿qué servicio se ve afectado y qué equipo la administra?» con `hybrid_graph` y *Descomponer*: mira las subpreguntas y la fusión RRF.
6. **Contextual.** Reingesta con *Contextual* y, para que se note, sin *Título de la sección en el texto vectorizado*; compara «Vectores con frase de contexto» en la evaluación.
7. **Extraído vs curado.** Construye el extraído con el prompt v1 y con v2, mira qué tripletas rechazó la validación y qué faltó, y pregunta con *Grafo = Extraído*.
8. **Global.** «¿Qué grandes grupos de servicios hay y qué hace cada uno?» en `graph_global` (después de resumir comunidades) y en `vector`. Lee los resúmenes: ¿se los puede creer?
9. **El control.** «¿Qué equipo administra el Servicio de notificaciones?»: aquí el grafo no tiene nada que aportar sobre los vectores.

## Resultados medidos (para no asumir)

`text-embedding-nomic-embed-text-v2-moe` y `llama-3.2-3b-instruct` (ambos locales, LM Studio), PostgreSQL + pgvector en Docker, 12 preguntas (8 del curso + 4 extra), k = 4 (salvo donde se indica), 4 saltos.
Cada fila mide **primero el contexto** (cobertura: fracción de las entidades esperadas que llegaron al prompt) y **después la respuesta** (recall y precisión de lo que dijo el modelo). Costo: US$ 0 (modelo local).
Con 12 preguntas, **una o dos de diferencia son ruido**; mira los patrones.

| Configuración | Contexto completo | Cobertura del contexto | Recall de la respuesta | Precisión | Completas | Citas válidas | Se abstuvo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vectorial (k=4) | 4/12 | 0,52 | 0,35 | 0,44 | 1/12 | 7/12 | 2 |
| Vectorial (k=8) | 5/12 | 0,68 | 0,40 | 0,51 | 2/12 | 8/12 | 2 |
| Vectorial (k=4) + contextual | 4/12 | 0,46 | 0,45 | 0,57 | 1/12 | 8/12 | 1 |
| Vectorial (k=4) + descomposición | 5/12 | 0,58 | 0,25 | 0,48 | 1/12 | 6/12 | 3 |
| **Grafo local** (curado, 4 saltos) | **11/12** | **0,92** | 0,61 | **0,68** | **4/12** | 7/12 | 1 |
| Grafo local (curado, 2 saltos) | 8/12 | 0,84 | 0,50 | 0,56 | 4/12 | 7/12 | 1 |
| Grafo local (extraído con v2, 4 saltos) | 11/12 | 0,92 | 0,48 | 0,49 | 3/12 | 8/12 | 1 |
| Grafo local (extraído con v1, 4 saltos) | 4/12 | 0,48 | 0,44 | 0,56 | 4/12 | 4/12 | 3 |
| **Híbrido** vectores + grafo | **12/12** | **1,00** | **0,63** | 0,60 | **4/12** | **10/12** | 1 |
| Híbrido + descomposición | 12/12 | 1,00 | 0,39 | 0,46 | 4/12 | 9/12 | 1 |
| Grafo global (comunidades) | 2/12 | 0,30 | 0,19 | 0,36 | 1/12 | 2/12 | 6 |

Latencia promedio por pregunta de 0,2 a 0,9 s con el modelo de 3B parámetros local (la primera configuración, 2,6 s, incluye cargar el modelo); los costos son US$ 0. La fila del grafo extraído con v1 se midió aparte
(solo esa configuración) sobre el grafo construido con el prompt v1; el resto sobre el de v2.

Cobertura del contexto → recall de la respuesta, por tipo (n = 5 impacto · 2 dependencias · 1 encadenada · 1 equipos · 1 global · 2 directas):

| Configuración | Impacto | Dependencias | Encadenada | Equipos | Global | Directa (control) |
| --- | --- | --- | --- | --- | --- | --- |
| Vectorial (k=4) | 0,21 → 0,20 | 0,83 → 0,36 | 0,00 → 0,00 | 0,50 → 0,50 | 1,00 → 0,00 | 1,00 → 1,00 |
| Vectorial (k=8) | 0,36 → 0,28 | 0,83 → 0,60 | 1,00 → 0,00 | 0,75 → 0,25 | 1,00 → 0,00 | 1,00 → 1,00 |
| Grafo local | **1,00 → 0,75** | **1,00 → 0,76** | 1,00 → 0,00 | 1,00 → 0,00 | 0,00 → 0,00 | 1,00 → 1,00 |
| Híbrido | **1,00 → 0,65** | **1,00 → 0,89** | 1,00 → 0,00 | 1,00 → 0,50 | 1,00 → 0,00 | 1,00 → 1,00 |
| Grafo global | 0,24 → 0,10 | 0,24 → 0,42 | 0,00 → 0,00 | 0,00 → 0,00 | 1,00 → 0,00 | 0,50 → 0,50 |

**Qué dicen estas cifras (y qué no):**

- **El grafo gana donde se esperaba**: en impacto y dependencias, la cobertura pasa de 0,21–0,36 (vectores) a 1,00, y el recall de la respuesta de ≈ 0,2–0,3 a 0,65–0,75. Con un 3B, subir k a 8 ayuda a los vectores
  (0,52 → 0,68) pero no cierra la brecha: las cadenas largas no están escritas en ningún fragmento.
- **Recuperar no es razonar.** Con el contexto completo (11/12) el modelo solo nombra el 61 % de lo esperado y acierta completa 4 de 12. En la pregunta encadenada y en la de equipos, el grafo local entrega los hechos
  correctos y el modelo igual responde mal (escribió el equipo de la API de facturación, no el de la base de datos: el hecho de dos saltos estaba ahí). Un modelo más capaz, o un cálculo (un conteo, una unión) hecho por código, lo resolvería; este proyecto no lo hace a propósito.
- **Lo que el grafo no puede**: la pregunta global («¿cuál es el servicio del que dependen más servicios?») no nombra ninguna entidad: el grafo local devuelve 0 de cobertura. Y el modo global con un 3B **es el peor de la tabla**:
  los resúmenes salieron imprecisos (uno dice que los centros de datos «ejecutan» la cola de mensajes), el mapa descartó casi todo y el modelo se abstuvo 6 veces. La cobertura de 1,00 en «global» de las demás filas es en buena parte trivial.
- **Un contexto más grande no es mejor.** El híbrido tiene la mejor cobertura y citas (10/12), pero su precisión baja respecto al grafo local (0,60 contra 0,68): más texto, más ruido para el modelo. **La descomposición empeoró** con este modelo
  (recall 0,63 → 0,39): las subpreguntas no aportaron entidades nuevas y añadieron contexto.
- **La frase de contexto no ayudó en la recuperación** (cobertura 0,52 → 0,46, el recall de la respuesta subió 0,35 → 0,45: ruido). Era de esperar: con el título de la sección ya en el texto vectorizado, el catálogo casi no tiene fragmentos huérfanos.
  Reingesta sin ese título para ver el caso donde sí debería notarse.
- **El control**: en las preguntas directas todos aciertan (1,00 → 1,00); ahí el grafo no aporta, solo cuesta.
- **El grafo extraído depende de la extracción**: con v2 (precisión y recall 0,98) iguala al curado en cobertura; con v1 (recall 0,75) cae a 0,48 porque los recorridos se cortan donde falta una arista.
- **Los saltos importan**: con 2 saltos la cobertura baja de 0,92 a 0,84 (las cadenas de 3 y 4 saltos quedan cortas).

## Pruebas

```bash
task test               # 98 pruebas de Python (sin red ni Docker) + vet y pruebas de Go
task test:integration   # contra el PostgreSQL real: levanta SOLO `db`, corre 8 pruebas y lo elimina con su volumen
```

Cubren: extracción, normalización y validación; recorridos en ambos sentidos, con saltos, varias semillas y ciclos; comunidades; RRF; las citas (válidas, inexistentes, sin
citar); los cuatro modos y la descomposición; la evaluación y que las respuestas esperadas coincidan con el grafo; la API (409, 422, 503); la resiliencia de la conexión a PostgreSQL;
y la interfaz Go. La prueba de integración compara la CTE de PostgreSQL con el recorrido en memoria, mata la conexión a mitad y comprueba que se recupera.

## Versión simplificada de la búsqueda global: qué le falta

Esta `graph_global` es una versión **simplificada** de la búsqueda global de **Microsoft GraphRAG**: una sola partición (Louvain), un resumen por comunidad, un mapa-reducción en una
pasada. Frente a las implementaciones reales le falta:

- **Microsoft GraphRAG:** comunidades *jerárquicas* con Leiden (varios niveles para elegir cuánto detalle leer); que el LLM *extraiga* entidades, relaciones y afirmaciones de cada fragmento (aquí el inventario es curado);
  informes de comunidad con hallazgos puntuados y un mapa con *puntaje de utilidad* que descarta y ordena respuestas parciales; la búsqueda DRIFT; ajuste de prompts por dominio.
- **LightRAG:** indexar entidades *y relaciones* con descripciones, búsquedas de doble nivel (palabras clave locales y globales) y **actualización incremental** sin reindexar todo (evita los resúmenes por comunidad para gastar menos).

## Qué NO hace

Sin historial, sin agentes (ni routers que decidan, ni bucles de corrección, ni LangGraph: el RAG agéntico se verá en sesiones posteriores), sin extracción del inventario de entidades
(es curado), sin comunidades jerárquicas, sin actualización incremental del grafo, sin evaluación con juez LLM (las métricas son por código), sin autenticación en la API.

## Origen

Corresponde a la sesión de RAG avanzado del curso (`edutek-genai-cohorte1/modulo-1/s07-rag-avanzado`): el mismo catálogo ficticio, el mismo grafo curado y las mismas 8 preguntas
multi-salto, más 4 preguntas extra (`data/eval/preguntas_extra.json`) y los prompts de extracción y descomposición, versionados en `backend/prompts/`.
