"use strict";

// ───────────── utilidades ─────────────
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

/** Crea un elemento. Todo el texto entra con textContent: la salida del modelo nunca se interpreta como HTML. */
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) if (child != null) el.append(child);
  return el;
}

/** Igual que h(), para elementos SVG. */
function s(tag, props = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) if (value != null && value !== false) el.setAttribute(key, value);
  for (const child of children.flat()) if (child != null) el.append(child);
  return el;
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
  } catch (error) {
    throw new Error("No se pudo contactar al servidor web.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Error ${response.status}`);
  }
  return data;
}

function showError(message) {
  const box = $("alert");
  box.textContent = message;
  box.hidden = !message;
}

async function guarded(button, task, busyText) {
  showError("");
  const label = button.textContent;
  button.disabled = true;
  if (busyText) button.textContent = busyText;
  try {
    await task();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

const money = (usd) => (!usd ? "US$ 0" : `US$ ${usd.toFixed(usd < 0.01 ? 6 : 4)}`);
const pct = (x) => `${Math.round(x * 100)}%`;
const chip = (text, kind = "") => h("span", { class: `chip ${kind}` }, text);
const short = (text, n = 90) => (text.length > n ? `${text.slice(0, n).trim()}…` : text);
const stat = (value, label) => h("div", { class: "stat" }, h("b", {}, String(value)), h("span", {}, label));
const th = (text, cls = "") => h("th", { class: cls }, text);

let appStatus = null;
let fullGraphs = {};          // grafo completo por origen, para dibujar el subgrafo sobre él

// ───────────── pestañas ─────────────
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "docs") loadDocs();
  if (name === "graph") loadGraphView();
  if (name === "eval") loadEvalConfigs();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ───────────── estado ─────────────
async function refreshStatus() {
  const st = await api("/api/status");
  appStatus = st;
  const built = Object.entries(st.graphs).map(([name, g]) => `${name === "curated" ? "curado" : "extraído"} (${g.entities} nodos, ${g.triples} aristas)`);
  $("chips").replaceChildren(
    chip(`Modelo: ${st.llm_provider} · ${st.llm_model}`),
    chip(`Embeddings: ${st.embeddings_provider}`),
    chip(`Almacén: ${st.store === "postgres" ? "PostgreSQL + pgvector" : "memoria"}`),
    chip(st.chunks ? `${st.chunks} fragmentos${st.contextual_chunks ? ` (${st.contextual_chunks} con contexto)` : ""}` : "Sin ingestar", st.chunks ? "ok" : "warn"),
    chip(built.length ? `Grafo: ${built.join(" · ")}` : "Sin grafo", built.length ? "ok" : "warn"),
  );
  if (st.store_error) showError(`Almacén: ${st.store_error}`);
  updateControls();
  return st;
}

// ───────────── preguntar ─────────────
const MODE_HELP = {
  vector: "Línea base: los k fragmentos más parecidos a la pregunta (similitud coseno). Funciona cuando la respuesta está escrita en un párrafo; falla cuando hay que encadenar relaciones.",
  graph_local: "Búsqueda local: enlaza las entidades que nombra la pregunta, recorre el grafo N saltos (en el sentido correcto: una pregunta de impacto recorre las dependencias AL REVÉS) y entrega los HECHOS con su procedencia más los fragmentos de donde salieron.",
  graph_global: "Búsqueda global (versión simplificada de Microsoft GraphRAG): cada comunidad de servicios tiene un resumen; un LLM extrae de cada resumen lo útil para la pregunta (mapa) y con eso se redacta la respuesta (reducción). Pensada para preguntas sobre TODO el catálogo. Necesita resúmenes (pestaña Grafo).",
  hybrid_graph: "Vectores + hechos del grafo en UN solo contexto numerado ([1], [2]…), cada elemento con su tipo: hecho o fragmento. Cobertura amplia a cambio de más contexto.",
};

function updateControls() {
  const mode = $("mode").value;
  $("mode-help").textContent = MODE_HELP[mode];
  const usesVector = mode === "vector" || mode === "hybrid_graph";
  const usesGraph = mode !== "vector";
  $("decompose").disabled = !usesVector;
  $("contextual").disabled = !usesVector || !(appStatus && appStatus.contextual_chunks);
  $("contextual-wrap").title = appStatus && appStatus.contextual_chunks ? "Busca con los vectores que llevan la frase de contexto" : "Primero reingesta con «Contextual» activado (pestaña Ingesta)";
  if ($("decompose").disabled) $("decompose").checked = false;
  if ($("contextual").disabled) $("contextual").checked = false;
  $("hops").disabled = $("direction").disabled = !(mode === "graph_local" || mode === "hybrid_graph");
  $("graph-source").disabled = !usesGraph;
  $("k").disabled = mode === "graph_global";
}
$("mode").addEventListener("change", updateControls);

const TYPE_LABELS = {
  impacto: "Impacto (al revés)",
  dependencias: "Dependencias (adelante)",
  encadenada: "Encadenada (2 saltos)",
  equipos: "Equipos que intervienen",
  global: "Global (contar)",
  directa: "Directa (control)",
};

async function loadExamples() {
  const examples = await api("/api/examples");
  $("examples").replaceChildren(...Object.keys(TYPE_LABELS).map((type) => {
    const list = examples.filter((e) => e.type === type);
    return h("div", { class: "examples-group" }, h("span", { class: "group-name" }, TYPE_LABELS[type]),
      ...list.map((e) => h("button", {
        type: "button", class: `example${e.origin === "extra" ? " extra" : ""}`, title: e.origin === "extra" ? "Pregunta extra de este proyecto" : "Pregunta del curso",
        onclick: () => { $("question").value = e.question; $("question").focus(); },
      }, e.question)));
  }));
}

/** Convierte "…depende [1]" en texto + botones [1] que llevan al elemento citado. Marca en rojo las citas inexistentes. */
function renderAnswer(text, count) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    nodes.push(text.slice(last, match.index));
    for (const n of match[1].split(/\s*,\s*/).map(Number)) {
      const valid = n >= 1 && n <= count;
      nodes.push(h("button", {
        type: "button", class: `cite${valid ? "" : " invalid"}`,
        title: valid ? `Ir al elemento ${n}` : `El elemento ${n} no existe`,
        onclick: () => valid && focusItem(n),
      }, String(n)));
    }
    last = match.index + match[0].length;
  }
  nodes.push(text.slice(last));
  return nodes;
}

let lastItems = [];
function focusItem(n) {
  const el = $(`item-${n}`);
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  // si es un hecho, resalta su arista en el subgrafo
  document.querySelectorAll("#subgraph .g-edge.focus").forEach((e) => e.classList.remove("focus"));
  const item = lastItems[n - 1];
  if (item && item.kind === "fact") {
    const key = `${item.meta.subject}|${item.meta.relation}|${item.meta.object}`;
    document.querySelectorAll("#subgraph .g-edge").forEach((e) => { if (e.dataset.key === key) e.classList.add("focus"); });
  }
}

const KIND_TEXT = { fact: "hecho", chunk: "fragmento", community: "comunidad" };

function renderStages(stages) {
  return stages.map((stage) => {
    const shown = stage.items.slice(0, 12);
    const items = shown.map((it, i) => {
      const ranks = it.meta && it.meta.ranks ? ` · ${Object.entries(it.meta.ranks).map(([name, pos]) => `${name}#${pos}`).join(" ")}` : "";
      return h("li", {}, h("b", {}, `${i + 1}. ${it.title}`), it.detail ? h("div", { class: "meta" }, short(it.detail + ranks, 110)) : null);
    });
    if (stage.items.length > shown.length) items.push(h("li", {}, h("span", { class: "meta" }, `… y ${stage.items.length - shown.length} más`)));
    return h("div", { class: "stage" },
      h("h3", {}, h("span", {}, stage.title), h("span", { class: "muted small" }, stage.ms ? `${stage.ms.toFixed(0)} ms` : "")),
      h("p", { class: "note" }, stage.note),
      stage.llm ? h("p", { class: "note" }, stage.llm.model ? `${stage.llm.model} · ${stage.llm.tokens_in}→${stage.llm.tokens_out} tokens · ${money(stage.llm.cost_usd)}` : `${stage.llm.calls} llamadas al LLM · ${money(stage.llm.cost_usd)}`) : null,
      items.length ? h("ol", {}, ...items) : h("p", { class: "empty" }, "Sin resultados."));
  });
}

function renderFacts(items) {
  const facts = items.map((it, i) => ({ it, n: i + 1 })).filter((x) => x.it.kind === "fact");
  $("facts-card").hidden = facts.length === 0;
  if (!facts.length) return;
  $("facts").replaceChildren(h("table", {},
    h("thead", {}, h("tr", {}, th("#"), th("Hecho"), th("Salto", "num"), th("Procedencia: sección y oración del catálogo"), th("Fragmento"))),
    h("tbody", {}, ...facts.map(({ it, n }) => h("tr", {},
      h("td", {}, h("button", { type: "button", class: "cite", onclick: () => focusItem(n) }, String(n))),
      h("td", {}, it.text),
      h("td", { class: "num" }, it.meta.depth ? String(it.meta.depth) : "equipo"),
      h("td", {}, h("b", {}, `«${it.meta.section}»`), " ", h("em", {}, it.meta.evidence || "(sin oración)")),
      h("td", {}, it.meta.chunk_id || "—"))))));
}

function renderSql(queries) {
  $("sql-card").hidden = !queries.length;
  if (!queries.length) return;
  const eq = queries[0].equivalent;
  $("sql-note").textContent = eq
    ? "En este modo (almacén en memoria) el recorrido se calculó con networkx. Este es el SQL equivalente que ejecutaría PostgreSQL con el mismo resultado; con el servicio `db` del compose se ejecuta tal cual."
    : "Esta consulta se ejecutó en PostgreSQL. Una CTE recursiva parte de las entidades de la pregunta y sigue las aristas un salto por iteración, sin repetir nodos de un mismo camino (así un ciclo no la deja dando vueltas). El grafo también puede vivir en una base relacional.";
  const names = { reverse: "Al revés: quién depende de la entidad", forward: "Hacia adelante: de qué depende la entidad" };
  $("sql").replaceChildren(...queries.map((q) => h("div", {},
    h("h3", {}, names[q.direction] || q.direction),
    h("pre", { class: "sql" }, q.sql),
    h("p", { class: "sql-params" }, "Parámetros: ", Object.entries(q.params).map(([k, v]) => `:${k} = ${JSON.stringify(v)}`).join(" · ")))));
}

function renderItems(items) {
  lastItems = items;
  $("items").replaceChildren(...items.map((it) =>
    h("div", { class: `hit${it.cited ? " cited" : ""}`, id: `item-${it.number}` },
      h("div", { class: "hit-head" },
        h("span", { class: "num" }, String(it.number)),
        h("span", { class: `kind ${it.kind}` }, KIND_TEXT[it.kind]),
        h("span", { class: "hit-source" }, it.kind === "chunk" ? `${it.meta.source} › ${it.label}` : it.label),
        it.cited ? chip("citado", "ok") : null),
      it.kind === "chunk" && it.meta.context ? h("p", { class: "ctx" }, `Contexto del LLM: ${it.meta.context}`) : null,
      h("p", {}, it.text))));
}

function renderResult(r) {
  $("result").hidden = false;
  $("answer-text").replaceChildren(...renderAnswer(r.answer, r.items.length));
  const badges = [];
  if (!r.usage) badges.push(chip("No se llamó al modelo: no hubo contexto", "warn"));
  else if (r.abstained) badges.push(chip("El modelo dijo que no lo encuentra", "warn"));
  else if (r.invalid_citations.length) badges.push(chip(`Citas inexistentes: ${r.invalid_citations.join(", ")} (posible invención)`, "bad"));
  else if (r.uncited) badges.push(chip("Respondió sin citar: no se puede verificar", "bad"));
  else badges.push(chip(`Citas verificadas: apuntan a elementos reales (${r.cited_kinds.map((k) => KIND_TEXT[k]).join(" y ")})`, "ok"));
  if (r.direction) badges.push(chip(`Recorrido: ${{ reverse: "al revés", forward: "hacia adelante", both: "ambos sentidos" }[r.direction]}`));
  $("badges").replaceChildren(...badges);
  $("usage").textContent = r.usage
    ? `${r.usage.provider} · ${r.usage.model} · ${r.usage.tokens_in} tokens de entrada, ${r.usage.tokens_out} de salida · costo total de la pregunta (con descomponer o mapa si se usaron): ${money(r.cost_usd)} · ${(r.timings_ms.total / 1000).toFixed(1)} s en total`
    : "No se llamó al modelo (US$ 0).";

  $("stages").replaceChildren(...renderStages(r.stages));
  renderItems(r.items);
  renderFacts(r.items);
  renderSql(r.sql || []);
  renderSubgraph(r);
  $("prompt").textContent = r.prompt || "(No se armó un prompt porque no se llamó al modelo.)";
  const names = { link: "Enlazar entidades", traverse: "Recorrer el grafo", facts: "Hechos", origin: "Fragmentos de origen", vector: "Búsqueda vectorial", decompose: "Descomposición (LLM)", rrf: "Fusión RRF", merge: "Contexto combinado", map: "Mapa (LLM por comunidad)", generate: "Generar con el modelo", total: "Total" };
  $("timings").replaceChildren(h("table", {}, h("tbody", {}, ...Object.entries(r.timings_ms).map(([k, v]) =>
    h("tr", {}, h("td", {}, names[k] || k), h("td", {}, `${v} ms`))))));
}

let lastSubgraph = null;
async function renderSubgraph(r) {
  lastSubgraph = r;
  const sub = r.subgraph;
  $("subgraph-card").hidden = !sub;
  if (!sub) return;
  const source = r.graph || $("graph-source").value;
  let full = null;
  if ($("show-full").checked) full = await getFullGraph(source);
  const hlNodes = new Set(sub.nodes.map((n) => n.id));
  const roles = Object.fromEntries(sub.nodes.map((n) => [n.id, n.role]));
  const edgeKeys = new Set(sub.edges.map((e) => `${e.subject}|${e.relation}|${e.object}`));
  const depth = Object.fromEntries(sub.edges.map((e) => [`${e.subject}|${e.relation}|${e.object}`, e.depth]));
  const nodes = full ? full.nodes.map((n) => ({ id: n.id, type: n.type, community: n.community })) : sub.nodes.map((n) => ({ id: n.id, type: n.type }));
  const edges = full ? full.edges : sub.edges;
  const teamsShown = sub.nodes.some((n) => n.type === "equipo");
  drawGraph($("subgraph"), {
    nodes, edges, prefix: "sub", showTeams: teamsShown, colorBy: "type", communities: [],
    isHl: (id) => hlNodes.has(id), role: (id) => roles[id], edgeHl: (key) => edgeKeys.has(key), depth: (key) => depth[key],
  });
  $("subgraph-note").textContent = `${sub.nodes.length} entidades y ${sub.edges.length} relaciones recorridas. El borde grueso marca la entidad de partida; los números sobre las aristas son los saltos desde ella. Pulsa una cita de un hecho en la respuesta para resaltar su arista.`;
  $("graph-legend").replaceChildren(...legend("type", [], true));
}
$("show-full").addEventListener("change", () => lastSubgraph && renderSubgraph(lastSubgraph));

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ask-btn"), async () => {
    renderResult(await api("/api/ask", {
      method: "POST",
      body: {
        question: $("question").value.trim(), mode: $("mode").value, k: Number($("k").value), hops: Number($("hops").value),
        direction: $("direction").value, graph: $("graph-source").value, decompose: $("decompose").checked, contextual: $("contextual").checked,
      },
    }));
  }, "Pensando…");
});

// ───────────── dibujar un grafo en SVG (sin librerías) ─────────────
const DEP_RELATIONS = ["depende_de", "corre_en"];
const TYPE_COLOR = { infraestructura: 8, datos: 1, seguridad: 4, plataforma: 3, externo: 2, negocio: 5, soporte: 6, equipo: 7, otro: 1 };
const TYPE_NAME = { infraestructura: "Infraestructura", datos: "Datos", seguridad: "Seguridad", plataforma: "Plataforma", externo: "Externo", negocio: "Negocio", soporte: "Soporte", equipo: "Equipo" };
const colorVar = (n) => `var(--c${((n - 1) % 8) + 1})`;

/** Capas: un servicio queda debajo de todos los que dependen de él (camino más largo desde quien nadie usa). Orden dentro
 * de cada capa por el baricentro de sus vecinos, para cruzar menos aristas. Tolera ciclos (el relleno se corta). */
function layout(nodes, edges, width) {
  const ids = nodes.map((n) => n.id);
  const layer = Object.fromEntries(ids.map((id) => [id, 0]));
  const deps = edges.filter((e) => DEP_RELATIONS.includes(e.relation) && e.subject in layer && e.object in layer);
  for (let pass = 0; pass < ids.length; pass++) {
    let changed = false;
    for (const e of deps) {
      if (layer[e.object] < layer[e.subject] + 1 && layer[e.subject] + 1 < ids.length) { layer[e.object] = layer[e.subject] + 1; changed = true; }
    }
    if (!changed) break;
  }
  const teams = nodes.filter((n) => n.type === "equipo").map((n) => n.id);
  let maxService = Math.max(0, ...ids.filter((id) => !teams.includes(id)).map((id) => layer[id]));
  // la infraestructura (no depende de nada) va abajo del todo, aunque su camino más largo sea corto
  const hasOut = new Set(deps.map((e) => e.subject));
  for (const id of ids) if (!teams.includes(id) && !hasOut.has(id) && deps.some((e) => e.object === id)) layer[id] = maxService;
  for (const t of teams) layer[t] = maxService + 1;
  const rows = [];
  for (const id of ids) (rows[layer[id]] ||= []).push(id);
  const order = rows.map((row) => (row || []).sort());
  const pos = {};
  const place = () => order.forEach((row) => row.forEach((id, i) => (pos[id] = (i + 0.5) / row.length)));
  place();
  const neighbours = (id) => edges.flatMap((e) => (e.subject === id && e.object in pos ? [e.object] : e.object === id && e.subject in pos ? [e.subject] : []));
  for (let sweep = 0; sweep < 6; sweep++) {
    const sequence = sweep % 2 === 0 ? order.keys() : [...order.keys()].reverse();
    for (const r of sequence) {
      const row = order[r];
      const bary = (id) => { const ns = neighbours(id).filter((n) => layer[n] !== r); return ns.length ? ns.reduce((a, n) => a + pos[n], 0) / ns.length : pos[id]; };
      row.sort((a, b) => bary(a) - bary(b) || a.localeCompare(b));
      row.forEach((id, i) => (pos[id] = (i + 0.5) / row.length));
    }
  }
  return { order, layer };
}

function wrapName(name, max) {
  const words = name.split(" ");
  const lines = [""];
  for (const w of words) {
    const cur = lines[lines.length - 1];
    if (cur && (cur + " " + w).length > max && lines.length < 2) lines.push(w);
    else lines[lines.length - 1] = cur ? `${cur} ${w}` : w;
  }
  return lines;
}

/** Dibuja `nodes`/`edges` en `container`. opciones: prefix (ids únicos), showTeams, colorBy ("type" | "community"), isHl, role, edgeHl, depth. */
function drawGraph(container, opts) {
  const { prefix, showTeams, colorBy } = opts;
  const nodes = opts.nodes.filter((n) => showTeams || n.type !== "equipo");
  const present = new Set(nodes.map((n) => n.id));
  const edges = opts.edges.filter((e) => present.has(e.subject) && present.has(e.object));
  if (!nodes.length) { container.replaceChildren(h("p", { class: "muted", style: "padding:12px" }, "No hay nodos que mostrar.")); return; }
  const W = 980, GAP = 92, TOP = 28, BOX_H = 38;
  const { order, layer } = layout(nodes, edges, W);
  const widest = Math.max(...order.map((r) => (r ? r.length : 0)));
  const boxW = Math.max(84, Math.min(158, W / widest - 14));
  const rowsUsed = order.length;
  const height = TOP * 2 + (rowsUsed - 1) * GAP + BOX_H;
  const where = {};
  order.forEach((row, r) => (row || []).forEach((id, i) => (where[id] = { x: ((i + 0.5) / row.length) * W, y: TOP + r * GAP + BOX_H / 2 })));
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const stripe = (n) => (colorBy === "community" && n.community ? colorVar(n.community) : colorVar(TYPE_COLOR[n.type] || 1));

  const svg = s("svg", { viewBox: `0 0 ${W} ${height}`, role: "img", "aria-label": "Grafo de servicios: la flecha A → B significa «A depende de B»" });
  svg.append(s("defs", {},
    s("marker", { id: `${prefix}-arrow`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "g-arrow" })),
    s("marker", { id: `${prefix}-arrow-hl`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "g-arrow hl" }))));

  const edgeLayer = s("g", {});
  const badgeLayer = s("g", {});
  const hlEdge = (e) => (opts.edgeHl ? opts.edgeHl(`${e.subject}|${e.relation}|${e.object}`) : false);
  for (const e of edges) {
    const a = where[e.subject], b = where[e.object];
    if (!a || !b) continue;
    const key = `${e.subject}|${e.relation}|${e.object}`;
    const hl = opts.edgeHl ? hlEdge(e) : false;
    const dim = opts.edgeHl && !hl;
    const dy = b.y >= a.y ? BOX_H / 2 : -BOX_H / 2;
    const x1 = a.x, y1 = a.y + dy, x2 = b.x, y2 = b.y - dy;
    const path = s("path", { d: `M${x1},${y1} L${x2},${y2}`, class: `g-edge ${e.relation}${hl ? " hl" : ""}${dim ? " dim" : ""}`, "data-key": key,
      "marker-end": `url(#${prefix}-arrow${hl ? "-hl" : ""})` }, s("title", {}, `${e.subject} ${e.relation.replaceAll("_", " ")} ${e.object}`));
    edgeLayer.append(path);
    const d = opts.depth && hl ? opts.depth(key) : null;
    if (d) badgeLayer.append(s("g", { class: "g-depth", transform: `translate(${(x1 + x2) / 2},${(y1 + y2) / 2})` }, s("circle", { r: 8 }), s("text", {}, String(d))));
  }
  svg.append(edgeLayer, badgeLayer);

  for (const n of nodes) {
    const p = where[n.id];
    const role = opts.role ? opts.role(n.id) : null;
    const dim = opts.isHl && !opts.isHl(n.id);
    const lines = wrapName(n.id, Math.floor(boxW / 6.3));
    const g = s("g", { class: `g-node${role ? " " + role : ""}${dim ? " dim" : ""}${n.type === "equipo" ? " team" : ""}`, transform: `translate(${p.x - boxW / 2},${p.y - BOX_H / 2})` },
      s("title", {}, `${n.id} (${n.type})`),
      s("rect", { width: boxW, height: BOX_H, rx: 7 }),
      s("rect", { width: 7, height: BOX_H, rx: 3, style: `fill:${stripe(n)};stroke:none` }),
      ...lines.map((line, i) => s("text", { x: 14, y: lines.length === 1 ? BOX_H / 2 + 4 : 15 + i * 13 }, line)));
    svg.append(g);
  }
  container.replaceChildren(svg);
}

function legend(colorBy, communities, withPath) {
  const items = [];
  if (colorBy === "community") {
    for (const c of communities) items.push(h("span", {}, h("i", { style: `background:${colorVar(c.id)}` }), `Comunidad ${c.id}`));
  } else {
    for (const [type, name] of Object.entries(TYPE_NAME)) items.push(h("span", {}, h("i", { style: `background:${colorVar(TYPE_COLOR[type])}` }), name));
  }
  items.push(h("span", {}, h("i", { class: "line" }), "depende de"), h("span", {}, h("i", { class: "line dashed" }), "se ejecuta en"), h("span", {}, h("i", { class: "line dotted" }), "es administrado por"));
  if (withPath) items.push(h("span", {}, h("i", { style: "background:var(--edge-hl);border-color:var(--edge-hl)" }), "camino recorrido"));
  items.push(h("span", {}, "Flecha A → B: «A depende de B»"));
  return items;
}

async function getFullGraph(source) {
  if (!fullGraphs[source]) {
    const g = await api(`/api/graph?source=${source}`);
    if (!g.built) throw new Error(`El grafo «${source}» no está construido (pestaña Grafo).`);
    fullGraphs[source] = g;
  }
  return fullGraphs[source];
}

// ───────────── grafo ─────────────
let viewGraph = null;
function drawFull() {
  if (!viewGraph) return;
  const colorBy = $("color-by").value;
  drawGraph($("full-graph"), { nodes: viewGraph.nodes, edges: viewGraph.edges, prefix: "full", showTeams: $("show-teams").checked, colorBy });
  $("full-legend").replaceChildren(...legend(colorBy, viewGraph.communities, false));
}
["color-by", "show-teams"].forEach((id) => $(id).addEventListener("change", drawFull));
$("view-source").addEventListener("change", loadGraphView);

async function loadGraphView() {
  const source = $("view-source").value;
  showError("");
  try {
    viewGraph = await api(`/api/graph?source=${source}`);
    if (!viewGraph.built) throw new Error("sin construir");
    fullGraphs[source] = viewGraph;
  } catch (error) {
    viewGraph = null;
    $("full-graph").replaceChildren();
    $("full-legend").replaceChildren();
    $("graph-stats").replaceChildren();
    $("communities").replaceChildren();
    $("graph-empty").hidden = false;
    return;
  }
  $("graph-empty").hidden = true;
  const st = viewGraph.stats;
  $("graph-stats").replaceChildren(
    stat(st.nodes, "nodos (entidades)"), stat(st.edges, "aristas (tripletas)"), stat(st.avg_degree, "grado promedio"),
    stat(st.components, "componentes conectados"), stat(st.has_cycles ? "sí" : "no", "ciclos en las dependencias"),
    ...Object.entries(st.by_relation).map(([rel, n]) => stat(n, rel.replaceAll("_", " "))),
    h("div", { class: "stat" }, h("b", { class: "small-b" }, st.most_depended_on[0] ? `${st.most_depended_on[0].name} (${st.most_depended_on[0].degree})` : "—"), h("span", {}, "el servicio del que más dependen (directo)")));
  drawFull();
  $("communities").replaceChildren(...viewGraph.communities.map((c) => h("div", { class: "community", style: `border-left-color:${colorVar(c.id)}` },
    h("h3", {}, c.title),
    h("div", { class: "members" }, ...c.members.map((m) => chip(m))),
    c.summary ? h("p", {}, c.summary) : h("p", { class: "muted" }, "Sin resumen todavía: pulsa «Resumir comunidades»."))));
}

function renderBuild(report) {
  const box = $("build-result");
  const nodes = [h("p", {}, chip(`Grafo ${report.source === "curated" ? "curado" : "extraído"}`, "ok"), ` ${report.entities} entidades, ${report.triples} tripletas, ${report.communities} comunidades, ${report.seconds} s.`),
    ...report.warnings.map((w) => chip(w, "warn"))];
  const e = report.extraction;
  if (e) {
    const row = (label, m) => h("tr", {}, h("td", {}, label), h("td", { class: "num" }, String(m.extracted)), h("td", { class: "num" }, `${m.correct}/${m.curated}`),
      h("td", { class: "num" }, m.precision.toFixed(2)), h("td", { class: "num" }, m.recall.toFixed(2)), h("td", { class: "num" }, m.f1.toFixed(2)));
    nodes.push(
      h("p", { class: "muted small" }, `Prompt ${e.prompt_version}: ${e.calls} llamadas al LLM (${e.cache_hits} salieron de la caché) sobre ${e.sections} secciones · ${e.tokens_in}→${e.tokens_out} tokens · ${money(e.cost_usd)} · ${e.llm_seconds} s de LLM · ${e.raw_triples} tripletas crudas.`),
      h("div", { class: "table-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, th("Medido contra el grafo curado"), th("Tripletas", "num"), th("Correctas", "num"), th("Precisión", "num"), th("Recall", "num"), th("F1", "num"))),
        h("tbody", {}, row("Solo normalizado (sin validar)", e.metrics_without_validation), row("Normalizado y validado (el grafo guardado)", e.metrics)))),
      h("p", { class: "muted small" }, "Precisión: de lo que se extrajo, cuánto es correcto. Recall: de lo curado, cuánto se encontró. Una arista falsa contamina todas las respuestas que pasan por ella."));
    if (e.rejected.length) nodes.push(h("details", {}, h("summary", {}, `${e.rejected.length} tripletas rechazadas por la validación`),
      h("ul", { class: "reject-list" }, ...e.rejected.map((r) => h("li", {}, `[${r.section}] ${r.triple.join(" → ")}: ${r.reason}`)))));
    if (e.failed_sections.length) nodes.push(chip(`Secciones sin JSON válido: ${e.failed_sections.join(", ")}`, "bad"));
    const miss = e.metrics.missing.length, extra = e.metrics.extra.length;
    if (miss || extra) nodes.push(h("details", {}, h("summary", {}, `Faltan ${miss} tripletas curadas · sobran ${extra} no curadas`),
      h("ul", { class: "reject-list" }, ...e.metrics.missing.map((t) => h("li", {}, `falta: ${t.join(" → ")}`)), ...e.metrics.extra.map((t) => h("li", {}, `sobra: ${t.join(" → ")}`)))));
  }
  box.replaceChildren(...nodes);
}

$("build-curated").addEventListener("click", () => guarded($("build-curated"), async () => {
  renderBuild(await api("/api/graph/build", { method: "POST", body: { source: "curated" } }));
  $("view-source").value = "curated";
  delete fullGraphs.curated;
  await Promise.all([refreshStatus(), loadGraphView()]);
}, "Construyendo…"));

$("build-extracted").addEventListener("click", () => guarded($("build-extracted"), async () => {
  renderBuild(await api("/api/graph/build", { method: "POST", body: { source: "extracted", prompt_version: $("extract-version").value, force: $("extract-force").checked } }));
  $("view-source").value = "extracted";
  delete fullGraphs.extracted;
  await Promise.all([refreshStatus(), loadGraphView()]);
}, "Extrayendo con el LLM…"));

$("summarize").addEventListener("click", () => guarded($("summarize"), async () => {
  const r = await api("/api/graph/summarize", { method: "POST", body: { source: $("view-source").value } });
  $("build-result").replaceChildren(chip(`Resúmenes listos: ${r.communities} comunidades, ${r.calls} llamadas al LLM (${r.cache_hits} de la caché), ${money(r.cost_usd)}, ${r.llm_seconds} s`, "ok"));
  delete fullGraphs[$("view-source").value];
  await Promise.all([refreshStatus(), loadGraphView()]);
}, "Resumiendo…"));

// ───────────── ingesta ─────────────
$("strategy").addEventListener("change", () => {
  $("overlap").disabled = $("strategy").value === "section";
  $("max-words").disabled = $("strategy").value === "section";
});

$("ingest-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ingest-btn"), async () => {
    const r = await api("/api/ingest", {
      method: "POST",
      body: {
        strategy: $("strategy").value, max_words: Number($("max-words").value), overlap: Number($("overlap").value),
        heading: $("heading").checked, contextual: $("contextual-ingest").checked, force_context: $("force-context").checked,
      },
    });
    const box = $("ingest-result");
    box.hidden = false;
    box.replaceChildren(
      h("h2", {}, "Ingesta lista"),
      h("p", {}, `${r.documents} documento → ${r.chunks} fragmentos de ${r.dimension} dimensiones en ${r.store === "postgres" ? "PostgreSQL (pgvector)" : "memoria"}, ${r.seconds.toFixed(1)} s.`),
      r.contextual
        ? h("p", { class: "muted small" }, `Contextual: ${r.context_calls} llamadas al LLM, ${r.context_cache_hits} frases salieron de la caché, ${money(r.cost_usd)}. Cada fragmento tiene ahora dos vectores: con y sin la frase.`)
        : h("p", { class: "muted small" }, "Sin frases de contexto: los modos que usan «vectores con frase de contexto» no estarán disponibles hasta reingestar con «Contextual»."),
      h("div", { class: "table-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, th("Fragmento"), th("Sección"), th("Texto"), th("Frase de contexto (LLM)"))),
        h("tbody", {}, ...r.preview.map((p) => h("tr", {}, h("td", {}, p.id), h("td", {}, p.section), h("td", {}, p.text), h("td", {}, p.context || "—")))))),
      h("p", { class: "muted small" }, `Se muestran los primeros ${r.preview.length} de ${r.chunks}.`));
    await refreshStatus();
  }, "Ingestando…");
});

// ───────────── evaluación ─────────────
let evalConfigs = null;
async function loadEvalConfigs() {
  if (evalConfigs) return;
  evalConfigs = await api("/api/evaluate/configs");
  $("eval-configs").replaceChildren(...evalConfigs.map((c) => h("label", {}, h("input", { type: "checkbox", value: c.name, checked: true }), c.label, c.needs_llm_in_retrieval ? h("span", { class: "llm" }, "(usa LLM)") : null)));
}
const chosenConfigs = () => [...document.querySelectorAll("#eval-configs input:checked")].map((i) => i.value);

function bestIndex(values, lowerIsBetter = false) {
  const nums = values.filter((v) => v != null);
  if (!nums.length) return null;
  return lowerIsBetter ? Math.min(...nums) : Math.max(...nums);
}

function renderEval(data) {
  const { summary: sm, rows } = data;
  const cfgs = Object.entries(sm.configs);
  const gen = sm.generate;
  const best = (key) => bestIndex(cfgs.map(([, c]) => c[key]));
  const cell = (value, text, best_) => h("td", { class: `num${best_ != null && value === best_ ? " best" : ""}` }, text);

  const head = ["Configuración", "Contexto completo", "Cobertura del contexto", "Palabras al modelo", ...(gen ? ["Recall de la respuesta", "Precisión", "Completas", "Citas válidas"] : []), "Costo", "Latencia"];
  const summaryTable = h("table", {}, h("thead", {}, h("tr", {}, ...head.map((x, i) => th(x, i ? "num" : "")))),
    h("tbody", {}, ...cfgs.map(([, c]) => h("tr", {},
      h("td", {}, c.label),
      cell(c.context_full, `${c.context_full}/${c.n}`, best("context_full")),
      cell(c.context_recall, pct(c.context_recall), best("context_recall")),
      h("td", { class: "num" }, String(c.context_words)),
      ...(gen ? [cell(c.answer_recall, pct(c.answer_recall), best("answer_recall")), cell(c.answer_precision, pct(c.answer_precision), best("answer_precision")),
        cell(c.complete, `${c.complete}/${c.n}`, best("complete")), cell(c.valid_citations, `${c.valid_citations}/${c.n}`, best("valid_citations"))] : []),
      h("td", { class: "num" }, money(c.cost_usd)), h("td", { class: "num" }, `${c.avg_ms} ms`)))));

  const types = Object.keys(sm.types);
  const typeTable = h("table", {}, h("thead", {}, h("tr", {}, th("Configuración"), ...types.map((t) => th(`${TYPE_LABELS[t] || t} (${sm.types[t]})`, "num")))),
    h("tbody", {}, ...cfgs.map(([, c]) => h("tr", {}, h("td", {}, c.label), ...types.map((t) => {
      const b = c.by_type[t];
      if (!b) return h("td", { class: "num" }, "—");
      const bestType = bestIndex(cfgs.map(([, o]) => (o.by_type[t] ? (gen ? o.by_type[t].answer_recall : o.by_type[t].context_recall) : null)));
      const main = gen ? b.answer_recall : b.context_recall;
      return h("td", { class: `num${main === bestType ? " best" : ""}`, title: gen ? `contexto ${pct(b.context_recall)} · respuesta ${pct(b.answer_recall)}` : "cobertura del contexto" },
        gen ? `${pct(b.context_recall)} → ${pct(b.answer_recall)}` : pct(b.context_recall));
    })))));

  const detail = h("table", {}, h("thead", {}, h("tr", {}, th("Pregunta"), th("Tipo"), ...cfgs.map(([, c]) => th(c.label, "num")))),
    h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, h("b", {}, r.id), ` ${r.question}`), h("td", {}, r.type),
      ...cfgs.map(([name]) => {
        const x = r.results[name];
        const miss = [...x.context_missing, ...(x.answer_missing || [])];
        return h("td", { class: `num${x.context_recall < 1 ? " bad" : ""}`, title: x.context_missing.length ? `Faltan en el contexto: ${x.context_missing.join(", ")}` : "El contexto tiene todo" },
          gen ? `${pct(x.context_recall)} → ${pct(x.answer_recall)}` : pct(x.context_recall));
      })))));

  $("eval-result").hidden = false;
  $("eval-result").replaceChildren(
    h("div", { class: "stats" }, stat(sm.n, "preguntas"), stat(cfgs.length, "configuraciones medidas"), stat(money(sm.cost_usd), "costo de la evaluación"), stat(sm.model.split(" · ")[1] || sm.model, gen ? "modelo que respondió" : "modelo (no se usó para responder)")),
    h("div", { class: "alert-soft card" }, h("strong", {}, "Ojo con el ruido. "), sm.warning, " Los resultados dependen del modelo de embeddings y del modelo de lenguaje: ", h("code", {}, sm.embeddings), " · ", h("code", {}, sm.model), "."),
    h("div", { class: "card table-scroll" }, h("h2", {}, "Comparación"), summaryTable,
      h("p", { class: "muted small" }, "Verde = mejor valor de la columna. Cobertura del contexto: fracción de las entidades esperadas que llegaron al prompt (sin contar las que ya nombraba la pregunta). Recall de la respuesta: las que el modelo nombró. Precisión: cuánto de lo que nombró era esperado. Completas: recall y precisión de 100%. En «Grafo global» los resúmenes enumeran a sus miembros, y por eso la cobertura sale alta aunque el contexto sea poco preciso: mira la precisión.")),
    h("div", { class: "card table-scroll" }, h("h2", {}, `Por tipo de pregunta ${gen ? "(contexto → respuesta)" : "(cobertura del contexto)"}`), typeTable,
      h("p", { class: "muted small" }, "Cada técnica ataca un tipo distinto. La pregunta «directa» es el control: ahí los vectores bastan y el grafo no debería aportar.")),
    h("details", { class: "card" }, h("summary", {}, "Detalle por pregunta"), h("div", { class: "table-scroll" }, detail)),
    Object.keys(sm.skipped).length ? h("div", { class: "card" }, h("h2", {}, "No se pudieron medir"), h("ul", { class: "reject-list" }, ...Object.entries(sm.skipped).map(([k, why]) => h("li", {}, `${k}: ${why}`)))) : null);
}

async function runEval(button, generate) {
  await guarded(button, async () => {
    const all = chosenConfigs();
    const free = new Set(evalConfigs.filter((c) => !c.needs_llm_in_retrieval).map((c) => c.name));
    const configs = generate ? all : all.filter((c) => free.has(c));
    if (!configs.length) throw new Error("Elige al menos una configuración" + (generate ? "." : " que no use el LLM."));
    renderEval(await api("/api/evaluate", { method: "POST", body: { configs, generate } }));
  }, "Evaluando…");
}
$("eval-free").addEventListener("click", () => runEval($("eval-free"), false));
$("eval-full").addEventListener("click", () => runEval($("eval-full"), true));

// ───────────── documentos ─────────────
let docsLoaded = false;
async function loadDocs() {
  if (docsLoaded) return;
  const docs = await api("/api/documents");
  $("docs").replaceChildren(h("h2", {}, "Documento de origen"),
    h("p", { class: "muted" }, "El catálogo de servicios de TI de Andes Fibra (ficticio). Cada sección dice de qué depende un servicio, dónde se ejecuta y quién lo administra, solo desde su propio punto de vista: nadie escribió «de la base de datos dependen X, Y y Z». Eso hay que construirlo."),
    ...docs.map((d) => h("div", { class: "doc" }, h("h3", {}, d.source),
      ...d.sections.map((sec) => h("details", {}, h("summary", {}, `${sec.title} · ${sec.words} palabras`), h("p", {}, sec.text))))));
  docsLoaded = true;
}

// ───────────── arranque ─────────────
(async () => {
  updateControls();
  try {
    await Promise.all([refreshStatus(), loadExamples()]);
  } catch (error) {
    showError(error.message);
  }
})();
