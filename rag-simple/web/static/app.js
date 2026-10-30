"use strict";

// ───────────── utilidades ─────────────
const $ = (id) => document.getElementById(id);

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

async function guarded(button, task) {
  showError("");
  button.disabled = true;
  try {
    await task();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
  }
}

const money = (usd) => (usd === 0 ? "US$ 0" : `US$ ${usd.toFixed(usd < 0.01 ? 6 : 4)}`);
const chip = (text, kind = "") => h("span", { class: `chip ${kind}` }, text);

// ───────────── pestañas ─────────────
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "docs") loadDocs();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ───────────── estado del sistema ─────────────
async function refreshStatus() {
  const s = await api("/api/status");
  const chips = $("chips");
  chips.replaceChildren(
    chip(`Modelo: ${s.llm_provider} · ${s.llm_model}`),
    chip(`Embeddings: ${s.embeddings_provider}`),
    chip(`Almacén: ${s.store}`),
    chip(s.chunks ? `${s.chunks} fragmentos indexados` : "Sin ingestar", s.chunks ? "ok" : "warn"),
  );
  if (s.store_error) showError(`Almacén: ${s.store_error}`);
  return s;
}

// ───────────── preguntar ─────────────
async function loadExamples() {
  const examples = await api("/api/examples");
  $("examples").replaceChildren(
    ...examples.map((e) =>
      h("button", {
        type: "button", class: `example${e.out_of_scope ? " out" : ""}`,
        title: e.out_of_scope ? "Fuera de alcance: los documentos no la contestan" : "",
        onclick: () => { $("question").value = e.question; $("question").focus(); },
      }, e.out_of_scope ? `${e.question} (fuera de alcance)` : e.question),
    ),
  );
}

/** Convierte "…cuesta 39 [1]" en texto + botones [1] que llevan al fragmento citado. Marca en rojo las citas inexistentes. */
function renderAnswer(text, hitCount) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    nodes.push(text.slice(last, match.index));
    for (const n of match[1].split(/\s*,\s*/).map(Number)) {
      const valid = n >= 1 && n <= hitCount;
      nodes.push(h("button", {
        type: "button", class: `cite${valid ? "" : " invalid"}`,
        title: valid ? `Ir al fragmento ${n}` : `El fragmento ${n} no existe`,
        onclick: () => valid && focusHit(n),
      }, String(n)));
    }
    last = match.index + match[0].length;
  }
  nodes.push(text.slice(last));
  return nodes;
}

function focusHit(n) {
  const el = $(`hit-${n}`);
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

function renderResult(r) {
  $("result").hidden = false;
  $("answer-text").replaceChildren(...renderAnswer(r.answer, r.hits.length));

  const badges = [];
  if (r.skipped_reason) badges.push(chip(`Sin llamar al modelo: ${r.skipped_reason}`, "warn"));
  else if (r.abstained) badges.push(chip("El modelo dijo que no lo encuentra", "warn"));
  else if (r.invalid_citations.length) badges.push(chip(`Citas inexistentes: ${r.invalid_citations.join(", ")} (posible invención)`, "bad"));
  else if (r.uncited) badges.push(chip("Respondió sin citar: no se puede verificar", "bad"));
  else badges.push(chip("Citas verificadas: todas apuntan a fragmentos reales", "ok"));
  $("badges").replaceChildren(...badges);

  $("usage").textContent = r.usage
    ? `${r.usage.provider} · ${r.usage.model} · ${r.usage.tokens_in} tokens de entrada, ${r.usage.tokens_out} de salida · ${money(r.usage.cost_usd)} · ${r.usage.seconds.toFixed(1)} s`
    : "No se llamó al modelo (US$ 0).";

  $("hits").replaceChildren(...r.hits.map((hit) =>
    h("div", { class: `hit${hit.cited ? " cited" : ""}`, id: `hit-${hit.number}` },
      h("div", { class: "hit-head" },
        h("span", { class: "num" }, String(hit.number)),
        h("span", { class: "hit-source" }, `${hit.source} › ${hit.section}`),
        hit.cited ? chip("citado", "ok") : null,
        h("span", { class: "score", title: "Similitud coseno: 1 = idéntico" },
          h("span", { class: "bar" }, h("i", { style: `width:${Math.max(0, Math.min(1, hit.score)) * 100}%` })),
          hit.score.toFixed(3))),
      h("p", {}, hit.text))));

  $("prompt").textContent = r.prompt || "(No se armó un prompt porque no se llamó al modelo.)";
  const t = r.timings_ms;
  $("timings").replaceChildren(h("table", {}, h("tbody", {}, ...Object.entries(t).map(([k, v]) =>
    h("tr", {}, h("td", {}, { embed_query: "Vectorizar la pregunta", search: "Buscar en el almacén", generate: "Generar con el modelo", total: "Total" }[k] || k), h("td", {}, `${v} ms`))))));
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ask-btn"), async () => {
    const threshold = $("threshold").value;
    renderResult(await api("/api/ask", {
      method: "POST",
      body: { question: $("question").value.trim(), k: Number($("k").value), threshold: threshold === "" ? null : Number(threshold) },
    }));
  });
});

// ───────────── ingesta ─────────────
// El solape no significa lo mismo en cada estrategia: se dice en la etiqueta y se desactiva cuando no aplica.
const OVERLAP_UNITS = {"sentences": "oraciones", "fixed": "× 10 palabras", "section": null};   // null = la estrategia no usa solape
function updateOverlapLabel() {
  const unit = OVERLAP_UNITS[$("strategy").value];
  $("overlap-label").textContent = unit ? `Solape (${unit})` : "Solape (no aplica)";
  $("overlap").disabled = !unit;
}
$("strategy").addEventListener("change", updateOverlapLabel);
updateOverlapLabel();

$("ingest-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ingest-btn"), async () => {
    const r = await api("/api/ingest", {
      method: "POST",
      body: { strategy: $("strategy").value, max_words: Number($("max-words").value), overlap: Number($("overlap").value) },
    });
    const box = $("ingest-result");
    box.hidden = false;
    box.replaceChildren(
      h("h2", {}, "Ingesta lista"),
      h("p", {}, `${r.documents} documentos → ${r.chunks} fragmentos de ${r.dimension} dimensiones, guardados en ${r.store} en ${r.seconds.toFixed(1)} s.`),
      h("table", {}, h("tbody", {}, ...Object.entries(r.per_document).map(([name, n]) => h("tr", {}, h("td", {}, name), h("td", {}, `${n} fragmentos`))))),
    );
    await refreshStatus();
  });
});

// ───────────── evaluación ─────────────
function renderEval(data) {
  const { summary: s, rows } = data;
  const stats = [
    stat(`${s.retrieval.hit1}/${s.in_scope}`, "hit@1: la sección correcta quedó primera"),
    stat(`${s.retrieval.hitk}/${s.in_scope}`, `hit@${s.k}: está entre los ${s.k} primeros`),
    stat(s.retrieval.mrr.toFixed(2), "MRR: 1 = siempre primera"),
  ];
  if (s.generation) {
    stats.push(
      stat(`${s.generation.fact}/${s.in_scope}`, "Dato clave en la respuesta"),
      stat(`${s.generation.valid_citations}/${s.in_scope}`, "Citas válidas"),
      stat(`${s.generation.abstain_ok}/${s.out_of_scope}`, "“No sé” bien dicho"),
      stat(money(s.generation.cost_usd), "Costo de la evaluación"),
    );
  }
  const head = ["Pregunta", "Esperada", "Posición", "Mejores"];
  if (s.generation) head.push("Respuesta");
  const body = rows.map((r) => {
    const cells = r.out_of_scope
      ? [r.question, "Fuera de alcance", "—", r.top_score == null ? "—" : `mejor puntaje ${r.top_score.toFixed(2)}`]
      : [r.question, r.expected, r.position ? String(r.position) : "no apareció", r.got.join(" · ")];
    if (s.generation) cells.push(`${r.answer || ""} ${r.out_of_scope ? (r.abstained_ok ? "✔" : "✘") : (r.fact_ok && r.citations_ok ? "✔" : "✘")}`);
    return h("tr", {}, ...cells.map((c, i) => h("td", i === 2 && c === "no apareció" ? { class: "bad" } : {}, c)));
  });
  $("eval-result").hidden = false;
  $("eval-result").replaceChildren(
    h("div", { class: "stats" }, ...stats),
    h("div", { class: "card table-scroll" },
      h("table", {}, h("thead", {}, h("tr", {}, ...head.map((x) => h("th", {}, x)))), h("tbody", {}, ...body)),
      h("p", { class: "muted small" }, "Las preguntas fuera de alcance muestran el puntaje de su mejor fragmento: si es parecido al de una pregunta con respuesta, un umbral de similitud no las detecta; eso lo resuelve el prompt (“no encuentro”)."),
    ),
  );
}
const stat = (value, label) => h("div", { class: "stat" }, h("b", {}, value), h("span", {}, label));

for (const [id, withGeneration] of [["eval-retrieval", false], ["eval-full", true]]) {
  $(id).addEventListener("click", () => guarded($(id), async () => {
    $(id).textContent = "Evaluando…";
    try {
      renderEval(await api("/api/evaluate", { method: "POST", body: { k: 4, with_generation: withGeneration } }));
    } finally {
      $(id).textContent = withGeneration ? "Evaluar también la generación (cuesta)" : "Evaluar recuperación (gratis)";
    }
  }));
}

// ───────────── documentos ─────────────
let docsLoaded = false;
async function loadDocs() {
  if (docsLoaded) return;
  const docs = await api("/api/documents");
  $("docs").replaceChildren(h("h2", {}, "Documentos de origen"), h("p", { class: "muted" }, "Estos son los archivos Markdown que se fragmentan. Cada encabezado “##” es una sección."),
    ...docs.map((d) => h("div", { class: "doc" }, h("h3", {}, d.source),
      ...d.sections.map((sec) => h("details", {}, h("summary", {}, `${sec.title} · ${sec.words} palabras`), h("p", {}, sec.text))))));
  docsLoaded = true;
}

// ───────────── arranque ─────────────
(async () => {
  try {
    await Promise.all([refreshStatus(), loadExamples()]);
  } catch (error) {
    showError(error.message);
  }
})();
