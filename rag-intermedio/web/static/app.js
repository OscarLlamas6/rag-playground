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
const audienceTag = (a) => h("span", { class: `aud ${a}` }, a);
const mmss = (s) => { const t = Math.max(0, Math.floor(s)); const hh = Math.floor(t / 3600); const mm = Math.floor((t % 3600) / 60); const ss = t % 60;
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`; };
const short = (text, n = 90) => (text.length > n ? `${text.slice(0, n).trim()}…` : text);

// ───────────── pestañas ─────────────
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "docs") loadDocs();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ───────────── estado ─────────────
async function refreshStatus() {
  const s = await api("/api/status");
  $("chips").replaceChildren(
    chip(`Modelo: ${s.llm_provider} · ${s.llm_model}`),
    chip(`Embeddings: ${s.embeddings_provider}`),
    chip(s.index ? `Pinecone: ${s.index} / ${s.namespace}` : `Almacén: ${s.store}`),
    chip(s.chunks ? `${s.chunks} fragmentos indexados` : "Sin ingestar", s.chunks ? "ok" : "warn"),
  );
  if (s.store_error) showError(`Almacén: ${s.store_error}`);
  const m = s.media;
  $("media-status").replaceChildren(
    chip(m.docling ? "Docling (PDF) instalado" : "PDF: falta el grupo media", m.docling ? "ok" : "warn"),
    chip(m.faster_whisper && m.ffmpeg ? `Whisper (${m.whisper_model}) instalado` : "Audio y video: falta el grupo media o ffmpeg", m.faster_whisper && m.ffmpeg ? "ok" : "warn"),
    chip(`Tope de ${m.docling_max_pages || "sin tope"} páginas por PDF · OCR: ${m.docling_ocr}`));
  return s;
}

// ───────────── preguntar ─────────────
const TYPE_LABELS = {
  semantica: "Paráfrasis (gana el significado)",
  lexica: "Código exacto (gana BM25)",
  tabla: "Datos de una tabla",
  filtro: "Depende de la audiencia (filtro)",
};

async function loadExamples() {
  const examples = await api("/api/examples");
  const groups = Object.keys(TYPE_LABELS).map((type) => {
    const list = examples.filter((e) => e.type === type);
    // se muestran pocas por tipo: el set completo se usa en la pestaña Evaluación
    return h("div", { class: "examples-group" }, h("span", { class: "group-name" }, TYPE_LABELS[type]),
      ...list.slice(0, 3).map((e) => h("button", {
        type: "button", class: "example",
        onclick: () => {
          $("question").value = e.question;
          $("audience").value = e.audience || "";
          $("question").focus();
        },
      }, e.question)));
  });
  $("examples").replaceChildren(...groups);
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

/** Reproduce un audio o video en el instante citado (HTML5; el backend atiende Range, así que salta sin descargar todo). */
function playAt(media, type, seconds) {
  const slot = $("player-slot");
  const url = `/api/media/${encodeURIComponent(media)}`;
  let el = slot.firstElementChild;
  if (!el || el.dataset.media !== media) {
    el = h(type === "video" ? "video" : "audio", { controls: true, preload: "metadata", src: url, "data-media": media, class: "media-el" });
    slot.replaceChildren(el);
  }
  $("player-title").textContent = `${media} · ${mmss(seconds)}`;
  $("player").hidden = false;
  const go = () => { el.currentTime = seconds; el.play().catch(() => {}); };
  if (el.readyState >= 1) go(); else el.addEventListener("loadedmetadata", go, { once: true });
  $("player").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
$("player-close").addEventListener("click", () => { $("player-slot").replaceChildren(); $("player").hidden = true; });

/** La cita de un fragmento: botón ▶ mm:ss en audio y video, enlace a la página en un PDF, nada en el resto. */
function citeLocation(hit) {
  if (hit.media != null && hit.start_s != null) {
    return h("button", { type: "button", class: "play", title: `Reproducir ${hit.media} desde ${mmss(hit.start_s)}`,
      onclick: () => playAt(hit.media, hit.type, hit.start_s) }, `▶ ${mmss(hit.start_s)}`);
  }
  if (hit.type === "pdf" && hit.page) {
    return h("a", { class: "pagelink", href: `/api/pdf/${encodeURIComponent(hit.source)}#page=${hit.page}`, target: "_blank", rel: "noopener",
      title: "Abrir el PDF en esa página" }, hit.location);
  }
  return null;
}

function focusHit(n) {
  const el = $(`hit-${n}`);
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

/** Una tarjeta por etapa: qué hizo, cuánto tardó y qué fragmentos dejó (en orden). */
function renderStages(stages, finalIds) {
  return stages.map((stage) => {
    const items = stage.hits.map((hit, i) => {
      const ranks = hit.ranks ? ` · ${Object.entries(hit.ranks).map(([name, pos]) => `${name === "vector" ? "V" : "B"}#${pos}`).join(" ")}` : "";
      return h("li", { class: finalIds.has(hit.id) ? "final" : "" },
        h("b", {}, `${i + 1}. ${hit.section}`),
        h("div", { class: "meta" }, `${hit.source} · `, hit.location ? `${hit.location} · ` : "", audienceTag(hit.audience), ` · ${hit.score}${ranks}`),
        h("div", { class: "meta" }, short(hit.text, 70)));
    });
    return h("div", { class: "stage" },
      h("h3", {}, h("span", {}, stage.title), h("span", { class: "muted small" }, stage.ms ? `${stage.ms.toFixed(0)} ms` : "")),
      h("p", { class: "note" }, stage.note),
      stage.llm ? h("p", { class: "note" }, `${stage.llm.model} · ${stage.llm.tokens_in}→${stage.llm.tokens_out} tokens · ${money(stage.llm.cost_usd)}`) : null,
      items.length ? h("ol", {}, ...items) : h("p", { class: "empty" }, "Sin resultados."));
  });
}

function renderResult(r) {
  $("result").hidden = false;
  $("answer-text").replaceChildren(...renderAnswer(r.answer, r.hits.length));

  const badges = [];
  if (!r.usage) badges.push(chip("No se llamó al modelo: ningún fragmento coincidió", "warn"));
  else if (r.abstained) badges.push(chip("El modelo dijo que no lo encuentra", "warn"));
  else if (r.invalid_citations.length) badges.push(chip(`Citas inexistentes: ${r.invalid_citations.join(", ")} (posible invención)`, "bad"));
  else if (r.uncited) badges.push(chip("Respondió sin citar: no se puede verificar", "bad"));
  else badges.push(chip("Citas verificadas: todas apuntan a fragmentos reales", "ok"));
  $("badges").replaceChildren(...badges);
  $("usage").textContent = r.usage
    ? `${r.usage.provider} · ${r.usage.model} · ${r.usage.tokens_in} tokens de entrada, ${r.usage.tokens_out} de salida · costo total de la pregunta (con reranking si se usó): ${money(r.cost_usd)} · ${r.usage.seconds.toFixed(1)} s`
    : "No se llamó al modelo (US$ 0).";

  $("stages").replaceChildren(...renderStages(r.stages, new Set(r.hits.map((x) => x.id))));
  $("hits").replaceChildren(...r.hits.map((hit) =>
    h("div", { class: `hit${hit.cited ? " cited" : ""}`, id: `hit-${hit.number}` },
      h("div", { class: "hit-head" },
        h("span", { class: "num" }, String(hit.number)),
        h("span", { class: "hit-source" }, `${hit.source} › ${hit.section}`),
                citeLocation(hit),
        audienceTag(hit.audience),
        hit.cited ? chip("citado", "ok") : null,
        h("span", { class: "score", title: "Puntaje de la última etapa que lo ordenó" }, hit.score.toFixed(3))),
      h("p", {}, hit.text))));

  $("prompt").textContent = r.prompt || "(No se armó un prompt porque no se llamó al modelo.)";
  const names = { vector: "Búsqueda vectorial", bm25: "Búsqueda BM25", rrf: "Fusión RRF", parent: "Padre-hijo", rerank: "Reranking (LLM)", generate: "Generar con el modelo", total: "Total" };
  $("timings").replaceChildren(h("table", {}, h("tbody", {}, ...Object.entries(r.timings_ms).map(([k, v]) =>
    h("tr", {}, h("td", {}, names[k] || k), h("td", {}, `${v} ms`))))));
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ask-btn"), async () => {
    renderResult(await api("/api/ask", {
      method: "POST",
      body: {
        question: $("question").value.trim(), mode: $("mode").value, audience: $("audience").value || null,
        k: Number($("k").value), candidates: Number($("candidates").value), rerank: $("rerank").checked,
      },
    }));
  });
});

// ───────────── ingesta ─────────────
/** Lo que se sabe de cada archivo tras leerlo: OCR y páginas (PDF), idioma y duración (audio y video), tiempos y caché. */
function ingestDetail(d) {
  if (d.skipped) return "omitido (ver advertencias)";
  const parts = [];
  if (d.type === "pdf") {
    parts.push(d.ocr ? "OCR: sí" : "OCR: no", `${d.pages}${d.pages_total > d.pages ? ` de ${d.pages_total}` : ""} págs.`);
  } else if (d.type === "audio" || d.type === "video") {
    parts.push(`${mmss(d.duration_s)}`, `idioma ${d.language}`, `Whisper ${d.model}`);
    if (!d.from_cache) parts.push(`transcripción ${d.transcribe_seconds} s`);
  }
  if (d.seconds != null) parts.push(d.from_cache ? "desde caché" : `${d.seconds} s`);
  if (d.strategy_used && d.strategy_used !== "docling_hybrid" && d.strategy_used.includes("respaldo")) parts.push(d.strategy_used);
  return parts.join(" · ");
}

// El solape no significa lo mismo en cada estrategia: se dice en la etiqueta y se desactiva cuando no aplica.
const OVERLAP_UNITS = {"sentences": "oraciones", "parent_child": null, "section": null, "docling_hybrid": null};   // null = la estrategia no usa solape
function updateOverlapLabel() {
  const unit = OVERLAP_UNITS[$("strategy").value];
  $("overlap-label").textContent = unit ? `Solape (${unit})` : "Solape (no aplica)";
  $("overlap").disabled = !unit;
}
$("strategy").addEventListener("change", updateOverlapLabel);
updateOverlapLabel();

// Una frase por estrategia: qué hace y, sobre todo, qué NO es.
const STRATEGY_HELP = {
  parent_child: "Busca con trozos pequeños y entrega al modelo la sección completa a la que pertenecen.",
  sentences: "Junta oraciones hasta el máximo de palabras, repitiendo las últimas del fragmento anterior (solape).",
  section: "Un fragmento por sección: más contexto por fragmento, pero un vector más diluido.",
  docling_hybrid: "Chunking híbrido de Docling: sigue la estructura del PDF (encabezados, tablas, listas) y respeta un límite de tokens, uniendo lo muy pequeño y partiendo lo muy grande; se vectoriza cada fragmento con sus encabezados. NO es la «búsqueda híbrida» BM25 + vectores. Md, html y Word no pasan por Docling: usan «por oraciones» con un tamaño equivalente. Audio y video ya vienen en ventanas de tiempo.",
};
function updateStrategyUi() {
  const s = $("strategy").value;
  $("strategy-help").textContent = STRATEGY_HELP[s] || "";
  $("words-field").hidden = s === "docling_hybrid";
  $("tokens-field").hidden = s !== "docling_hybrid";
}
$("strategy").addEventListener("change", () => {
  $("max-words").value = $("strategy").value === "parent_child" ? 35 : 70;   // los hijos son más pequeños
  updateStrategyUi();
});
updateStrategyUi();

$("ingest-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ingest-btn"), async () => {
    const r = await api("/api/ingest", {
      method: "POST",
      body: { strategy: $("strategy").value, max_words: Number($("max-words").value), overlap: Number($("overlap").value), max_tokens: Number($("max-tokens").value) },
    });
    const box = $("ingest-result");
    box.hidden = false;
    box.replaceChildren(
      h("h2", {}, "Ingesta lista"),
      h("p", {}, `${r.documents} documentos → ${r.chunks} fragmentos de ${r.dimension} dimensiones, en ${r.store} (más el índice BM25), ${r.seconds.toFixed(1)} s.`),
      h("div", { class: "chips" }, ...r.warnings.map((w) => chip(w, "warn"))),
      h("div", { class: "table-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["Archivo", "Formato", "Secciones", "Fragmentos", "Detalle"].map((x) => h("th", {}, x)))),
        h("tbody", {}, ...Object.entries(r.per_document).map(([name, d]) => h("tr", {}, h("td", {}, name), h("td", {}, d.type), h("td", {}, String(d.sections)), h("td", {}, String(d.chunks)), h("td", {}, ingestDetail(d)))))))
    );
    await refreshStatus();
  });
});

// ───────────── evaluación ─────────────
const stat = (value, label) => h("div", { class: "stat" }, h("b", {}, value), h("span", {}, label));

function renderEval(data) {
  const { summary: s, rows } = data;
  const modes = Object.entries(s.modes);
  const bestHit1 = Math.max(...modes.map(([, m]) => m.hit1));
  const bestMrr = Math.max(...modes.map(([, m]) => m.mrr));
  const types = ["semantica", "lexica", "tabla", "filtro"];

  const overall = h("table", {}, h("thead", {}, h("tr", {}, ...["Modo", "hit@1", `hit@${s.k}`, "MRR", "Contexto (palabras)", ...types.map((t) => `${t} hit@1`)].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...modes.map(([, m]) => h("tr", {},
      h("td", {}, m.label),
      h("td", { class: `num${m.hit1 === bestHit1 ? " best" : ""}` }, `${m.hit1}/${s.n}`),
      h("td", { class: "num" }, `${m.hitk}/${s.n}`),
      h("td", { class: `num${m.mrr === bestMrr ? " best" : ""}` }, m.mrr.toFixed(2)),
      h("td", { class: "num" }, String(m.context_words)),
      ...types.map((t) => h("td", { class: "num" }, `${m.by_type[t].hit1}/${m.by_type[t].n}`))))));

  const names = modes.map(([key]) => key);
  const detail = h("table", {}, h("thead", {}, h("tr", {}, ...["Pregunta", "Tipo", ...modes.map(([, m]) => m.label)].map((x) => h("th", {}, x)))),
    h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, r.question), h("td", {}, r.type),
      ...names.map((n) => h("td", { class: `num${r.positions[n] === 1 ? "" : " bad"}` }, r.positions[n] ? `#${r.positions[n]}` : "no apareció"))))));

  $("eval-result").hidden = false;
  $("eval-result").replaceChildren(
    h("div", { class: "stats" }, stat(String(s.n), "preguntas de prueba"), stat(String(s.k), "k: fragmentos al modelo"), stat(money(s.cost_usd), "costo de la evaluación")),
    h("div", { class: "card table-scroll" }, h("h2", {}, "Resumen por modo"), overall,
      h("p", { class: "muted small" }, "Verde = mejor valor de la columna. Lee el desglose por tipo: la híbrida existe porque ningún buscador gana en todos los tipos. Mira también el contexto: más palabras al modelo = más tokens = más costo.")),
    h("details", { class: "card" }, h("summary", {}, "Detalle por pregunta (posición del primer fragmento correcto)"), h("div", { class: "table-scroll" }, detail)),
  );
}

for (const [id, withRerank, label] of [["eval-modes", false, "Comparar modos (gratis)"], ["eval-rerank", true, "Comparar también con reranking (cuesta)"]]) {
  $(id).addEventListener("click", () => guarded($(id), async () => {
    $(id).textContent = "Evaluando…";
    try {
      renderEval(await api("/api/evaluate", { method: "POST", body: { k: 3, candidates: 8, with_rerank: withRerank } }));
    } finally {
      $(id).textContent = label;
    }
  }));
}

// ───────────── documentos ─────────────
let docsLoaded = false;
async function loadDocs() {
  if (docsLoaded) return;
  const docs = await api("/api/documents");
  $("docs").replaceChildren(h("h2", {}, "Documentos de origen"),
    h("p", { class: "muted" }, "Un Markdown, una página web y un Word: cada lector extrae el texto y conserva la estructura (en el HTML se descartan menús y scripts; en el Word, las tablas pasan a líneas “celda | celda”)."),
    ...docs.map((d) => h("div", { class: "doc" }, h("h3", {}, `${d.source} `, chip(d.type)),
      d.type === "pdf" ? h("p", { class: "muted small" }, `${d.size_kb} KB · `, h("a", { href: d.url, target: "_blank", rel: "noopener" }, "abrir el PDF"), ` · ${d.warnings.join(" ")}`) : null,
      ...d.sections.map((sec) => h("details", {}, h("summary", {}, `${sec.title} · ${sec.words} palabras`), h("p", {}, sec.text))))));
  const media = await api("/api/media");
  $("media-list").replaceChildren(h("h2", {}, "Audio y video"),
    h("p", { class: "muted" }, "Carpeta de medios (MEDIA_DIR, fuera de Git). Se transcriben con faster-whisper al ingestar; los fragmentos citan el instante y el botón ▶ de una respuesta los reproduce."),
    media.length ? null : h("p", { class: "empty" }, "No hay audios ni videos. Copia archivos .mp3 .wav .m4a .flac .mp4 .mkv .mov .webm a backend/data/media/."),
    ...media.map((m) => h("div", { class: "doc" }, h("h3", {}, `${m.name} `, chip(m.type), m.duration ? chip(m.duration) : null),
      h(m.type === "video" ? "video" : "audio", { controls: true, preload: "none", src: m.url, class: "media-el" }))));
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
