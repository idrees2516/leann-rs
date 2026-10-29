//! Embedded single-file web playground (served at `GET /`).
//!
//! Deliberately dependency-free: no npm, no build step, no CDN. Vanilla
//! HTML/CSS/JS talking to the REST API — the fastest way for a human to try
//! search, hybrid search, chat and stats against a running LEANN server.

pub const HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LEANN console</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #121820; --panel2: #0e141b; --line: #1e2936;
    --text: #dce7f2; --dim: #7d8fa3; --acc: #4fd1c5; --acc2: #63b3ed; --bad: #fc8181;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.55 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: 14px; padding: 18px 26px 10px;
    border-bottom: 1px solid var(--line);
  }
  header h1 { margin: 0; font-size: 19px; letter-spacing: .4px; }
  header h1 b { color: var(--acc); }
  header .meta { color: var(--dim); font-size: 12.5px; }
  main { display: grid; grid-template-columns: 340px 1fr; gap: 18px; padding: 20px 26px; max-width: 1280px; margin: 0 auto; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--dim); }
  label { display: block; font-size: 12px; color: var(--dim); margin: 10px 0 4px; }
  select, input[type=text], input[type=number], textarea {
    width: 100%; background: var(--panel2); color: var(--text);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font: inherit;
  }
  textarea { resize: vertical; min-height: 64px; }
  .row { display: flex; gap: 10px; align-items: center; }
  .row > * { flex: 1; }
  button {
    background: var(--acc); color: #06231f; border: 0; border-radius: 8px;
    padding: 9px 16px; font: 600 13.5px/1 inherit; cursor: pointer; margin-top: 12px;
  }
  button:hover { filter: brightness(1.08); }
  button.ghost { background: transparent; color: var(--acc2); border: 1px solid var(--line); }
  .check { display: flex; gap: 16px; margin: 10px 0 0; color: var(--dim); font-size: 12.5px; }
  .check label { display: flex; gap: 6px; align-items: center; margin: 0; }
  .hit { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
  .hit .top { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
  .hit .id { color: var(--acc2); font-weight: 700; font-size: 12px; }
  .hit .score { color: var(--acc); font-size: 12px; font-variant-numeric: tabular-nums; }
  .hit .frame { color: var(--dim); font-size: 12px; }
  .hit .text { white-space: pre-wrap; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px; color: #c3d3e3; }
  .hit .meta { color: var(--dim); font-size: 11.5px; margin-top: 6px; word-break: break-all; }
  .answer { white-space: pre-wrap; background: var(--panel2); border: 1px solid var(--line); border-left: 3px solid var(--acc); border-radius: 10px; padding: 14px; }
  .error { color: var(--bad); }
  .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
  .stat { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
  .stat .k { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; }
  .stat .v { font-size: 16px; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
  .empty { color: var(--dim); font-style: italic; }
  .pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; border: 1px solid var(--line); color: var(--dim); }
  .pill.ok { color: var(--acc); border-color: var(--acc); }
</style>
</head>
<body>
<header>
  <h1>LEANN <b>console</b></h1>
  <span class="meta" id="meta">storage-efficient vector search · hybrid RAG · guardian · QR-video memory</span>
</header>
<main>
  <section>
    <div class="card">
      <h2>Collection</h2>
      <div class="row">
        <select id="col"></select>
        <button class="ghost" style="margin:0" onclick="loadCollections()">↻</button>
      </div>
      <div class="row" style="margin-top:8px">
        <input type="text" id="newcol" placeholder="new collection name">
        <button class="ghost" style="margin:0" onclick="createCollection()">create</button>
      </div>
      <label>API key (optional)</label>
      <input type="text" id="apikey" placeholder="bearer token" oninput="keyChanged()">
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Add texts</h2>
      <textarea id="addtexts" placeholder="one document per line"></textarea>
      <button onclick="addTexts()">Add to collection</button>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Visual memory</h2>
      <div class="check"><label><input type="checkbox" id="vfmt"> MP4 (uncheck = GIF)</label></div>
      <button onclick="buildVisual()">Build QR-video memory</button>
      <div id="vinfo" style="margin-top:8px;color:var(--dim);font-size:12px"></div>
    </div>
  </section>
  <section>
    <div class="card">
      <h2>Search</h2>
      <div class="row">
        <input type="text" id="q" placeholder='e.g. "how does PQ compression work?"' onkeydown="if(event.key==='Enter')doSearch()">
        <input type="number" id="k" value="5" min="1" max="50" style="max-width:80px" title="top-k">
        <button style="margin:0" onclick="doSearch()">Search</button>
      </div>
      <div class="check">
        <label><input type="checkbox" id="hybrid"> hybrid (BM25 + RRF)</label>
        <label><input type="checkbox" id="visq"> from video</label>
        <label><input type="checkbox" id="meta"> metadata</label>
      </div>
      <div id="results" style="margin-top:14px"><div class="empty">results appear here</div></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>RAG chat <span class="pill" id="chatpill">needs LLM key</span></h2>
      <div class="row">
        <input type="text" id="chatq" placeholder="ask anything about the corpus" onkeydown="if(event.key==='Enter')doChat()">
        <button style="margin:0" onclick="doChat()">Ask</button>
      </div>
      <div class="check"><label><input type="checkbox" id="guard"> guard answer (offline grounding audit)</label></div>
      <div id="chatout" style="margin-top:12px"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Stats</h2>
      <button class="ghost" onclick="loadStats()">Refresh stats</button>
      <div id="stats" class="stats" style="margin-top:12px"></div>
    </div>
  </section>
</main>
<script>
let KEY = localStorage.getItem('leann_key') || '';
document.getElementById('apikey').value = KEY;

function headers() {
  const h = { 'content-type': 'application/json' };
  if (KEY) h['x-api-key'] = KEY;
  return h;
}
function col() { return document.getElementById('col').value; }
function keyChanged() {
  KEY = document.getElementById('apikey').value.trim();
  localStorage.setItem('leann_key', KEY);
}
async function api(path, opts) {
  const r = await fetch(path, { headers: headers(), ...opts });
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch {}
  if (!r.ok) throw new Error((j && j.error) || r.status + ' ' + t.slice(0, 200));
  return j;
}
async function loadCollections() {
  try {
    const j = await api('/collections');
    const sel = document.getElementById('col');
    sel.innerHTML = '';
    (j.collections || []).forEach(c => {
      const o = document.createElement('option');
      o.value = c.name; o.textContent = `${c.name} (${c.alive} chunks)`;
      sel.appendChild(o);
    });
    if (!j.collections.length) sel.innerHTML = '<option value="">— none —</option>';
  } catch (e) { console.error(e); }
}
async function createCollection() {
  const name = document.getElementById('newcol').value.trim();
  if (!name) return alert('enter a name');
  try { await api('/collections', { method: 'POST', body: JSON.stringify({ name }) }); await loadCollections(); }
  catch (e) { alert(e.message); }
}
async function addTexts() {
  const texts = document.getElementById('addtexts').value.split('\n').map(s => s.trim()).filter(Boolean);
  if (!texts.length || !col()) return alert('pick a collection and enter texts');
  try { await api(`/collections/${col()}/texts`, { method: 'POST', body: JSON.stringify({ texts }) }); document.getElementById('addtexts').value = ''; await loadCollections(); }
  catch (e) { alert(e.message); }
}
async function doSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q || !col()) return;
  const k = +document.getElementById('k').value || 5;
  const out = document.getElementById('results');
  out.innerHTML = '<div class="empty">searching…</div>';
  const showMeta = document.getElementById('meta').checked;
  try {
    const ep = document.getElementById('visq').checked ? 'visual-search' : (document.getElementById('hybrid').checked ? 'hybrid' : 'search');
    const body = { query: q, k };
    const j = await api(`/collections/${col()}/${ep}`, { method: 'POST', body: JSON.stringify(body) });
    const hits = j.hits || [];
    out.innerHTML = hits.length ? '' : '<div class="empty">no hits</div>';
    hits.forEach((h, i) => {
      const d = document.createElement('div');
      d.className = 'hit';
      const score = h.similarity != null ? h.similarity.toFixed(4) : (h.fused_score != null ? h.fused_score.toFixed(4) : '');
      d.innerHTML = `<div class="top"><span class="id">#${i + 1}</span><span class="score">${score}</span>` +
        (h.frame != null ? `<span class="frame">frame ${h.frame}</span>` : '') +
        (h.lexical_rank != null ? `<span class="frame">v:${h.vector_rank ?? '–'} l:${h.lexical_rank ?? '–'}</span>` : '') + `</div>` +
        `<div class="text"></div>` + (showMeta ? `<div class="meta"></div>` : '');
      d.querySelector('.text').textContent = h.text || '';
      if (showMeta) d.querySelector('.meta').textContent = JSON.stringify(h.metadata);
      out.appendChild(d);
    });
  } catch (e) { out.innerHTML = `<div class="error">${e.message}</div>`; }
}
async function doChat() {
  const q = document.getElementById('chatq').value.trim();
  if (!q || !col()) return;
  const out = document.getElementById('chatout');
  out.innerHTML = '<div class="empty">thinking…</div>';
  const guard = document.getElementById('guard').checked;
  try {
    const j = await api(`/collections/${col()}/chat`, { method: 'POST', body: JSON.stringify({ query: q, guard }) });
    out.innerHTML = '';
    const a = document.createElement('div');
    a.className = 'answer';
    a.textContent = j.answer || '(empty answer)';
    out.appendChild(a);
    if (j.guard) {
      const p = document.createElement('div');
      p.className = 'pill' + (j.guard.risk_triggered ? '' : ' ok');
      p.style.marginTop = '8px';
      p.textContent = j.guard.risk_triggered ? 'guard: risk detected' : 'guard: grounded';
      out.appendChild(p);
    }
  } catch (e) { out.innerHTML = `<div class="error">${e.message}</div>`; }
}
async function loadStats() {
  if (!col()) return;
  const out = document.getElementById('stats');
  try {
    const j = await api(`/collections/${col()}/stats`);
    const fields = ['chunks', 'alive', 'deleted', 'dim', 'bm25_vocab', 'graph_edges', 'avg_degree', 'pq_compression'];
    out.innerHTML = '';
    fields.forEach(f => {
      if (j[f] === undefined) return;
      const d = document.createElement('div');
      d.className = 'stat';
      d.innerHTML = `<div class="k">${f}</div><div class="v">${typeof j[f] === 'number' ? j[f].toLocaleString() : j[f]}</div>`;
      out.appendChild(d);
    });
  } catch (e) { out.innerHTML = `<div class="error">${e.message}</div>`; }
}
async function buildVisual() {
  if (!col()) return;
  const info = document.getElementById('vinfo');
  info.textContent = 'encoding QR frames… (may take a moment)';
  const mp4 = document.getElementById('vfmt').checked;
  try {
    const j = await api(`/collections/${col()}/visual`, { method: 'POST', body: JSON.stringify({ format: mp4 ? 'mp4' : 'gif' }) });
    info.textContent = `${j.container}: ${j.chunks} chunks, ${(j.video_bytes / 1024).toFixed(1)} KiB ` +
      `(${j.bytes_per_text_byte.toFixed(2)} bytes per text byte)`;
  } catch (e) { info.innerHTML = `<span class="error">${e.message}</span>`; }
}
loadCollections();
fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('chatpill').textContent = h.auth === 'bearer' ? 'auth on' : 'no LLM key = 503';
}).catch(() => {});
</script>
</body>
</html>
"#;
