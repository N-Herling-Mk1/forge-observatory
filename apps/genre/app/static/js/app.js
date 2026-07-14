"use strict";
const $ = s => document.querySelector(s);
let DATA = null, PHASE = "before", AVAIL = {};

// display names: FORGE is the umbrella; each experiment has a model + task label
const DISPLAY = {
  genre:  "BEARDOWN · genre recognition",
  phonon: "phonon · DOS reproduction",
  atlas:  "atlas · Run 3",
};

// cache-bust on the EDA generation timestamp so regenerated figures never serve stale
const figURL = file =>
  `/figures/${PHASE}/${file}?v=${encodeURIComponent((DATA && DATA.generated) || "")}`;

async function init(){
  try{
    const c = await (await fetch("/api/config")).json();
    PHASE = c.default_phase; AVAIL = c.available || {};
    const nm = $("#expName"); if (nm) nm.textContent = DISPLAY[c.experiment] || (c.experiment || "experiment").toUpperCase();
    const lg = $("#expLogo"); if (lg && c.logo){ lg.src = c.logo; lg.alt = c.experiment; lg.hidden = false; }
    buildPhaseToggle(c.phases || ["before","after"]);
  }catch(e){ /* config optional; fall back to before */ }
  await load();
}

function buildPhaseToggle(phases){
  $("#phaseToggle").innerHTML = phases.map(p=>{
    const dis = AVAIL[p] ? "" : "disabled";
    const act = p===PHASE ? "active" : "";
    return `<button class="${act}" data-phase="${p}" ${dis}>${p}</button>`;
  }).join("");
  $("#phaseToggle").querySelectorAll("button").forEach(b=>{
    b.onclick = () => { if(b.disabled) return; PHASE=b.dataset.phase; syncToggle(); load(); };
  });
}
function syncToggle(){
  $("#phaseToggle").querySelectorAll("button").forEach(b=>
    b.classList.toggle("active", b.dataset.phase===PHASE));
}

async function load(){
  let r;
  try{ r = await fetch(`/api/eda?phase=${PHASE}`); }
  catch(e){ return fail("backend unreachable"); }
  if(!r.ok){ const j = await r.json().catch(()=>({})); return fail(j.error || ("HTTP "+r.status)); }
  DATA = await r.json();
  render();
}
function fail(msg){
  $("#cards").innerHTML = `<div class="err">⚠ ${msg}<br><span class="dim">reload after generating this snapshot</span></div>`;
  ["#hero","#integrityPanel","#featStats","#varDetail","#featTable"].forEach(s=>$(s).innerHTML="");
  $("#typeTable").innerHTML="";
}

function render(){
  const pb = $("#phaseBadge"); if (pb) pb.textContent = DATA.phase || PHASE;
  $("#gen").textContent = DATA.generated ? "generated " + DATA.generated : "";
  renderSummary(); renderCards(); renderHero(); renderIntegrity(); renderTypes(); renderFeatures();
}

function renderSummary(){
  const box = $("#edaSummary"); if(!box) return;
  const s = DATA.summary;
  if(!s){ box.innerHTML = ""; return; }
  const flag = (ok,label)=>`<span class="sflag ${ok?'ok':'no'}">${ok?'✓':'✕'} ${label}</span>`;
  const n = (s.n_rows != null) ? Number(s.n_rows).toLocaleString() : "—";
  box.innerHTML = `
    <div class="sum-head">
      <span class="sum-tag">data health</span>
      ${flag(s.all_present,    "all present · no NaN")}
      ${flag(s.all_types_pass, "all types pass")}
      <span class="sum-n">n = ${n} rows${s.n_features ? " · " + s.n_features + " features" : ""}</span>
    </div>
    <p class="sum-text">${s.narrative || ""}</p>`;
}

function renderCards(){
  const mc = DATA.missing_corrupt || {}, c = mc.counts || {};
  const cards = [
    ["wav files", c.wav_total ?? "—"],
    ["spectrograms", c.grey_total ?? "—"],
    ["features", DATA.nerd_stats?.n_features ?? "—"],
    ["known issues", (mc.known_issues||[]).length],
    ["corrupt", (mc.corrupt_audio||[]).length],
    ["figures", (DATA.figures||[]).length],
  ];
  $("#cards").innerHTML = cards.map(([l,n])=>
    `<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

function renderHero(){
  const figs = DATA.figures || [];
  const pick = k => figs.find(f=>f.kind===k);
  // class balance spans the row; grey + colored spectrograms sit side by side below it
  const balance = pick("class_balance");
  const grey = pick("exemplars_grey");
  const color = pick("exemplars_color") || pick("exemplars");  // back-compat
  let html = "";
  if (balance) html += `<div class="imgwrap hero-wide">
      <img src="${figURL(balance.file)}" alt="class balance">
      ${balance.caption ? `<div class="hero-cap">${balance.caption}</div>` : ""}</div>`;
  for (const [f,tag] of [[grey,"grey-scale"],[color,"colored (magma)"]]){
    if(!f) continue;
    html += `<div class="imgwrap"><div class="hero-tag">${tag}</div>
        <img src="${figURL(f.file)}" alt="${f.kind}">
        ${f.caption ? `<div class="hero-cap">${f.caption}</div>` : ""}</div>`;
  }
  $("#hero").innerHTML = html || `<div class="dim">no hero figures emitted</div>`;
}

function renderIntegrity(){
  const mc = DATA.missing_corrupt || {};
  const sec = (title, rows, cols) => {
    if(!rows || !rows.length) return `<h2>${title}</h2><div class="dim">none</div>`;
    const head = cols.map(c=>`<th>${c}</th>`).join("");
    const body = rows.map(r=>`<tr>${cols.map(c=>`<td>${r[c]??""}</td>`).join("")}</tr>`).join("");
    return `<h2>${title} <span class="dim">(${rows.length})</span></h2><table><tr>${head}</tr>${body}</table>`;
  };
  const miss = Object.entries(mc.missing_per_representation||{})
    .flatMap(([rep,ids]) => ids.map(id => ({id, representation:rep})));
  let html = "";
  html += sec("Corrupt audio", mc.corrupt_audio, ["id","genre","error","bytes"]);
  html += sec("Off-duration", mc.off_duration, ["id","genre","duration_s","sample_rate"]);
  html += sec("Segment anomalies (3s)", mc.segment_anomalies, ["track","segments"]);
  html += sec("Missing across representations", miss, ["id","representation"]);
  html += `<h2>Known issues <span class="dim">(${(mc.known_issues||[]).length})</span></h2>`;
  html += (mc.known_issues||[]).map(s=>`<div class="issue">${s}</div>`).join("");
  $("#integrityPanel").innerHTML = html;
}

function renderTypes(){
  const cols = DATA.type_audit?.columns || [];
  const rows = cols.map(a=>`<tr>
      <td>${a.column}</td><td class="dim">${a.expected}</td><td>${a.actual_dtype}</td>
      <td>${a.n ?? "—"}</td>
      <td>${a.match ? '<span class="pill ok">match</span>' : '<span class="pill no">check</span>'}</td>
      <td class="dim">${a.note||""}</td></tr>`).join("");
  let html = `<tr><th>column</th><th>expected</th><th>actual</th><th>n</th><th>match</th><th>note</th></tr>${rows}`;
  const sp = DATA.type_audit?.spectrograms;
  if(sp){
    const nseen = sp.n_examined ?? Object.values(sp.observed_modes||{}).reduce((a,b)=>a+b,0);
    html += `<tr><td colspan="6" class="dim">spectrograms · n ${nseen} · modes ${JSON.stringify(sp.observed_modes)} · sizes ${
      JSON.stringify(sp.observed_sizes)} · unreadable ${(sp.unreadable||[]).length}</td></tr>`;
  }
  $("#typeTable").innerHTML = html;
}

// ---- numeric feature statistics: cards + variable toggle + sortable table ---
const COLS = [
  {k:"feature", label:"feature"}, {k:"mean", label:"mean"}, {k:"median", label:"median"},
  {k:"std", label:"std"}, {k:"min", label:"min"}, {k:"max", label:"max"},
  {k:"iqr", label:"IQR"}, {k:"n_outliers_iqr", label:"outliers"}, {k:"skew", label:"skew"},
];
let SORT = {k:"feature", dir:1};
let SEL = null;
let GENRE = null;   // null = combined; otherwise a genre name

const fnum = v => {
  if (v == null || isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-2)) return v.toExponential(2);
  return (Math.round(v * 1000) / 1000).toString();
};
const figForFeature = f => (DATA.figures||[]).find(g=>g.kind==="feature_dist" && g.feature===f);

// stats under the current scope: combined (GENRE=null) or one genre's slice
const scopeStat = feat => {
  const p = DATA.nerd_stats?.per_feature?.[feat];
  if (!p) return null;
  return GENRE ? (p.per_genre?.[GENRE] || null) : p;
};

function renderFeatures(){
  const per = DATA.nerd_stats?.per_feature || {};
  const names = Object.keys(per);
  if (!names.length){ ["#featStats","#varDetail","#featTable","#perGenre"].forEach(s=>$(s).innerHTML=""); return; }

  const genres = DATA.nerd_stats?.genres || [];
  $("#genreSelect").innerHTML = `<option value="">All genres (combined)</option>` +
    genres.map(g=>`<option value="${g}">${g}</option>`).join("");
  $("#genreSelect").value = GENRE || "";
  $("#genreSelect").onchange = e => { GENRE = e.target.value || null; applyScope(); };

  $("#varSelect").innerHTML = names.map(n=>`<option value="${n}">${n}</option>`).join("");
  $("#varSelect").onchange = e => selectVar(e.target.value);
  $("#search").oninput = () => drawTable();

  applyScope();
}

// (re)render cards + table + detail for the current GENRE scope
function applyScope(){
  const per = DATA.nerd_stats?.per_feature || {};
  const names = Object.keys(per);
  let totOut=0, skewName=null, iqrName=null, flagged=0, counted=0;
  for (const n of names){
    const p = scopeStat(n); if(!p) continue;
    counted++;
    totOut += p.n_outliers_iqr || 0;
    if (skewName===null || Math.abs(p.skew) > Math.abs(scopeStat(skewName).skew)) skewName = n;
    if (iqrName===null || (p.iqr||0) > (scopeStat(iqrName).iqr||0)) iqrName = n;
    if ((p.outlier_pct||0) > 5) flagged++;
  }
  $("#featStats").innerHTML = [
    ["numeric features", counted], ["total outliers", totOut],
    [">5% outliers", flagged], ["most skewed", skewName||"—"], ["widest IQR", iqrName||"—"],
  ].map(([l,n])=>`<div class="card"><div class="n" style="font-size:18px">${n}</div><div class="l">${l}</div></div>`).join("");

  drawTable();
  selectVar(SEL && per[SEL] ? SEL : names[0]);
}

function drawTable(){
  const per = DATA.nerd_stats?.per_feature || {};
  const q = ($("#search").value || "").toLowerCase();
  let rows = Object.keys(per).map(feature => ({feature, ...(scopeStat(feature) || {})}))
                   .filter(r => !q || r.feature.toLowerCase().includes(q));
  rows.sort((a,b)=>{
    const x=a[SORT.k], y=b[SORT.k];
    const c = (SORT.k==="feature") ? String(x).localeCompare(String(y)) : (x-y);
    return c * SORT.dir;
  });
  $("#figCount").textContent = rows.length + " / " + Object.keys(per).length;
  const head = COLS.map(c=>{
    const arrow = SORT.k===c.k ? (SORT.dir>0?" ▲":" ▼") : "";
    return `<th data-k="${c.k}" class="sortable">${c.label}${arrow}</th>`;
  }).join("");
  const body = rows.map(r=>{
    const cls = r.feature===SEL ? ' class="sel"' : "";
    const tds = COLS.map(c => c.k==="feature" ? `<td>${r.feature}</td>`
      : c.k==="n_outliers_iqr" ? `<td>${r.n_outliers_iqr} <span class="dim">(${r.outlier_pct}%)</span></td>`
      : `<td>${fnum(r[c.k])}</td>`).join("");
    return `<tr data-f="${r.feature}"${cls}>${tds}</tr>`;
  }).join("");
  $("#featTable").innerHTML = `<tr>${head}</tr>${body}`;
}

// client-drawn histogram + box for one feature under one genre (combined uses the PNG)
function genreChartSVG(feat, genre){
  const F = DATA.nerd_stats?.per_feature?.[feat];
  const h = F?.hist, s = F?.per_genre?.[genre];
  if(!h || !s) return `<div class="dim">no per-genre data</div>`;
  const edges = h.edges, counts = h.per_genre[genre] || [];
  const lo = edges[0], hi = edges[edges.length-1], span = (hi-lo) || 1;
  const maxC = Math.max(1, ...counts);
  const HX0=46, HX1=560, HY0=24, HY1=212;
  const bw = (HX1-HX0)/counts.length;
  const bars = counts.map((c,i)=>{
    const x = HX0 + i*bw, bh = (c/maxC)*(HY1-HY0);
    return `<rect x="${x.toFixed(1)}" y="${(HY1-bh).toFixed(1)}" width="${Math.max(0.5,bw-1).toFixed(1)}" height="${bh.toFixed(1)}" fill="#1FB6C1" opacity="0.82"/>`;
  }).join("");
  const tx = v => HX0 + ((v-lo)/span)*(HX1-HX0);
  const xt = [lo,(lo+hi)/2,hi].map(v=>
    `<text x="${tx(v).toFixed(1)}" y="228" fill="#6f8693" font-size="10" text-anchor="middle">${fnum(v)}</text>`).join("");
  // box (right), shares the same value range as the histogram
  const BX=632, BW=46, BY0=24, BY1=212;
  const yv = v => BY1 - ((v-lo)/span)*(BY1-BY0);
  const q1=yv(s.q1), q3=yv(s.q3), med=yv(s.median);
  const wl=yv(s.whislo ?? s.min), wh=yv(s.whishi ?? s.max), cx=BX+BW/2;
  let box = `
    <line x1="${cx}" y1="${wh}" x2="${cx}" y2="${wl}" stroke="#6f8693"/>
    <line x1="${BX+8}" y1="${wh}" x2="${BX+BW-8}" y2="${wh}" stroke="#6f8693"/>
    <line x1="${BX+8}" y1="${wl}" x2="${BX+BW-8}" y2="${wl}" stroke="#6f8693"/>
    <rect x="${BX}" y="${q3.toFixed(1)}" width="${BW}" height="${Math.max(1,q1-q3).toFixed(1)}" fill="#F0A500" opacity="0.85" stroke="#F0A500"/>
    <line x1="${BX}" y1="${med}" x2="${BX+BW}" y2="${med}" stroke="#1a1205" stroke-width="2"/>`;
  if((s.n_outliers_iqr||0) > 0){
    if(s.min < (s.whislo ?? s.min)) box += `<circle cx="${cx}" cy="${yv(s.min)}" r="2.5" fill="#FF6A1A"/>`;
    if(s.max > (s.whishi ?? s.max)) box += `<circle cx="${cx}" cy="${yv(s.max)}" r="2.5" fill="#FF6A1A"/>`;
  }
  return `<svg viewBox="0 0 720 250" class="genre-chart" preserveAspectRatio="xMidYMid meet">
    <text x="${(HX0+HX1)/2}" y="15" fill="#cfe6ea" font-size="11" text-anchor="middle">${feat} — ${genre} · histogram</text>
    <text x="${cx}" y="15" fill="#cfe6ea" font-size="11" text-anchor="middle">box</text>
    ${bars}
    <line x1="${HX0}" y1="${HY1}" x2="${HX1}" y2="${HY1}" stroke="#1c2a3a"/>
    ${xt}${box}
  </svg>`;
}

function selectVar(name){
  const base = DATA.nerd_stats?.per_feature?.[name]; if(!base) return;
  SEL = name;
  if ($("#varSelect").value !== name) $("#varSelect").value = name;
  const p = scopeStat(name) || base;     // scoped numbers (combined or one genre)
  const fig = figForFeature(name);        // combined distribution PNG
  const gfile = DATA.nerd_stats?.per_feature?.[name]?.per_genre_fig?.[GENRE];
  const scopeTag = GENRE
    ? `<span class="scope-tag">${GENRE}</span>`
    : `<span class="scope-tag all">all genres</span>`;
  const figHTML = GENRE
    ? (gfile ? `<img src="${figURL(gfile)}" alt="${name} — ${GENRE}" data-file="${gfile}">`
             : `<div class="dim">no figure</div>`)
    : (fig ? `<img src="${figURL(fig.file)}" alt="${name}" data-file="${fig.file}">`
           : `<div class="dim">no figure</div>`);
  const stats = [
    ["count", p.count], ["mean", fnum(p.mean)], ["median", fnum(p.median)],
    ["mode≈", fnum(p.mode_round3)], ["std", fnum(p.std)], ["min", fnum(p.min)],
    ["max", fnum(p.max)], ["Q1", fnum(p.q1)], ["Q3", fnum(p.q3)], ["IQR", fnum(p.iqr)],
    ["outliers", `${p.n_outliers_iqr} (${p.outlier_pct}%)`], ["skew", fnum(p.skew)],
  ];
  $("#varDetail").innerHTML = `
    <div class="var-fig">${figHTML}
      <div class="fig-note">distribution · ${GENRE || "all genres"}</div></div>
    <div class="stat-grid">
      <div class="grid-head">stats · ${scopeTag}</div>
      ${stats.map(([k,v])=>`<div class="stat"><span class="sk">${k}</span><span class="sv">${v}</span></div>`).join("")}
    </div>`;
  document.querySelectorAll("#featTable tr[data-f]").forEach(tr=>
    tr.classList.toggle("sel", tr.dataset.f===name));
  renderPerGenre(name);
}

// per-genre breakdown for the inspected feature (always shows every genre)
function renderPerGenre(name){
  const pg = DATA.nerd_stats?.per_feature?.[name]?.per_genre || {};
  const genres = DATA.nerd_stats?.genres || Object.keys(pg);
  if (!genres.length){ $("#perGenre").innerHTML = ""; return; }
  const cols = [["mean","mean"],["median","median"],["std","std"],["min","min"],
                ["max","max"],["iqr","IQR"],["n_outliers_iqr","outliers"],["skew","skew"]];
  const head = `<th>genre</th>` + cols.map(([,l])=>`<th>${l}</th>`).join("");
  const body = genres.map(g=>{
    const s = pg[g]; if(!s) return "";
    const cls = g===GENRE ? ' class="sel"' : "";
    const tds = cols.map(([k])=> k==="n_outliers_iqr"
      ? `<td>${s.n_outliers_iqr} <span class="dim">(${s.outlier_pct}%)</span></td>`
      : `<td>${fnum(s[k])}</td>`).join("");
    return `<tr data-g="${g}"${cls}><td>${g}</td>${tds}</tr>`;
  }).join("");
  $("#perGenre").innerHTML =
    `<h2>${name} — by genre <span class="dim">(click a row to scope everything to that genre)</span></h2>
     <div class="table-wrap"><table id="pgTable"><tr>${head}</tr>${body}</table></div>`;
}

// table: sort on header click, select variable on row click
$("#featTable").addEventListener("click", e=>{
  const th = e.target.closest("th.sortable");
  if (th){ const k=th.dataset.k; SORT = {k, dir: SORT.k===k ? -SORT.dir : 1}; drawTable(); return; }
  const tr = e.target.closest("tr[data-f]");
  if (tr){ selectVar(tr.dataset.f); $("#varDetail").scrollIntoView({behavior:"smooth", block:"nearest"}); }
});

// per-genre breakdown: click a genre row to scope the whole tab to it (toggle off to combined)
$("#perGenre").addEventListener("click", e=>{
  const tr = e.target.closest("tr[data-g]"); if(!tr) return;
  GENRE = (GENRE === tr.dataset.g) ? null : tr.dataset.g;
  const gs = $("#genreSelect"); if (gs) gs.value = GENRE || "";
  applyScope();
});

// enlarge the selected variable's figure
$("#varDetail").addEventListener("click", e=>{
  const img = e.target.closest("img"); if(!img) return;
  $("#modalImg").src = img.src; $("#modalCap").textContent = SEL || "";
  $("#modal").classList.add("open");
});
$("#modal").addEventListener("click", ()=>$("#modal").classList.remove("open"));

// tabs
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("nav button").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll("main section").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); $("#"+b.dataset.tab).classList.add("active");
});

init();
