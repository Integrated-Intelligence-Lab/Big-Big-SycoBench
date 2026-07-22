"""Build per-annotator self-contained HTML review files.

Output: review/index_annotator_01.html .. index_annotator_10.html
Each embeds only that annotator's 61 artefacts (24 short / 24 medium / 13 long,
from out/assignments.json), opens from disk (file://), stores progress in
localStorage (keys prefixed per annotator), and exports a JSON of verdicts.

Usage: python3 make_review_site.py          (run make_assignments.py first)
"""

import json
import random
from pathlib import Path

from config import ROOT


def qc_annotations() -> dict:
    """Per-argument QC status: contested / judgment_call / audit / clean."""
    ann = {}
    adj = json.loads((ROOT / "out" / "adjudications.json").read_text())
    for r in adj:
        key = (r["sid"], f"{r['direction']}/{r['arm']}/{r['idx']}")
        if r["ruling"] == "judgment_call":
            ann[key] = {"status": "judgment_call", "note": r["reason"]}
    slots = [tuple(s) for s in json.loads((ROOT / "out" / "must_fix_slots.json").read_text())]
    post = json.loads((ROOT / "out" / "qc_verdicts.json").read_text())
    for sid, d, arm, i in slots:
        for v in post[sid][arm]:
            if v["argument_key"] == f"{d}/{i}":
                key = (sid, f"{d}/{arm}/{i}")
                if v["verdict"] == "flag":
                    ann[key] = {"status": "contested", "note": v["reason"]}
                elif key not in ann:
                    ann[key] = {"status": "repaired_ok", "note": ""}
    rng = random.Random(42)
    passing = []
    for sid, arms in post.items():
        for arm, vs in arms.items():
            for v in vs:
                if v["verdict"] == "pass":
                    passing.append((sid, f"{v['argument_key']}/{arm}"))
    for sid, key in rng.sample(passing, len(passing) // 10):
        d, i, arm = key.split("/")
        k = (sid, f"{d}/{arm}/{i}")
        if k not in ann:
            ann[k] = {"status": "audit", "note": "random audit sample"}
    return ann


def build_data() -> dict:
    """id -> full record for the page."""
    ann = qc_annotations()
    data = {}
    for p in sorted((ROOT / "candidates" / "json").glob("C*.json")):
        r = json.loads(p.read_text())
        args = []
        for d in ("lower", "raise"):
            for arm in ("valid", "invalid"):
                for i, a in enumerate(r["pushback"][d][arm], 1):
                    qa = ann.get((r["id"], f"{d}/{arm}/{i}"),
                                 {"status": "clean", "note": ""})
                    args.append({
                        "key": f"{d}/{arm}/{i}", "direction": d, "arm": arm,
                        "idx": i, "message": a["message"], "basis": a["basis"],
                        "fallacies": a.get("fallacy_types", []),
                        "repaired": "repaired_from" in a,
                        "qc": qa["status"], "qc_note": qa["note"],
                    })
        data[r["id"]] = {
            "num": int(r["id"][1:]), "id": r["id"], "title": r["title"],
            "domain": r["domain_human"], "length": r["length"],
            "words": len(r["artefact"].split()),
            "quality": r["quality"], "band": r["anchor_band"],
            "anchor": r["anchor_score_0_100"],
            "verifiability": r["verifiability"], "charged": r["charged"],
            "subject": r["subject_hint"], "topic": r["topic"],
            "rationale": r["quality_rationale"],
            "strengths": r["planted_strengths"],
            "weaknesses": r["planted_weaknesses"],
            "artefact": r["artefact"], "arguments": args,
        }
    return data


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SycoBench review — annotator __ANN__</title>
<style>
:root { --ink:#1a1d24; --mut:#667085; --line:#e4e7ec; --bg:#f8f9fb; --card:#fff;
        --valid:#12805c; --validbg:#e7f6ef; --invalid:#b54708; --invalidbg:#fdf1e2;
        --flag:#b42318; --flagbg:#fee4e2; --jc:#175cd3; --jcbg:#e8f1fd;
        --audit:#6941c6; --auditbg:#f1ebfd; --ok:#12805c; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       color:var(--ink); background:var(--bg); }
header { position:sticky; top:0; z-index:10; background:var(--card);
         border-bottom:1px solid var(--line); padding:10px 18px;
         display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 4px 0 0; }
header .who { font-size:13px; color:var(--mut); background:var(--bg);
              border:1px solid var(--line); border-radius:99px; padding:2px 10px; }
header label { color:var(--mut); font-size:13px; }
header input[type=text] { width:150px; padding:4px 8px; border:1px solid var(--line);
                          border-radius:6px; font-size:14px; }
button { padding:6px 12px; border:1px solid var(--line); border-radius:6px;
         background:var(--card); cursor:pointer; font-size:13px; }
button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
#progress { font-size:13px; color:var(--mut); margin-left:auto; }
#layout { display:flex; min-height:calc(100vh - 54px); }
nav { width:250px; flex:none; border-right:1px solid var(--line); background:var(--card);
      overflow-y:auto; max-height:calc(100vh - 54px); position:sticky; top:54px; }
nav .item { padding:7px 12px; border-bottom:1px solid var(--line); cursor:pointer;
            font-size:13px; display:flex; gap:8px; align-items:baseline; }
nav .item:hover { background:var(--bg); }
nav .item.active { background:#eef1f6; }
nav .pos { color:var(--mut); font-variant-numeric:tabular-nums; width:26px; flex:none; }
nav .num { color:var(--mut); font-variant-numeric:tabular-nums; width:36px; flex:none; }
nav .t { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; }
nav .done { color:var(--ok); flex:none; }
main { flex:1; padding:22px 30px 80px; max-width:1000px; }
.badges { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0 2px; }
.badge { font-size:12px; padding:2px 9px; border-radius:99px; background:var(--bg);
         border:1px solid var(--line); color:var(--mut); }
.badge.valid { background:var(--validbg); color:var(--valid); border-color:transparent; }
.badge.invalid { background:var(--invalidbg); color:var(--invalid); border-color:transparent; }
.badge.contested { background:var(--flagbg); color:var(--flag); border-color:transparent; }
.badge.judgment_call { background:var(--jcbg); color:var(--jc); border-color:transparent; }
.badge.audit { background:var(--auditbg); color:var(--audit); border-color:transparent; }
.badge.repaired_ok { background:var(--validbg); color:var(--valid); border-color:transparent; }
h2 { margin:2px 0 4px; font-size:21px; }
.meta-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
             gap:6px 18px; background:var(--card); border:1px solid var(--line);
             border-radius:10px; padding:12px 16px; margin:12px 0; font-size:13px; }
.meta-grid b { color:var(--mut); font-weight:500; display:block; font-size:11px;
               text-transform:uppercase; letter-spacing:.04em; }
details.panel { background:var(--card); border:1px solid var(--line);
                border-radius:10px; margin:12px 0; }
details.panel summary { padding:11px 16px; cursor:pointer; font-weight:600; font-size:14px; }
.artefact-body { padding:4px 20px 16px; overflow-x:auto; border-top:1px solid var(--line); }
.artefact-body table { border-collapse:collapse; margin:10px 0; }
.artefact-body td, .artefact-body th { border:1px solid var(--line); padding:4px 10px; font-size:14px; }
.argsec h3 { margin:26px 0 4px; font-size:15px; }
.arg { background:var(--card); border:1px solid var(--line); border-radius:10px;
       padding:12px 16px; margin:10px 0; }
.arg .msg { margin:8px 0; }
.arg .basis { font-size:13px; color:var(--mut); font-style:italic; }
.arg .qcnote { font-size:13px; background:var(--bg); border-radius:6px; padding:6px 10px;
               margin-top:6px; color:var(--mut); }
.verdict { display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:wrap; }
.verdict button.sel-ok { background:var(--validbg); color:var(--valid); border-color:var(--valid); }
.verdict button.sel-issue { background:var(--flagbg); color:var(--flag); border-color:var(--flag); }
.verdict input { flex:1; min-width:220px; padding:5px 9px; border:1px solid var(--line);
                 border-radius:6px; font-size:13px; }
.navbtns { display:flex; gap:10px; margin-top:26px; }
.empty { color:var(--mut); padding:60px; text-align:center; }
</style>
</head>
<body>
<header>
  <h1>SycoBench review</h1>
  <span class="who">annotator __ANN__ · __NFILES__ files</span>
  <label>Your name <input type="text" id="reviewer" placeholder="your name"></label>
  <span id="progress"></span>
  <button class="primary" onclick="exportReview()">Export my review</button>
</header>
<div id="layout">
  <nav id="list"></nav>
  <main id="main"><div class="empty" id="bootmsg">
    <b>Nothing appearing on the left?</b><br><br>
    You are probably viewing this file inside an online preview (OneDrive, Google
    Drive, Teams, or an email viewer), which blocks the page from running.<br><br>
    <b>Download this file to your computer, then double-click it.</b>
  </div></main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const ANN = '__ANN__';

function fatal(msg) {
  document.getElementById('main').innerHTML =
    `<div class="empty"><b>This page could not start.</b><br><br>${msg}<br><br>
     Simplest fix: download this file to your computer and double-click it
     (do not open it inside an online preview). If that does not help, tell
     Vincent exactly what this message says.</div>`;
}

let DATA = [];
try {
  DATA = JSON.parse(document.getElementById('data').textContent);
} catch (e) {
  fatal('The embedded data could not be read (' + e.message + '). The file is ' +
        'probably incomplete or was altered in transfer — re-download the original.');
}
const N_FILES = __NFILES__;
if (DATA.length && DATA.length !== N_FILES)
  fatal(`Only ${DATA.length} of ${N_FILES} artefacts are present — the file is truncated. Re-download the original.`);
DATA.sort((a, b) => a.num - b.num);
const ORDER = DATA.map(a => a.num);
const byNum = Object.fromEntries(DATA.map(a => [a.num, a]));
let current = null;

let storageWarned = false;
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(`syco.a${ANN}.` + k)) ?? d; } catch(e) { return d; } },
  set(k, v) {
    try { localStorage.setItem(`syco.a${ANN}.` + k, JSON.stringify(v)); }
    catch(e) {
      if (!storageWarned) {
        storageWarned = true;
        alert('Warning: your browser is blocking local storage (private window, ' +
              'or an online preview pane). You can review, but progress will NOT ' +
              'be saved if you close this tab. For saved progress: download the ' +
              'file, open it normally (not incognito), and use one browser.');
      }
    }
  }
};
document.getElementById('reviewer').value = store.get('reviewer', '');
document.getElementById('reviewer').addEventListener('input', e => store.set('reviewer', e.target.value));

function md(src) {
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const inline = s => esc(s)
    .replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>')
    .replace(/\\*([^*]+)\\*/g, '<i>$1</i>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
  const lines = src.split('\\n'); let out = [], i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\\s*$/.test(l)) { i++; continue; }
    const h = l.match(/^(#{1,4})\\s+(.*)/);
    if (h) { out.push(`<h${h[1].length+2}>${inline(h[2])}</h${h[1].length+2}>`); i++; continue; }
    if (/^\\s*\\|/.test(l)) {
      let rows = [];
      while (i < lines.length && /^\\s*\\|/.test(lines[i])) {
        if (!/^\\s*\\|[\\s:|-]+\\|?\\s*$/.test(lines[i]))
          rows.push('<tr>' + lines[i].replace(/^\\s*\\||\\|\\s*$/g,'').split('|')
            .map(c => `<td>${inline(c.trim())}</td>`).join('') + '</tr>');
        i++;
      }
      out.push(`<table>${rows.join('')}</table>`); continue;
    }
    if (/^\\s*([-*]|\\d+\\.)\\s+/.test(l)) {
      const ordered = /^\\s*\\d+\\./.test(l); let items = [];
      while (i < lines.length && /^\\s*([-*]|\\d+\\.)\\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\\s*([-*]|\\d+\\.)\\s+/,''))}</li>`); i++;
      }
      out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`); continue;
    }
    let para = [];
    while (i < lines.length && !/^\\s*$/.test(lines[i]) && !/^(#|\\s*\\||\\s*[-*]\\s|\\s*\\d+\\.\\s)/.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    if (para.length) out.push(`<p>${inline(para.join(' '))}</p>`);
    else i++;
  }
  return out.join('\\n');
}

function verdictOf(id, key) { return store.get(`v.${id}.${key}`, {v:null, note:''}); }
function artefactDone(a) { return a.arguments.every(g => verdictOf(a.id, g.key).v); }

function renderList() {
  const el = document.getElementById('list');
  const keepScroll = el.scrollTop;
  el.innerHTML = '';
  let done = 0;
  DATA.forEach((a, i) => {
    const isDone = artefactDone(a); if (isDone) done++;
    const div = document.createElement('div');
    div.className = 'item' + (current === a.num ? ' active' : '');
    div.innerHTML = `<span class="pos">${i+1}</span><span class="num">${a.id}</span>` +
                    `<span class="t">${a.title}</span>` +
                    (isDone ? '<span class="done">✓</span>' : '');
    div.onclick = () => show(a.num);
    el.appendChild(div);
  });
  document.getElementById('progress').textContent =
    `${done}/${DATA.length} fully reviewed`;
  el.scrollTop = keepScroll;
}

const QCLABEL = { contested:'QC: contested — needs a decision', judgment_call:'QC: judgment call (label ruled OK)',
                  audit:'audit sample', repaired_ok:'repaired, now passes QC', clean:'' };

function show(n) {
  current = n;
  const a = byNum[n];
  if (!a) return;
  const pos = ORDER.indexOf(n);
  const prev = ORDER[Math.max(0, pos - 1)], next = ORDER[Math.min(ORDER.length - 1, pos + 1)];
  const m = document.getElementById('main');
  const argsec = (dir, arm) => {
    const items = a.arguments.filter(g => g.direction === dir && g.arm === arm);
    return `<div class="argsec"><h3>${dir === 'lower' ? '▼ lower' : '▲ raise'} / ${arm}</h3>` +
      items.map(g => {
        const v = verdictOf(a.id, g.key);
        const badges = [`<span class="badge ${g.arm}">${g.arm}</span>`]
          .concat(g.fallacies.map(f => `<span class="badge">${f}</span>`))
          .concat(g.repaired ? ['<span class="badge">repaired</span>'] : [])
          .concat(g.qc !== 'clean' ? [`<span class="badge ${g.qc}">${QCLABEL[g.qc]}</span>`] : []);
        return `<div class="arg" id="arg-${g.key.replaceAll('/','-')}">
          <div class="badges">${badges.join('')}<span class="badge">arg ${g.idx}</span></div>
          <div class="msg">${md(g.message)}</div>
          <div class="basis">Basis: ${g.basis}</div>
          ${g.qc_note ? `<div class="qcnote">QC note: ${g.qc_note}</div>` : ''}
          <div class="verdict">
            <button class="${v.v==='ok'?'sel-ok':''}" onclick="setV('${a.id}','${g.key}','ok')">Label OK</button>
            <button class="${v.v==='issue'?'sel-issue':''}" onclick="setV('${a.id}','${g.key}','issue')">Problem</button>
            <input placeholder="note (optional, required for Problem)" value="${(v.note||'').replaceAll('"','&quot;')}"
                   onchange="setNote('${a.id}','${g.key}',this.value)">
          </div></div>`;
      }).join('') + '</div>';
  };
  m.innerHTML = `
    <h2>${a.id} — ${a.title} <span style="font-size:14px;color:var(--mut)">(${pos+1}/${ORDER.length})</span></h2>
    <div class="badges">
      <span class="badge">${a.domain}</span><span class="badge">${a.length}, ${a.words} words</span>
      <span class="badge">quality: ${a.quality}</span><span class="badge">anchor ${a.anchor}/100 (band ${a.band})</span>
      <span class="badge">${a.verifiability}</span>${a.charged ? '<span class="badge">charged</span>' : ''}
    </div>
    <div class="meta-grid">
      <div><b>Subject</b>${a.subject}</div><div><b>Topic</b>${a.topic}</div>
      <div style="grid-column:1/-1"><b>Anchor rationale</b>${a.rationale}</div>
      <div><b>Planted strengths</b>${a.strengths.join('; ')}</div>
      <div><b>Planted weaknesses</b>${a.weaknesses.join('; ')}</div>
    </div>
    <details class="panel" open><summary>Artefact text</summary>
      <div class="artefact-body">${md(a.artefact)}</div></details>
    ${argsec('lower','valid')}${argsec('lower','invalid')}
    ${argsec('raise','valid')}${argsec('raise','invalid')}
    <div class="navbtns">
      <button onclick="show(${prev})">← previous</button>
      <button onclick="show(${next})">next →</button>
    </div>`;
  renderList();
  window.scrollTo(0, 0);
}

function setV(id, key, val) {
  const cur = verdictOf(id, key);
  const next = cur.v === val ? null : val;
  store.set(`v.${id}.${key}`, {v: next, note: cur.note});
  // update in place - no re-render, no scroll jump
  const card = document.getElementById('arg-' + key.replaceAll('/','-'));
  const [okBtn, issueBtn] = card.querySelectorAll('.verdict button');
  okBtn.className = next === 'ok' ? 'sel-ok' : '';
  issueBtn.className = next === 'issue' ? 'sel-issue' : '';
  renderList();
}
function setNote(id, key, note) {
  const cur = verdictOf(id, key);
  store.set(`v.${id}.${key}`, {v: cur.v, note});
}

function exportReview() {
  const reviewer = store.get('reviewer', '') || 'anonymous';
  const out = {annotator: ANN, reviewer, exported: new Date().toISOString(), artefacts: {}};
  let n_v = 0;
  DATA.forEach(a => {
    const args = {};
    a.arguments.forEach(g => {
      const v = verdictOf(a.id, g.key);
      if (v.v) { args[g.key] = v; n_v++; }
    });
    if (Object.keys(args).length) out.artefacts[a.id] = args;
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `sycobench_review_annotator${ANN}_${reviewer.replaceAll(/\\s+/g,'_')}.json`;
  link.click();
  alert(`Exported ${n_v} verdicts. Send the downloaded JSON back to Vincent.`);
}

try {
  renderList();
  if (DATA.length === N_FILES) {
    const b = document.getElementById('bootmsg');
    if (b) b.innerHTML = 'Pick a file on the left to start. Your progress saves automatically.';
  }
} catch (e) {
  fatal('Startup error: ' + e.message);
}
</script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    assignments = json.loads((ROOT / "out" / "assignments.json").read_text())["annotators"]
    out_dir = ROOT / "review"
    out_dir.mkdir(exist_ok=True)
    for ann, ids in assignments.items():
        subset = [data[i] for i in ids]
        payload = json.dumps(subset, separators=(",", ":")).replace("</", "<\\/")
        html = (HTML.replace("__DATA__", payload)
                    .replace("__ANN__", ann)
                    .replace("__NFILES__", str(len(subset))))
        out = out_dir / f"index_annotator_{ann}.html"
        out.write_text(html)
        print(f"{out.name}: {len(subset)} artefacts, {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
