"""Analyze the returned annotation files and build an HTML summary.

Input:  ../../Annotations/*.json[l]   (all annotator exports)
Output: annotation_summary.html      (self-contained, theme-aware)
        annotation_analysis.json     (raw aggregates for further work)

Usage: python3 analyze_annotations.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANN_DIR = ROOT.parent / "Annotations"

sys.path.insert(0, str(ROOT))
from make_review_site import qc_annotations  # noqa: E402


# ---------------------------------------------------------------- loading

def load_annotations() -> tuple[dict, dict, list]:
    """votes[(sid, key)][ann] = {v, note}; reviewers[ann] = name; anomalies."""
    per_ann_files = defaultdict(list)
    for p in sorted(ANN_DIR.iterdir()):
        m = re.search(r"annotator(\d+)", p.name)
        if not m or p.suffix not in (".json", ".jsonl"):
            continue
        per_ann_files[m.group(1)].append(p)

    votes = defaultdict(dict)
    reviewers, anomalies = {}, []
    assignments = json.loads((ROOT / "out" / "assignments.json").read_text())["annotators"]

    for ann, files in sorted(per_ann_files.items()):
        # exports as (timestamp, sid, args, reviewer); later timestamp wins per artefact
        records = []
        for p in files:
            text = p.read_text()
            if p.suffix == ".jsonl":
                for line in text.strip().split("\n"):
                    d = json.loads(line)
                    records.append((d.get("exported", ""), d["id"],
                                    d["artefact"], d.get("reviewer", "")))
            else:
                d = json.loads(text)
                for sid, args in d["artefacts"].items():
                    records.append((d.get("exported", ""), sid, args,
                                    d.get("reviewer", "")))
        records.sort(key=lambda r: r[0])
        for exported, sid, args, reviewer in records:
            if reviewer and reviewer != "anonymous":
                reviewers[ann] = reviewer
            if sid not in assignments.get(ann, []):
                anomalies.append(f"annotator {ann}: verdicts for {sid}, "
                                 f"which is not in their assignment (kept)")
            for key, v in args.items():
                if v.get("v") in ("ok", "issue"):
                    votes[(sid, key)][ann] = {"v": v["v"],
                                              "note": (v.get("note") or "").strip()}
        reviewers.setdefault(ann, files[0].name)
    return votes, reviewers, anomalies


def load_metadata() -> dict:
    """(sid, key) -> {tier, domain, band, quality, arm, direction, fallacies, repaired, qc}."""
    qa = qc_annotations()
    meta = {}
    for p in sorted((ROOT / "candidates" / "json").glob("C*.json")):
        r = json.loads(p.read_text())
        for d in ("lower", "raise"):
            for arm in ("valid", "invalid"):
                for i, a in enumerate(r["pushback"][d][arm], 1):
                    key = f"{d}/{arm}/{i}"
                    q = qa.get((r["id"], key), {"status": "clean", "note": ""})
                    meta[(r["id"], key)] = {
                        "tier": r["length"], "domain": r["domain_human"],
                        "band": r["anchor_band"], "quality": r["quality"],
                        "title": r["title"],
                        "direction": d, "arm": arm,
                        "fallacies": a.get("fallacy_types", []),
                        "repaired": "repaired_from" in a,
                        "qc": q["status"],
                    }
    return meta


# ---------------------------------------------------------------- stats

def rate(n_issue: int, n: int) -> float:
    return round(100 * n_issue / n, 1) if n else 0.0


def krippendorff_alpha(units: list) -> float:
    """Binary alpha; units = list of (n_issue, n_ok) with n_issue+n_ok >= 2."""
    D_o = sum(i * o / (i + o - 1) for i, o in units)
    N = sum(i + o for i, o in units)
    NI = sum(i for i, _ in units)
    NO = N - NI
    D_e = NI * NO / (N - 1) if N > 1 else 0
    return round(1 - D_o / D_e, 3) if D_e else 1.0


def analyze():
    votes, reviewers, anomalies = load_annotations()
    meta = load_metadata()
    assignments = json.loads((ROOT / "out" / "assignments.json").read_text())["annotators"]

    # ---- per annotator
    per_ann = {}
    for ann, ids in sorted(assignments.items()):
        expected = {(sid, m_key) for sid in ids for (s2, m_key) in
                    [(sid, k) for (s, k) in meta if s == sid]}
        got = [(u, v[ann]) for u, v in votes.items() if ann in v]
        n_v = len(got)
        n_issue = sum(1 for _, x in got if x["v"] == "issue")
        n_notes = sum(1 for _, x in got if x["note"])
        done_art = Counter()
        for (sid, _), _x in got:
            done_art[sid] += 1
        complete = sum(1 for sid in ids if done_art.get(sid, 0) == 12)
        per_ann[ann] = {"reviewer": reviewers.get(ann, "?"), "assigned": len(ids) * 12,
                        "verdicts": n_v, "complete_artefacts": complete,
                        "n_issue": n_issue, "issue_pct": rate(n_issue, n_v),
                        "notes": n_notes}

    # ---- grouped problem rates
    def grouped(keyfn):
        g = defaultdict(lambda: [0, 0])
        for u, vs in votes.items():
            if u not in meta:
                continue
            for x in vs.values():
                k = keyfn(meta[u])
                g[k][1] += 1
                if x["v"] == "issue":
                    g[k][0] += 1
        return {k: {"issue": i, "n": n, "pct": rate(i, n)}
                for k, (i, n) in sorted(g.items(), key=lambda t: -t[1][0] / max(t[1][1], 1))}

    by_qc = grouped(lambda m: m["qc"])
    by_arm = grouped(lambda m: m["arm"])
    by_armdir = grouped(lambda m: f"{m['direction']}/{m['arm']}")
    by_tier = grouped(lambda m: m["tier"])
    by_band = grouped(lambda m: str(m["band"]))
    fal = defaultdict(lambda: [0, 0])
    for u, vs in votes.items():
        if u not in meta or meta[u]["arm"] != "invalid":
            continue
        for x in vs.values():
            for f in meta[u]["fallacies"]:
                fal[f][1] += 1
                if x["v"] == "issue":
                    fal[f][0] += 1
    by_fallacy = {k: {"issue": i, "n": n, "pct": rate(i, n)}
                  for k, (i, n) in sorted(fal.items(), key=lambda t: -t[1][0] / max(t[1][1], 1))}

    # ---- agreement on co-rated arguments
    multi = {u: vs for u, vs in votes.items() if len(vs) >= 2}
    agree_pairs = disagree_pairs = 0
    units_by_qc = defaultdict(list)
    units_all = []
    for u, vs in multi.items():
        vals = [x["v"] for x in vs.values()]
        i, o = vals.count("issue"), vals.count("ok")
        units_all.append((i, o))
        units_by_qc[meta[u]["qc"] if u in meta else "unknown"].append((i, o))
        agree_pairs += i * (i - 1) // 2 + o * (o - 1) // 2
        disagree_pairs += i * o
    total_pairs = agree_pairs + disagree_pairs
    agreement = {
        "co_rated_arguments": len(multi),
        "pairwise_agreement_pct": rate(agree_pairs, total_pairs),
        "krippendorff_alpha": krippendorff_alpha(units_all) if units_all else None,
        "by_qc": {k: {"agreement_pct": rate(
                        sum(i*(i-1)//2 + o*(o-1)//2 for i, o in us),
                        sum(i*(i-1)//2 + o*(o-1)//2 + i*o for i, o in us)),
                      "n_units": len(us)}
                  for k, us in sorted(units_by_qc.items())},
    }

    # ---- decisions per argument
    decisions = []
    for u, vs in votes.items():
        vals = [x["v"] for x in vs.values()]
        i, o = vals.count("issue"), vals.count("ok")
        if i == 0:
            continue
        m = meta.get(u, {})
        verdict = ("majority_issue" if i > o else
                   "tie" if i == o else "minority_issue")
        decisions.append({
            "sid": u[0], "key": u[1], "issue": i, "ok": o, "verdict": verdict,
            "qc": m.get("qc", "?"), "arm": m.get("arm", "?"),
            "notes": [f"{per_ann[a]['reviewer']}: {x['note']}"
                      for a, x in sorted(vs.items()) if x["v"] == "issue" and x["note"]][:3],
        })
    order = {"majority_issue": 0, "tie": 1, "minority_issue": 2}
    decisions.sort(key=lambda d: (order[d["verdict"]], -d["issue"], d["sid"]))
    n_maj = sum(1 for d in decisions if d["verdict"] == "majority_issue")
    n_tie = sum(1 for d in decisions if d["verdict"] == "tie")

    artefact_notes = []
    for u, vs in votes.items():
        for a, x in vs.items():
            if x["note"].upper().startswith("ARTEFACT:"):
                artefact_notes.append({"sid": u[0], "by": per_ann[a]["reviewer"],
                                       "note": x["note"]})

    # ---- the 11 contested arguments: human outcome
    contested_out = []
    for u, vs in votes.items():
        if meta.get(u, {}).get("qc") == "contested":
            vals = [x["v"] for x in vs.values()]
            contested_out.append({"sid": u[0], "key": u[1],
                                  "issue": vals.count("issue"), "ok": vals.count("ok")})
    contested_out.sort(key=lambda d: -d["issue"])

    total_v = sum(p["verdicts"] for p in per_ann.values())
    total_i = sum(p["n_issue"] for p in per_ann.values())
    return {
        "per_annotator": per_ann, "anomalies": anomalies,
        "totals": {"verdicts": total_v, "issues": total_i,
                   "issue_pct": rate(total_i, total_v),
                   "arguments_covered": len(votes),
                   "majority_issue": n_maj, "ties": n_tie},
        "by_qc": by_qc, "by_arm": by_arm, "by_armdir": by_armdir,
        "by_tier": by_tier, "by_band": by_band, "by_fallacy": by_fallacy,
        "agreement": agreement, "decisions": decisions,
        "artefact_notes": artefact_notes, "contested_outcomes": contested_out,
    }


# ---------------------------------------------------------------- html

def svg_hbar(items, color_by=None, width=640, unit="%") -> str:
    """Horizontal bars: items = [(label, value, sublabel)], values in 0-100."""
    bh, gap, lw = 26, 10, 170
    h = len(items) * (bh + gap) + 8
    vmax = max((v for _, v, _ in items), default=1) or 1
    bars = []
    for r, (label, v, sub) in enumerate(items):
        y = 4 + r * (bh + gap)
        w = max(3, (width - lw - 90) * v / vmax)
        c = (color_by or (lambda _l: "var(--series-1)"))(label)
        bars.append(
            f'<text x="{lw-8}" y="{y+bh/2+4}" text-anchor="end" class="lbl">{label}</text>'
            f'<rect x="{lw}" y="{y}" width="{w:.0f}" height="{bh}" rx="4" fill="{c}">'
            f'<title>{label}: {v}{unit} ({sub})</title></rect>'
            f'<text x="{lw+w+8:.0f}" y="{y+bh/2+4}" class="val">{v}{unit}'
            f' <tspan class="sub">({sub})</tspan></text>')
    return (f'<svg viewBox="0 0 {width} {h}" role="img" '
            f'style="max-width:{width}px;width:100%;height:auto">{"".join(bars)}</svg>')


def build_html(a: dict) -> str:
    t = a["totals"]

    def bar_items(g, label_map=None):
        return [((label_map or {}).get(k, k), v["pct"], f'{v["issue"]}/{v["n"]}')
                for k, v in g.items()]

    qc_labels = {"contested": "contested (11 args)", "judgment_call": "judgment call",
                 "audit": "audit sample", "repaired_ok": "repaired, QC-pass",
                 "clean": "clean (no badge)"}
    arm_color = lambda l: "var(--series-1)" if "valid" in l and "invalid" not in l else "var(--series-2)"

    ann_rows = "".join(
        f"<tr><td>{ann}</td><td>{p['reviewer']}</td>"
        f"<td>{p['complete_artefacts']}/61</td>"
        f"<td>{p['verdicts']}/{p['assigned']}</td>"
        f"<td>{p['n_issue']} ({p['issue_pct']}%)</td><td>{p['notes']}</td></tr>"
        for ann, p in a["per_annotator"].items())

    dec_rows = "".join(
        f"<tr><td>{d['sid']}</td><td><code>{d['key']}</code></td>"
        f"<td>{d['arm']}</td><td>{d['issue']}–{d['ok']}</td><td>{d['qc']}</td>"
        f"<td>{'<br>'.join(d['notes']) or ''}</td></tr>"
        for d in a["decisions"] if d["verdict"] in ("majority_issue", "tie"))

    minority = [d for d in a["decisions"] if d["verdict"] == "minority_issue"]
    min_rows = "".join(
        f"<tr><td>{d['sid']}</td><td><code>{d['key']}</code></td>"
        f"<td>{d['arm']}</td><td>{d['issue']}–{d['ok']}</td><td>{d['qc']}</td>"
        f"<td>{'<br>'.join(d['notes']) or ''}</td></tr>" for d in minority)

    cont_rows = "".join(
        f"<tr><td>{c['sid']}</td><td><code>{c['key']}</code></td>"
        f"<td>{c['issue']} issue / {c['ok']} ok</td></tr>"
        for c in a["contested_outcomes"])

    art_rows = "".join(
        f"<tr><td>{n['sid']}</td><td>{n['by']}</td><td>{n['note']}</td></tr>"
        for n in a["artefact_notes"]) or '<tr><td colspan="3">none</td></tr>'

    band_labels = {b: f"band {b}" for b in ("15", "30", "50", "70", "85")}
    agr = a["agreement"]
    agr_items = [(qc_labels.get(k, k), v["agreement_pct"], f'{v["n_units"]} args')
                 for k, v in agr["by_qc"].items()]

    anomalies = "".join(f"<li>{x}</li>" for x in a["anomalies"]) or "<li>none</li>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SycoBench annotation summary</title>
<style>
.viz-root {{ color-scheme:light;
  --surface-1:#fcfcfb; --surface-2:#f1f1ef; --line:#e2e1dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e;
  --series-1:#2a78d6; --series-2:#eb6834; --good:#0ca30c; --critical:#d03b3b; }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme:dark;
    --surface-1:#1a1a19; --surface-2:#242423; --line:#3a3a38;
    --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --series-1:#3987e5; --series-2:#d95926; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#242423; --line:#3a3a38;
  --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --series-1:#3987e5; --series-2:#d95926; }}
.viz-root {{ margin:0; background:var(--surface-1); color:var(--text-primary);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:30px 24px 80px; }}
h1 {{ font-size:24px; margin:0 0 4px; }}
h2 {{ font-size:18px; margin:34px 0 8px; }}
.sub {{ color:var(--text-secondary); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px; margin:20px 0; }}
.tile {{ background:var(--surface-2); border-radius:12px; padding:14px 16px; }}
.tile .n {{ font-size:30px; font-weight:700; }}
.tile .l {{ font-size:12px; color:var(--text-secondary); }}
svg .lbl {{ font-size:13px; fill:var(--text-primary); }}
svg .val {{ font-size:13px; fill:var(--text-primary); font-weight:600; }}
svg .val .sub {{ font-weight:400; fill:var(--text-secondary); }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
  vertical-align:top; }}
th {{ color:var(--text-secondary); font-weight:600; }}
code {{ background:var(--surface-2); padding:1px 5px; border-radius:5px; font-size:12.5px; }}
details {{ margin:10px 0; }} summary {{ cursor:pointer; font-weight:600; }}
p.note {{ color:var(--text-secondary); font-size:13.5px; }}
</style></head>
<body class="viz-root"><main>
<h1>SycoBench scale-v1 — annotation summary</h1>
<p class="sub">11 annotators · 450 artefacts · human check of all 5,400 argument labels
· generated 2026-09-03</p>

<div class="tiles">
  <div class="tile"><div class="n">{t['verdicts']:,}</div><div class="l">verdicts collected</div></div>
  <div class="tile"><div class="n">{t['issue_pct']}%</div><div class="l">flagged as Problem ({t['issues']:,})</div></div>
  <div class="tile"><div class="n">{agr['pairwise_agreement_pct']}%</div><div class="l">pairwise agreement on {agr['co_rated_arguments']:,} co-rated args</div></div>
  <div class="tile"><div class="n">{agr['krippendorff_alpha']}</div><div class="l">Krippendorff's α (binary)</div></div>
  <div class="tile"><div class="n">{t['majority_issue']}</div><div class="l">majority-Problem arguments</div></div>
  <div class="tile"><div class="n">{t['ties']}</div><div class="l">tied verdicts</div></div>
</div>

<h2>Problem rate by pipeline QC category</h2>
<p class="note">Does human judgement track the automated QC? It should be highest for
contested, mid for judgment calls, low for clean/audit.</p>
{svg_hbar(bar_items(a['by_qc'], qc_labels))}

<h2>Problem rate by direction and arm</h2>
<p class="note">Blue = valid arms, orange = invalid arms.</p>
{svg_hbar(bar_items(a['by_armdir']), color_by=arm_color)}

<h2>Problem rate by fallacy family (invalid arms)</h2>
{svg_hbar(bar_items(a['by_fallacy']), color_by=lambda _:"var(--series-2)")}

<h2>Problem rate by tier and anchor band</h2>
{svg_hbar(bar_items(a['by_tier']))}
{svg_hbar(bar_items(a['by_band'], band_labels))}

<h2>Inter-annotator agreement by QC category</h2>
<p class="note">Pairwise agreement among co-rated arguments (overlap pool).</p>
{svg_hbar(agr_items)}

<h2>Per annotator</h2>
<div class="tablewrap"><table>
<tr><th>#</th><th>Reviewer</th><th>Artefacts complete</th><th>Verdicts</th>
<th>Problems</th><th>Notes</th></tr>
{ann_rows}
</table></div>

<h2>Action list: majority-Problem and tied arguments</h2>
<p class="note">These need a fix or drop before freezing v1. Ordered by vote margin;
sample issue notes shown.</p>
<div class="tablewrap"><table>
<tr><th>Artefact</th><th>Argument</th><th>Arm</th><th>issue–ok</th><th>QC</th><th>Notes</th></tr>
{dec_rows}
</table></div>

<details><summary>Minority-Problem arguments ({len(minority)}) — flagged by one
reviewer, outvoted or single-rated context</summary>
<div class="tablewrap"><table>
<tr><th>Artefact</th><th>Argument</th><th>Arm</th><th>issue–ok</th><th>QC</th><th>Notes</th></tr>
{min_rows}
</table></div></details>

<h2>The 11 QC-contested arguments — human outcome</h2>
<div class="tablewrap"><table>
<tr><th>Artefact</th><th>Argument</th><th>Votes</th></tr>
{cont_rows}
</table></div>

<h2>Artefact-level flags ("ARTEFACT:" notes)</h2>
<div class="tablewrap"><table>
<tr><th>Artefact</th><th>By</th><th>Note</th></tr>
{art_rows}
</table></div>

<h2>Data hygiene</h2>
<ul>{anomalies}</ul>
</main></body></html>"""


def main() -> None:
    a = analyze()
    (HERE / "annotation_analysis.json").write_text(json.dumps(a, indent=2))
    (HERE / "annotation_summary.html").write_text(build_html(a))
    t = a["totals"]
    print(f"verdicts {t['verdicts']}, problem rate {t['issue_pct']}%, "
          f"agreement {a['agreement']['pairwise_agreement_pct']}%, "
          f"alpha {a['agreement']['krippendorff_alpha']}")
    print(f"majority-issue {t['majority_issue']}, ties {t['ties']}, "
          f"artefact notes {len(a['artefact_notes'])}, anomalies {len(a['anomalies'])}")
    print(f"-> {HERE/'annotation_summary.html'}")


if __name__ == "__main__":
    main()
