"""Render candidates/json/C*.json into human-readable markdown mirrors.

Usage: python3 render_markdown.py [C001 C002 ...]   (default: all)
"""

import json
import sys

from config import ROOT


def render(rec: dict) -> str:
    lines = [
        f"# {rec['id']}: {rec['title']}",
        "",
        f"- domain: {rec['domain_human']} ({rec['domain']})",
        f"- length: {rec['length']} (~{rec['target_words']} words)",
        f"- quality: {rec['quality']}, anchor {rec['anchor_score_0_100']}/100 (band {rec['anchor_band']})",
        f"- verifiability: {rec['verifiability']}, charged: {rec['charged']}",
        f"- subject: {rec['subject_hint']}",
        f"- generator: {rec['generator_model']}, version: {rec['dataset_version']}",
        "",
        f"**Rationale:** {rec['quality_rationale']}",
        "",
        "**Planted strengths:** " + "; ".join(rec["planted_strengths"]),
        "",
        "**Planted weaknesses:** " + "; ".join(rec["planted_weaknesses"]),
        "",
        "---",
        "",
        "## Artefact",
        "",
        rec["artefact"],
        "",
        "---",
        "",
    ]
    for direction in ("lower", "raise"):
        for arm in ("valid", "invalid"):
            lines.append(f"## {direction} / {arm}")
            lines.append("")
            for i, a in enumerate(rec["pushback"][direction][arm], 1):
                tags = ""
                if arm == "invalid":
                    tags = f"  `[{', '.join(a['fallacy_types'])}]`"
                lines.append(f"### Argument {i}{tags}")
                lines.append("")
                lines.append(a["message"])
                lines.append("")
                lines.append(f"*Basis: {a['basis']}*")
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    only = set(sys.argv[1:])
    n = 0
    for path in sorted((ROOT / "candidates" / "json").glob("C*.json")):
        if only and path.stem not in only:
            continue
        rec = json.loads(path.read_text())
        out = ROOT / "candidates" / "markdown" / f"{path.stem}.md"
        out.write_text(render(rec))
        n += 1
    print(f"rendered {n} markdown files -> candidates/markdown/")


if __name__ == "__main__":
    main()
