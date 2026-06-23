"""Parse the judge model's batch output into a_wins booleans and attach them
to the pairs built by 01_build_pairs.py.

Usage:
    python 03_process_judge_output.py <judge_batch_output.jsonl>
"""
import argparse
import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


def parse_winner(text):
    m = re.search(r"@<\s*([ab])\s*>@", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([ab])\b", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_output", help="batch output jsonl from the judge model")
    a = ap.parse_args()

    winners = {}
    unparsed = 0
    for line in open(a.judge_output, encoding="utf-8"):
        r = json.loads(line)
        text = extract_text(r["response"]["body"])
        w = parse_winner(text)
        if w is None:
            unparsed += 1
            continue
        winners[r["custom_id"]] = w
    if unparsed:
        print(f"WARNING: {unparsed} judge responses had no parseable a/b winner")

    pairs_path = os.path.join(RESULTS_DIR, "pairs.jsonl")
    out_path = os.path.join(RESULTS_DIR, "pairs_with_results.jsonl")
    n_missing = 0
    n_written = 0
    with open(pairs_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            p = json.loads(line)
            w = winners.get(p["pair_id"])
            if w is None:
                n_missing += 1
                continue
            p["a_wins"] = (w == "a")
            json.dump(p, fout, ensure_ascii=False)
            fout.write("\n")
            n_written += 1
    if n_missing:
        print(f"WARNING: {n_missing} pairs missing a judged result")
    print(f"Wrote {n_written} judged pairs to {out_path}")


if __name__ == "__main__":
    main()
