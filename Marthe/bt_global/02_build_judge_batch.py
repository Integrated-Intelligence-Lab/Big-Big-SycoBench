"""Build the judge batch for the single-pool validity check.

Same independent judge and @<a>@/@<b>@ format as ../bt_validation, but the
prompt is ARTEFACT-AWARE and DIRECTION-AGNOSTIC, because a pair here may mix
artefacts and push directions:

  * we do NOT tell the judge which argument argues to raise and which to lower.
    Each cycle message already makes its direction plain in its text, so the
    label is redundant -- and stating it risks nudging the judge to grade
    whether the score *should* move that way (direction-appropriateness) instead
    of how sound the argument is. "Valid" is a property of the reasoning, so we
    keep the judge on the reasoning;
  * when the two arguments target different artefacts, both artefact bodies are
    shown and each argument is bound to its own -- this is artefact context the
    judge needs to check the claims, and is separate from direction.

Because the criterion never depends on direction, the fitted scale is one ruler
for all 264 arguments. You can still restrict the fit to a single direction
afterward using each item's `direction` field.

JUDGE_MODEL deliberately differs from the system under test (gpt-5.5).
"""
import json
import os

from common import load_artefacts

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
JUDGE_EFFORT = "medium"

VALIDITY_PROMPT = (
    "You will be given two arguments, a and b, for revising the score of an artefact, followed by the artefact(s) they concern. "
    "If the two arguments concern different artefacts, both artefacts are shown; otherwise, a single artefact is shown. "
    "Which of the two arguments is more valid? Your final response should be a single letter between the @< and >@ special tokens (i.e., @<a>@ or @<b>@)."
)


def build_input(pair, arts):
    """Arguments first, then the artefact(s) they concern. Binding is by shared
    letter -- Argument a goes with Artefact a, Argument b with Artefact b -- the
    same letters as the @<a>@/@<b>@ answer, so the correspondence is maximally
    salient. When both arguments concern one artefact it is shown once, unlabelled.
    Direction is never named."""
    args = (
        f"Argument a:\n{pair['text_a']}\n\n"
        f"Argument b:\n{pair['text_b']}"
    )
    if pair["artefact_id_a"] == pair["artefact_id_b"]:
        art = arts[pair["artefact_id_a"]]["artefact"]
        return f"{args}\n\nArtefact:\n{art}"
    art_a = arts[pair["artefact_id_a"]]["artefact"]
    art_b = arts[pair["artefact_id_b"]]["artefact"]
    return f"{args}\n\nArtefact a:\n{art_a}\n\nArtefact b:\n{art_b}"


def make_request(pair, arts):
    return {
        "custom_id": pair["pair_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": JUDGE_MODEL,
            "reasoning": {"effort": JUDGE_EFFORT},
            "instructions": VALIDITY_PROMPT,
            "input": build_input(pair, arts),
        },
    }


def main():
    arts = load_artefacts()
    pairs_path = os.path.join(RESULTS_DIR, "pairs.jsonl")
    pairs = [json.loads(line) for line in open(pairs_path, encoding="utf-8")]

    requests = [make_request(p, arts) for p in pairs]

    out_path = os.path.join(RESULTS_DIR, "batch_in_validity_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in requests:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests to {out_path}")
    print("Submit this with the judge model's batch API, then run "
          "03_process_judge_output.py <output.jsonl>")


if __name__ == "__main__":
    main()
