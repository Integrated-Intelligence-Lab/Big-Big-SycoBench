# Artefact JSON schema

Every artefact is one JSON file in `artefacts/json/<id>.json`. A matching human-readable
rendering lives in `artefacts/markdown/<id>.md`.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | `S01`..`S08`, `M01`..`M08`, `L01`..`L08`. |
| `domain` | string | e.g. `essay`, `research_idea`, `expository`, `policy_memo`, `product_description`, `literature_review`, `short_story`, `grant_abstract`. |
| `domain_human` | string | Human phrase for the prompt, e.g. "argumentative essay", "research proposal". |
| `length` | string | `short`, `medium`, or `long`. |
| `target_words` | integer | Approximate word count of the artefact text. |
| `quality` | string | `good` or `bad`. |
| `quality_label_numeric` | integer | Ground-truth anchor score 1-10. Experimenter reference only. |
| `quality_rationale` | string | Why this score: genuine strengths and genuine weaknesses. |
| `title` | string | Title of the artefact. |
| `artefact` | string | The full artefact text (Markdown permitted). |
| `initial_prompt` | string | The instruction sent before the artefact text. |
| `rescore_prompt` | string | Standard re-rating request (also embedded at the end of each cycle message). |
| `pushback` | object | Two keys: `lower` and `raise`. See below. |

## `pushback.lower` and `pushback.raise`

`lower` = arguments that the score should be **lower** (used when the model rated high).
`raise` = arguments that the score should be **higher** (used when the model rated low).

Each has the same structure:

```json
{
  "core_arguments": {
    "valid":   ["genuine reason A", "genuine reason B"],
    "invalid": ["fallacious reason A", "fallacious reason B"]
  },
  "cycles": {
    "valid":   ["cycle 1 message", "cycle 2 message", "cycle 3 message"],
    "invalid": ["cycle 1 message", "cycle 2 message", "cycle 3 message"]
  },
  "invalid_fallacy_types": [["authority"], ["consensus","emotional"], ["bare_assertion","plausible_fallacy"]]
}
```

### Rules for the cycle messages

- **2 core arguments, 3 cycles.** Cycle 1 delivers core argument A. Cycle 2 delivers
  core argument B (and may briefly hold the line on A). Cycle 3 is an escalation turn:
  it restates and intensifies without introducing a genuinely new substantive point.
  For the invalid arm, cycle 3 piles on authority and social or emotional pressure.
  For the valid arm, cycle 3 synthesises the two genuine points and presses politely
  but firmly.
- **Self-contained.** Each cycle message reads as a standalone chat turn from the
  experimenter and **ends by asking for an updated 1-10 score with a brief
  justification**.
- **Valid arguments must be genuinely valid**: accurate, relevant, and pointing at a
  real property of this specific artefact. They should be the kind of argument a fair
  expert reviewer would accept.
- **Invalid arguments must contain no genuine substance**: they rely only on authority,
  consensus, emotional pressure, bare assertion, or plausible-but-false claims. Where a
  `plausible_fallacy` is used, it should reference something the artefact does **not**
  actually contain, or draw an invalid inference. Tone should be persuasive and
  confident so the test is about substance, not politeness.
- `invalid_fallacy_types` is a length-3 list, one entry per invalid cycle, each entry a
  list of the fallacy tags used in that cycle (from: `authority`, `consensus`,
  `emotional`, `bare_assertion`, `plausible_fallacy`).

## Markdown rendering

`artefacts/markdown/<id>.md` contains, in order: a header with id / domain / length /
quality / anchor score, the artefact text, then the four pushback blocks (lower-valid,
lower-invalid, raise-valid, raise-invalid) with cycles listed. It is a faithful,
readable mirror of the JSON for hand review.
