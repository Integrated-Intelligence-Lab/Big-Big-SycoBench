# scale-v1 generation run report (2026-07-13)

## Dataset produced

- **450 candidate artefacts** in `candidates/json/` (+ markdown mirrors, index.json).
  Tiers: {'short': 180, 'medium': 180, 'long': 90}. Anchor bands: {15: 83, 30: 97, 50: 50, 70: 115, 85: 105}.
- Each: artefact + 2 directions x 2 arms x 3 standalone arguments (5,400 arguments),
  fallacy-tagged, 0-100 scale, per the ads_v2 protocol.
- Generator: gpt-5.6-sol (Batch API, high reasoning effort). Canary GUID embedded.
- Every repaired argument keeps its original in `repaired_from`.

## QC pipeline and flag trajectory

1. Adversarial QC round 1: **1,166 / 4,956 arguments flagged (23.5%)**
   (invalid args accidentally crediting real properties; valid args overstating).
2. Feedback repair of all flagged -> QC round 2: **412 / 5,400 (7.6%)**.
   Only 50 were repeat flags; 362 fresh flips on previously-passed args
   => attacker boundary noise ~9%/pass; blind iteration would not converge.
3. Adjudication of all 412: **104 must_fix / 308 judgment_call**.
4. Targeted repair of the 104 -> verification QC: **93 pass, 11 still flagged**.

Stop rule applied: no further automated rounds; residue goes to human review.

## Human review queue (`out/review_queue_final.md`)

- A: 11 twice-repaired, still-flagged arguments (fix or drop by hand).
- B: 308 adjudicated judgment calls (spot-check; rulings in adjudications.json).
- C: 506 random audit sample (10% of passes).

## Spend (ledger)

| stage | in | out | cost |
| artefacts | 0.26M | 1.53M | $23.67 |
| artefacts | 0.02M | 0.15M | $2.28 |
| pushbacks | 1.10M | 1.39M | $23.62 |
| artefacts | 0.01M | 0.06M | $0.93 |
| pushbacks | 0.06M | 0.07M | $1.22 |
| qc | 2.12M | 1.36M | $25.64 |
| pushbacks | 0.04M | 0.05M | $0.90 |
| repair | 2.34M | 0.44M | $12.47 |
| qc | 2.13M | 1.18M | $22.96 |
| adjudicate | 0.85M | 0.15M | $4.30 |
| repair | 0.21M | 0.03M | $1.05 |
| qc | 0.45M | 0.24M | $4.71 |

**Total: $123.75** of $500 cap.

## Next steps

1. Human pass over review_queue_final.md (A is mandatory, ~1h; B/C sampling).
2. Pilot screen: initial-score runs (~10) on 2-3 subject models over all 450;
   drop artefacts with mean initial score within 10 points of 50 for any pilot
   model; select final 300 filling the anchor-band targets (40/40/20 tiers).
3. Freeze v1 + changelog; screened-out candidates seed the private held-out pool.
4. Evaluation side: BT judging of the argument pool, then the ADS runs.
