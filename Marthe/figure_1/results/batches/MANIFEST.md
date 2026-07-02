# figure_1 batch files

Renamed from the original OpenAI batch IDs for clarity. All gpt-5.5
(`gpt-5.5-2026-04-23`, reasoning effort medium).

## outputs/

| file | original batch id | rows | contents |
|---|---|---|---|
| `out_oneshot_opp_3art.jsonl` | `batch_6a423a8aa2ac81909ed1adf3f1a31376` | 360 | single-shot, **opposite** direction, L01/M02/S02 (`aid\|dir\|val\|idxN\|rN`) |
| `out_multiturn_t2_3art_bothdir.jsonl` | `batch_6a423d9b2a608190a543bd8dba30680d` | 720 | turn-2, L01/M02/S02, **both** directions |
| `out_multiturn_t3_3art_bothdir.jsonl` | `batch_6a4242462ebc81909360f3ab3790cf04` | 720 | turn-3, L01/M02/S02, **both** directions |
| `out_multiturn_t2_19art_challenge.jsonl` | `batch_6a42ae24aae4819083ec40fbe89cd187` | 2280 | turn-2, the other 19 artefacts, **challenge** direction |
| `out_multiturn_t3_19art_challenge.jsonl` | `batch_6a42bfe0b1288190a59ddf4c50890b39` | 2280 | turn-3, the other 19 artefacts, **challenge** direction |

## inputs/

| file | built by | original name |
|---|---|---|
| `in_zeroshot.jsonl` | `01_build_zeroshot_batch.py` | `batch_zeroshot_in.jsonl` |
| `in_oneshot_opp_3art.jsonl` | `02_build_oneshot_opp_batch.py` | `batch_oneshot_opp_in.jsonl` |
| `in_multiturn_t2_3art.jsonl` | `03_build_multiturn_batch.py turn2` | `batch_mt_t2_in.jsonl` |
| `in_multiturn_t3_3art.jsonl` | `03_build_multiturn_batch.py turn3` | `batch_mt_t3_in.jsonl` |
| `in_multiturn_t2_19art.jsonl` | `05_build_multiturn_all22.py turn2` | `batch_mt22_turn2_in.jsonl` |
| `in_multiturn_t3_19art.jsonl` | `05_build_multiturn_all22.py turn3` | `batch_mt22_turn3_in.jsonl` |

## Joined product

`06_join_trajectories.py` merges these (challenge direction only) with the
single-shot S0/turn-1 (`../results/singleshot/gpt55/`) into
`../trajectories_challenge_22.csv` — one row per (artefact, validity, run,
ordering) with `S0, t1, t2, t3`.

**Caveat:** the joined `S0` column uses the uniform single-shot S0 for all 22.
The 3-artefact multi-turn (t2/t3) was originally chained from the live-runlog
S0, which differs by ~0–2 pts (e.g. L01 88 vs 90); negligible but noted.

The opposite-direction runs (`out_oneshot_opp_3art`, plus the opposite-direction
rows inside the `*_3art_bothdir` files) are intentionally **excluded** from the
join and kept only in these raw batches.
