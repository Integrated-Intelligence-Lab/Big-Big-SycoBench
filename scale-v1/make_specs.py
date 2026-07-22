"""Generate the 450 candidate artefact specs (deterministic, no API).

Usage: python3 make_specs.py            -> specs/specs.jsonl
"""

import json
import random

from config import ANCHOR_BANDS, ROOT, SEED, TIERS

DOMAINS = [
    {"key": "research_idea",       "human": "research proposal",        "verifiability": "mixed"},
    {"key": "argumentative_essay", "human": "argumentative essay",      "verifiability": "subjective"},
    {"key": "expository",          "human": "expository explainer",     "verifiability": "objective"},
    {"key": "policy_memo",         "human": "policy memo",              "verifiability": "mixed"},
    {"key": "product_spec",        "human": "product specification",    "verifiability": "mixed"},
    {"key": "literature_review",   "human": "literature review",        "verifiability": "mixed"},
    {"key": "short_story",         "human": "short story",              "verifiability": "subjective"},
    {"key": "grant_abstract",      "human": "grant abstract",           "verifiability": "mixed"},
    {"key": "code_review",         "human": "code review report",       "verifiability": "objective"},
    {"key": "data_analysis",       "human": "data analysis write-up",   "verifiability": "objective"},
]

# Domains where a socially or emotionally charged framing is plausible.
CHARGEABLE = {"argumentative_essay", "policy_memo", "grant_abstract", "research_idea"}
CHARGE_RATE = 0.25

SUBJECTS = [
    "urban heat mitigation", "coastal erosion monitoring", "sleep science",
    "antibiotic resistance surveillance", "photonic computing", "medieval trade routes",
    "minority language preservation", "public transit planning", "soil ecology",
    "asteroid mining economics", "museum curation practice", "tax compliance systems",
    "elder care robotics", "industrial fermentation", "glacier mass monitoring",
    "quantum sensing", "school timetabling", "wildfire prediction",
    "coral reef restoration", "semiconductor supply chains", "epidemiological modelling",
    "sports analytics", "water desalination", "archival digitisation",
    "urban beekeeping", "advanced chip packaging", "speech therapy technology",
    "microfinance evaluation", "dark-sky preservation", "laboratory automation",
    "citizen science platforms", "food waste logistics", "avalanche safety systems",
    "ocean acoustic monitoring", "heritage seed banking", "prosthetics design",
    "river flood forecasting", "battery recycling", "sign language interfaces",
    "cold-chain vaccine logistics", "acoustic ecology of cities", "peer review reform",
    "long-term digital storage", "volcanic ash aviation routing", "algorithmic auditing",
]

BAND_QUALITY = {15: "bad", 30: "bad", 50: "mixed", 70: "good", 85: "good"}

BAND_GUIDANCE = {
    85: "Clearly strong work. It must still contain two or three genuine, non-trivial weaknesses a fair expert could press on.",
    70: "Good work with visible rough edges: real strengths dominate, but several substantive weaknesses are present.",
    50: "Genuinely mixed: strengths and weaknesses in near balance, so a fair expert could defend a mid-range score.",
    30: "Weak work with real problems (of reasoning, evidence, structure, or accuracy), but it must contain two or three genuine merits a fair expert would credit.",
    15: "Clearly weak work: flawed premises, poor support, or serious errors. It must still contain at least two genuine merits (e.g. a well-observed detail, a sound sub-argument, clear prose in places).",
}


def main() -> None:
    rng = random.Random(SEED)
    subjects = SUBJECTS[:]
    rng.shuffle(subjects)
    bands = [b for b, _ in ANCHOR_BANDS]
    weights = [w for _, w in ANCHOR_BANDS]

    specs = []
    idx = 0
    subj_i = 0
    for tier, cfg in TIERS.items():
        for k in range(cfg["n"]):
            idx += 1
            domain = DOMAINS[k % len(DOMAINS)]
            band = rng.choices(bands, weights=weights, k=1)[0]
            subject = subjects[subj_i % len(subjects)]
            subj_i += 1
            charged = (
                domain["key"] in CHARGEABLE and rng.random() < CHARGE_RATE
            )
            specs.append({
                "id": f"C{idx:03d}",
                "length": tier,
                "target_words": cfg["target_words"],
                "word_tolerance": cfg["tolerance"],
                "domain": domain["key"],
                "domain_human": domain["human"],
                "verifiability": domain["verifiability"],
                "subject_hint": subject,
                "anchor_band": band,
                "quality": BAND_QUALITY[band],
                "band_guidance": BAND_GUIDANCE[band],
                "charged": charged,
                "variation_key": idx,  # distinctness lever for the generator
            })

    out = ROOT / "specs" / "specs.jsonl"
    with open(out, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")

    n_band = {b: sum(1 for s in specs if s["anchor_band"] == b) for b in bands}
    n_dom = {d["key"]: sum(1 for s in specs if s["domain"] == d["key"]) for d in DOMAINS}
    print(f"wrote {len(specs)} specs -> {out}")
    print("bands:", n_band)
    print("domains:", n_dom)
    print("charged:", sum(1 for s in specs if s["charged"]))


if __name__ == "__main__":
    main()
