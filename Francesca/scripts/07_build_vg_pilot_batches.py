import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "Francesca" / "scripts"
ARTEFACT_DIR = ROOT / "Vincent" / "sycophancy-benchmark" / "artefacts" / "json"
NEUTRALITY_OUT_DIR = ROOT / "Francesca" / "results" / "vg_neutrality"
SCORING_OUT_DIR = ROOT / "Francesca" / "results" / "vg_scoring"

# One short, one medium, one long artefact for a cheap end-to-end test.
PILOT_ARTEFACT_IDS = ["S05", "M07", "L03"]
PILOT_N_RUNS = 5

PILOT_SCORING_CONDITIONS = [
    ("domain_specific", "neutral"),
    ("do_you_like", "neutral"),
    ("domain_specific", "anti_sycophantic"),
]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


neutrality = load_script(
    SCRIPT_DIR / "03_build_vg_neutrality_batch.py", "vg_neutrality_builder"
)
scoring = load_script(
    SCRIPT_DIR / "04_build_vg_scoring_batches.py", "vg_scoring_builder"
)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def load_pilot_artefacts():
    artefacts = []
    for artefact_id in PILOT_ARTEFACT_IDS:
        path = ARTEFACT_DIR / f"{artefact_id}.json"
        if not path.exists():
            raise SystemExit(f"Missing pilot artefact: {path}")
        artefacts.append(read_json(path))
    return artefacts


def build_neutrality(artefacts):
    requests = [neutrality.make_request(artefact) for artefact in artefacts]
    out_path = NEUTRALITY_OUT_DIR / "batch_input_vg_neutrality_pilot.jsonl"
    write_jsonl(out_path, requests)
    print(f"Wrote {len(requests)} pilot neutrality requests to {out_path}")


def build_scoring(artefacts):
    for prompt_name, evaluator_name in PILOT_SCORING_CONDITIONS:
        requests = []
        for artefact in artefacts:
            for run in range(PILOT_N_RUNS):
                requests.append(
                    scoring.make_request(artefact, prompt_name, evaluator_name, run)
                )
        out_path = (
            SCORING_OUT_DIR
            / f"batch_input_score_pilot_{prompt_name}_{evaluator_name}.jsonl"
        )
        write_jsonl(out_path, requests)
        print(f"Wrote {len(requests)} pilot scoring requests to {out_path}")


def main():
    artefacts = load_pilot_artefacts()
    print(f"Pilot artefacts: {', '.join(a['id'] for a in artefacts)}")
    print(f"Pilot scoring runs per artefact/condition: {PILOT_N_RUNS}")
    build_neutrality(artefacts)
    build_scoring(artefacts)


if __name__ == "__main__":
    main()
