import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json

def write_jsonl(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def read_artefact(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Each artefact file is a single JSON object. Reading it directly with
# pd.read_json expands the nested "pushback" dict into multiple rows, so load
# each as one record and build the DataFrame from the list instead.
artefacts = [
    read_artefact('Marthe/artefacts/L01_scale100.json'),
    read_artefact('Marthe/artefacts/M02_scale100.json'),
    read_artefact('Marthe/artefacts/S02_scale100.json'),
]

df = pd.DataFrame(artefacts)

# Number of independent runs per artefact, used to build a score distribution.
N_RUNS = 20

# Create scoring batch for OpenAI API
messages = []
for i, row in df.iterrows():
    for run in range(N_RUNS):
        messages.append(
            {
                # custom_id must be unique within a batch, so suffix the run index.
                "custom_id": f"{row['id']}_run{run}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5.5-2026-04-23",
                    "reasoning": {
                        "effort": "medium",
                    },
                    "instructions": (
                        row['anti_sycophantic_prompt']
                    ),
                    "input": f"Artefact: {row['artefact']}\n",
                },
            }
        )
#write_jsonl("Marthe/results/initial_scores/test_batch1.jsonl", messages) #with "initial_prompt"
write_jsonl("Marthe/results/initial_scores/test_batch2.jsonl", messages) #with "anti_sycophantic_prompt"