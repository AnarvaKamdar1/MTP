"""One-off converter: test.pkl -> test.jsonl for convert_pkl_to_json.py.

convert_pkl_to_json.py expects each row to have an "abstract" (or "text") key.
test.pkl instead has the text under "input". This just adds an
"abstract" key while keeping every original field (id, raw_entities,
instruction, output) untouched, so nothing downstream needs to change.
"""

import json
import pickle

INPUT_FILE = "test.pkl"
OUTPUT_FILE = "EBM_NLP_Comet_ground_truth.jsonl"

with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for row in data:
        row = dict(row)          # copy, keep original keys as-is
        row["abstract"] = row["input"]
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Converted {len(data)} records -> {OUTPUT_FILE}")
