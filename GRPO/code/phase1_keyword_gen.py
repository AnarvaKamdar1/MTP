"""Phase 1 -- Keyword Generation (0.5B model).

Loads the small model, generates candidate PICO keyword spans for every
document in a split, saves them to JSON, and completely unloads the model.

Two modes:
  --group_size 1                 -> one greedy completion per doc (for eval /
                                     for Phase 4's final production pass).
  --group_size G>1 --temperature -> G sampled completions per doc, used as the
                                     GRPO rollout buffer for Phase 3 training.
"""
import argparse
import json
from pathlib import Path

from data_utils import load_conll
from prompts import keyword_prompt
from model_utils import load_keyword_model, free_model, generate_batch, KEYWORD_MODEL_ID


def run(split_path, out_path, checkpoint=KEYWORD_MODEL_ID, batch_size=8,
        limit=None, group_size=1, temperature=0.0, max_new_tokens=50):
    docs = load_conll(split_path)
    if limit:
        docs = docs[:limit]
    model, tokenizer = load_keyword_model(checkpoint, trainable=False)

    records = []
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        prompts = [keyword_prompt(d.text) for d in batch]
        completions = generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=group_size,
        )
        # completions is flat: [doc0_sample0, doc0_sample1, ..., doc1_sample0, ...]
        for i, doc in enumerate(batch):
            samples = completions[i * group_size:(i + 1) * group_size]
            records.append({"doc_id": doc.doc_id, "keywords": [s.strip() for s in samples]})
        print(f"[phase1] {min(start + batch_size, len(docs))}/{len(docs)} docs done")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[phase1] saved -> {out_path}")

    free_model(model, tokenizer)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/train.txt")
    ap.add_argument("--out", default="outputs/phase1_train_keywords.json")
    ap.add_argument("--checkpoint", default=KEYWORD_MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--group_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    args = ap.parse_args()
    run(args.split, args.out, args.checkpoint, args.batch_size, args.limit,
        args.group_size, args.temperature, args.max_new_tokens)
