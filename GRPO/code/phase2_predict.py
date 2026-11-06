"""Phase 2 -- Prediction Generation (8B model).

Loads the big model, feeds it the original abstract plus the Phase 1
keywords, generates the full PICO extraction, parses it, scores it against
gold, saves everything, and unloads the model.

Works against Phase 1 output of any group_size:
  - group_size 1 (eval mode)  -> one prediction + reward per doc.
  - group_size G (RL mode)    -> G predictions + rewards per doc, which
    Phase 3 turns into a GRPO rollout buffer (group-relative advantages).
"""
import argparse
import json
from pathlib import Path

from data_utils import load_conll
from prompts import extraction_prompt, KEYWORD_ELEMENT
from metrics import parse_model_output, reward_f1, reward_f1_element
from model_utils import load_big_model, free_model, generate_batch, BIG_MODEL_ID


def run(split_path, keywords_path, out_path, checkpoint=BIG_MODEL_ID,
        batch_size=4, max_new_tokens=200, big_model_thinks=False):
    docs = {d.doc_id: d for d in load_conll(split_path)}
    with open(keywords_path, encoding="utf-8") as f:
        keyword_records = json.load(f)

    # Flatten to one (doc, keyword_sample, sample_idx) row per generation call.
    rows = []
    for rec in keyword_records:
        doc = docs.get(rec["doc_id"])
        if doc is None:
            continue
        for idx, kw in enumerate(rec["keywords"]):
            rows.append({"doc_id": doc.doc_id, "sample_idx": idx, "text": doc.text,
                         "keywords": kw, "gold": doc.gold_spans})

    model, tokenizer = load_big_model(checkpoint)

    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [extraction_prompt(r["text"], r["keywords"]) for r in batch]
        completions = generate_batch(model, tokenizer, prompts,
                                      max_new_tokens=max_new_tokens, temperature=0.0,
                                      chat_template_kwargs={"enable_thinking": big_model_thinks})
        for row, completion in zip(batch, completions):
            pred = parse_model_output(completion)
            results.append({
                "doc_id": row["doc_id"],
                "sample_idx": row["sample_idx"],
                "keywords": row["keywords"],
                "raw_completion": completion.strip(),
                "prediction": pred,
                "gold": row["gold"],
                "reward": reward_f1(pred, row["gold"]),  # macro F1 over P/I/C/O, informational only
                "reward_element": reward_f1_element(pred, row["gold"], KEYWORD_ELEMENT),  # what Phase 3 trains on
            })
        print(f"[phase2] {min(start + batch_size, len(rows))}/{len(rows)} generations done")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[phase2] saved -> {out_path}")

    free_model(model, tokenizer)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/train.txt")
    ap.add_argument("--keywords", default="outputs/phase1_train_keywords.json")
    ap.add_argument("--out", default="outputs/phase2_train_predictions.json")
    ap.add_argument("--checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--big_model_thinks", action="store_true",
                     help="Let Qwen3 use its <think> reasoning mode (needs a much higher "
                          "max_new_tokens or the answer never gets generated). Off by default.")
    args = ap.parse_args()
    run(args.split, args.keywords, args.out, args.checkpoint,
        args.batch_size, args.max_new_tokens, args.big_model_thinks)
