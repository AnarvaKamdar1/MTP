"""Zero-shot baseline -- Qwen3-8B alone on a split, no 0.5B model involved at
all (no small model is loaded or called anywhere in this script).

Uses the exact same prompt and instruction as the rest of this pipeline --
`extraction_prompt(doc.text, keyword_hints=None)` -- so the only difference
from Phase 2's predictions is the absence of the hints block; the guideline
block, output-format instructions, and generation settings (greedy,
thinking off) are identical. That makes this a clean apples-to-apples "what
does the 8B model do completely on its own" reference point, comparable
directly against phase2_test_predictions.json (base 0.5B hints) and
phase4_test_predictions.json / phase_best_test_predictions.json (trained
0.5B hints).

Output schema matches phase2_predict.py's rows (doc_id, raw_completion,
prediction, gold, reward, reward_element), so it's a drop-in for
phase5_evaluate.py's --before/--after/--best -- each only needs the
'prediction'/'gold' keys. E.g.:

    python zeroshot_eval.py --out outputs/zeroshot_test_predictions.json
    python phase5_evaluate.py --before outputs/zeroshot_test_predictions.json \\
        --after outputs/phase4_test_predictions.json \\
        --best outputs/phase_best_test_predictions.json
"""
import argparse
import json
from pathlib import Path

from data_utils import load_conll
from prompts import extraction_prompt, KEYWORD_ELEMENT
from metrics import parse_model_output, reward_f1, reward_f1_element
from model_utils import load_big_model, free_model, generate_batch, BIG_MODEL_ID


def run(split_path="data/test.txt", out_path="outputs/zeroshot_test_predictions.json",
        checkpoint=BIG_MODEL_ID, batch_size=4, max_new_tokens=200, limit=None,
        big_model_thinks=False):
    docs = load_conll(split_path)
    if limit:
        docs = docs[:limit]

    model, tokenizer = load_big_model(checkpoint)

    results = []
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        prompts = [extraction_prompt(d.text, keyword_hints=None) for d in batch]
        completions = generate_batch(model, tokenizer, prompts,
                                      max_new_tokens=max_new_tokens, temperature=0.0,
                                      chat_template_kwargs={"enable_thinking": big_model_thinks})
        for doc, completion in zip(batch, completions):
            pred = parse_model_output(completion)
            results.append({
                "doc_id": doc.doc_id,
                "raw_completion": completion.strip(),
                "prediction": pred,
                "gold": doc.gold_spans,
                "reward": reward_f1(pred, doc.gold_spans),  # macro F1 over P/I/C/O, informational
                "reward_element": reward_f1_element(pred, doc.gold_spans, KEYWORD_ELEMENT),
            })
        print(f"[zeroshot] {min(start + batch_size, len(docs))}/{len(docs)} generations done")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[zeroshot] saved -> {out_path}")

    free_model(model, tokenizer)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/test.txt")
    ap.add_argument("--out", default="outputs/zeroshot_test_predictions.json")
    ap.add_argument("--checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="Subsample the split for a smoke test.")
    ap.add_argument("--big_model_thinks", action="store_true",
                     help="Let Qwen3 use its <think> reasoning mode (needs a much higher "
                          "max_new_tokens or the answer never gets generated). Off by default.")
    args = ap.parse_args()
    run(args.split, args.out, args.checkpoint, args.batch_size,
        args.max_new_tokens, args.limit, args.big_model_thinks)
