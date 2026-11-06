"""Cross-dataset inference on EBM_NLP_Comet -- run the already-trained,
per-element 0.5B checkpoints (+ the frozen 8B model) on the Comet EBM-NLP
ground truth, completely separate from this project's own train/dev/test split.

Input format (one JSON object per line), EBM_NLP_Comet_ground_truth.jsonl:
    {"id": "...", "abstract": "<text>", "raw_entities": {"PAR": [...], "INT": [...], "OUT": [...]}, ...}
This dataset has NO Comparator annotation, so only three elements are run and scored:
    PAR -> Population, INT -> Intervention, OUT -> Outcome
The Comparator checkpoint is never loaded and Comparator is never scored.

Evaluation is kept identical to the baselines (unified_metric.py) and to the
other GRPO evaluations:
  * gold spans are used EXACTLY as stored in the ground-truth file (only
    surrounding whitespace stripped) -- no detokenization;
  * scoring is metrics.aggregate_metrics, imported unchanged; the Comparator
    row it produces (empty on both sides) is dropped and so is its "overall"
    macro average -- report precision / recall / F1 per element only;
  * every record in the file is scored (no --limit in the real run).

For each record and each of the three elements (same as the Mod / EBM_NLP runs):
  1. load that element's own trained 0.5B checkpoint
     (default: outputs_<Element>/phase3_best_keyword_model)
  2. keyword_prompt(text, element=<Element>) -> greedy keyword hint
  3. extraction_prompt(text, keyword_hints=hint, hint_element=<Element>) -> the
     frozen 8B model's full P/I/C/O completion (prompt unchanged from training)
  4. only THIS element's slice of that completion is kept as its prediction.

Long run (~2k records x 3 elements): after each element finishes, its raw 8B
completions are cached to <csv_out stem>_completions_<Element>.json. Re-running
with --resume reuses any cached element whose record count matches, so a crash
only costs the element that was in progress.

Outputs:
  - metrics JSON (per-element precision / recall / F1 / tp / fp / fn)
  - per-record CSV (prediction / gold / per-record F1 / raw completion per element)
  - <csv_out stem>_predictions.json (predictions + gold + raw completions; re-scorable
    without re-running the models)

Usage (from the project root, i.e. the folder holding metrics.py and outputs_<Element>/):
    python cross_inference_EBM_NLP_Comet/cross_inference_ebm_nlp_comet.py \\
        --input cross_inference_EBM_NLP_Comet/EBM_NLP_Comet_ground_truth.jsonl \\
        --checkpoint_subdir phase3_best_keyword_model \\
        --metrics_out cross_inference_EBM_NLP_Comet/outputs/cross_comet_report.json \\
        --csv_out cross_inference_EBM_NLP_Comet/outputs/cross_comet_per_doc.csv --resume
(If launched from inside the subfolder, pass --checkpoint_dir_template "../outputs_{element}/{subdir}".)

Smoke test first:  ... --limit 5
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# This script lives one level below the project root that holds data_utils.py,
# prompts.py, metrics.py, model_utils.py and guidelines.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import keyword_prompt, extraction_prompt
from metrics import parse_model_output, reward_f1_element, aggregate_metrics
from model_utils import load_keyword_model, load_big_model, free_model, generate_batch, BIG_MODEL_ID

# Comet has no Comparator annotation -> fixed three-element set, canonical order.
ACTIVE_ELEMENTS = ["Population", "Intervention", "Outcome"]
_RAW_ENTITY_KEY = {"Population": "PAR", "Intervention": "INT", "Outcome": "OUT"}


class ExternalDoc:
    __slots__ = ("doc_id", "text", "gold_spans")

    def __init__(self, doc_id, text, gold_spans):
        self.doc_id = doc_id
        self.text = text
        self.gold_spans = gold_spans


def load_comet(path, limit=None):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw = rec.get("raw_entities", {})
            # gold used as-is (whitespace-stripped only) -- same as the baselines.
            # Comparator kept as an always-empty list so aggregate_metrics' fixed
            # 4-element loop runs unchanged; its row is dropped from the report.
            gold = {el: [s.strip() for s in raw.get(key, []) if s.strip()]
                    for el, key in _RAW_ENTITY_KEY.items()}
            gold["Comparator"] = []
            docs.append(ExternalDoc(rec["id"], rec["abstract"], gold))
    if limit:
        docs = docs[:limit]
    return docs


def _batched(seq, batch_size):
    for start in range(0, len(seq), batch_size):
        yield seq[start:start + batch_size]


def run(input_path, checkpoint_dir_template="outputs_{element}/{subdir}",
        checkpoint_subdir="phase3_best_keyword_model", big_checkpoint=BIG_MODEL_ID,
        metrics_out="outputs/cross_comet_report.json",
        csv_out="outputs/cross_comet_per_doc.csv",
        small_batch_size=8, big_batch_size=4,
        max_new_tokens_small=50, max_new_tokens_big=200,
        limit=None, big_model_thinks=False, resume=False):
    docs = load_comet(input_path, limit=limit)
    print(f"[cross_inference] loaded {len(docs)} documents from {input_path}")
    print(f"[cross_inference] elements run and scored: {ACTIVE_ELEMENTS} (no Comparator)")

    stem = str(Path(csv_out).with_suffix(""))
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)

    def cache_path(element):
        return Path(f"{stem}_completions_{element}.json")

    completions_by_element = {}
    if resume:
        for element in ACTIVE_ELEMENTS:
            p = cache_path(element)
            if p.exists():
                cached = json.load(open(p, encoding="utf-8"))
                if [c["doc_id"] for c in cached] == [d.doc_id for d in docs]:
                    completions_by_element[element] = [c["completion"] for c in cached]
                    print(f"[cross_inference] {element}: reusing {len(cached)} cached completions from {p}")

    todo = [e for e in ACTIVE_ELEMENTS if e not in completions_by_element]
    if todo:
        big_model, big_tok = load_big_model(big_checkpoint)
        for element in todo:
            ckpt = checkpoint_dir_template.format(element=element, subdir=checkpoint_subdir)
            print(f"[cross_inference] {element}: loading keyword checkpoint {ckpt}")
            small_model, small_tok = load_keyword_model(ckpt, trainable=False)

            hints = []
            for batch in _batched(docs, small_batch_size):
                prompts = [keyword_prompt(d.text, element=element) for d in batch]
                comps = generate_batch(small_model, small_tok, prompts,
                                        max_new_tokens=max_new_tokens_small, temperature=0.0)
                hints.extend(c.strip() for c in comps)
            free_model(small_model, small_tok)
            print(f"[cross_inference] {element}: generated {len(hints)} keyword hints")

            elem_completions = []
            for batch_docs, batch_hints in zip(_batched(docs, big_batch_size), _batched(hints, big_batch_size)):
                prompts = [extraction_prompt(d.text, keyword_hints=h, hint_element=element)
                           for d, h in zip(batch_docs, batch_hints)]
                comps = generate_batch(big_model, big_tok, prompts,
                                        max_new_tokens=max_new_tokens_big, temperature=0.0,
                                        chat_template_kwargs={"enable_thinking": big_model_thinks})
                elem_completions.extend(comps)
                if len(elem_completions) % 200 < big_batch_size or len(elem_completions) == len(docs):
                    print(f"[cross_inference] {element}: {len(elem_completions)}/{len(docs)} extractions done")
            completions_by_element[element] = elem_completions

            with open(cache_path(element), "w", encoding="utf-8") as f:
                json.dump([{"doc_id": d.doc_id, "keyword_hint": h, "completion": c}
                           for d, h, c in zip(docs, hints, elem_completions)], f, indent=1)
            print(f"[cross_inference] {element}: completions cached -> {cache_path(element)}")
        free_model(big_model, big_tok)

    # Per-record rows: each element's prediction comes only from its own hinted pass.
    rows = []
    for i, doc in enumerate(docs):
        combined_pred = {"Comparator": []}
        raw_by_element, f1_by_element = {}, {}
        for element in ACTIVE_ELEMENTS:
            raw = completions_by_element[element][i]
            full_pred = parse_model_output(raw)
            combined_pred[element] = full_pred.get(element, [])
            raw_by_element[element] = raw.strip()
            f1_by_element[element] = reward_f1_element(full_pred, doc.gold_spans, element)
        rows.append({"doc_id": doc.doc_id, "prediction": combined_pred, "gold": doc.gold_spans,
                     "raw_by_element": raw_by_element, "f1_by_element": f1_by_element})

    # metrics.aggregate_metrics, unchanged. Comparator is empty on both sides -> its
    # row is 0/0/0 and is dropped, as is the macro "overall" (per-element reporting only).
    full = aggregate_metrics([r["prediction"] for r in rows], [r["gold"] for r in rows])
    results = {el: full[el] for el in ACTIVE_ELEMENTS}

    print(f"\n{'Element':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for el in ACTIVE_ELEMENTS:
        r = results[el]
        print(f"{el:<14}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}")

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({"input_path": input_path, "checkpoint_dir_template": checkpoint_dir_template,
                   "checkpoint_subdir": checkpoint_subdir, "elements": ACTIVE_ELEMENTS,
                   "n_docs": len(docs), "max_new_tokens_big": max_new_tokens_big,
                   "results": results}, f, indent=2)
    print(f"[cross_inference] metrics saved -> {metrics_out}")

    fieldnames = ["doc_id"]
    for el in ACTIVE_ELEMENTS:
        low = el.lower()
        fieldnames += [f"{low}_pred", f"{low}_gold", f"{low}_f1", f"{low}_raw_completion"]
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row_out = {"doc_id": r["doc_id"]}
            for el in ACTIVE_ELEMENTS:
                low = el.lower()
                row_out[f"{low}_pred"] = " | ".join(r["prediction"][el]) or "none"
                row_out[f"{low}_gold"] = " | ".join(r["gold"][el]) or "none"
                row_out[f"{low}_f1"] = f"{r['f1_by_element'][el]:.4f}"
                row_out[f"{low}_raw_completion"] = r["raw_by_element"][el]
            writer.writerow(row_out)
    print(f"[cross_inference] per-doc CSV saved -> {csv_out}")

    preds_out = f"{stem}_predictions.json"
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump([{"doc_id": r["doc_id"], "prediction": r["prediction"], "gold": r["gold"],
                    "raw_by_element": r["raw_by_element"]} for r in rows], f, indent=1)
    print(f"[cross_inference] predictions saved -> {preds_out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="EBM_NLP_Comet_ground_truth.jsonl")
    ap.add_argument("--checkpoint_dir_template", default="outputs_{element}/{subdir}",
                    help="Relative to the directory you launch from. From inside the subfolder use "
                         "'../outputs_{element}/{subdir}'.")
    ap.add_argument("--checkpoint_subdir", default="phase3_best_keyword_model")
    ap.add_argument("--big_checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--metrics_out", default="outputs/cross_comet_report.json")
    ap.add_argument("--csv_out", default="outputs/cross_comet_per_doc.csv")
    ap.add_argument("--small_batch_size", type=int, default=8)
    ap.add_argument("--big_batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens_small", type=int, default=50)
    ap.add_argument("--max_new_tokens_big", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test only -- omit for the real run.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse per-element completion caches from a previous (crashed) run.")
    ap.add_argument("--big_model_thinks", action="store_true")
    args = ap.parse_args()
    run(args.input, args.checkpoint_dir_template, args.checkpoint_subdir, args.big_checkpoint,
        args.metrics_out, args.csv_out, args.small_batch_size, args.big_batch_size,
        args.max_new_tokens_small, args.max_new_tokens_big, args.limit, args.big_model_thinks,
        args.resume)
