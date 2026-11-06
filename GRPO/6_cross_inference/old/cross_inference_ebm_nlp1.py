"""Cross-dataset inference -- run the already-trained, per-element 0.5B
checkpoints (+ the frozen 8B model) on an EXTERNAL dataset (EBM-NLP-style
ground truth), completely separate from this project's own train/dev/test
split.

Input format (one JSON object per line), as in the uploaded
EBM_NLP_ground_truth.jsonl:
    {
      "id": "...",
      "abstract": "<plain text abstract>",
      "raw_entities": {"PAR": [...], "INT": [...], "OUT": [...]},
      "comparator_hierarchical": [...]
    }
Mapped to this project's PICO element names as instructed:
    PAR -> Population, INT -> Intervention, OUT -> Outcome,
    comparator_hierarchical -> Comparator

raw_entities / comparator_hierarchical spans come pre-tokenized with a
literal space before punctuation (e.g. "seasonal allergic rhinitis ( SAR ) .")
-- this is EBM-NLP's own BIO-style tokenization, not natural text. Comparing
that directly against the model's naturally-formatted output would create
spurious mismatches (formatting, not extraction errors), so each gold span is
run through this project's own data_utils.detokenize() (split on whitespace,
then glue back together with the same no-space-before-punctuation /
no-space-after-open-bracket heuristic already used for train.txt/test.txt),
exactly like this project's native BIO gold spans are built.

For each document and EACH element separately (Population, Intervention,
Comparator, Outcome), this mirrors the training pipeline exactly:
  1. load that element's own trained 0.5B checkpoint
     (default: outputs_<Element>/phase3_best_keyword_model -- the dev-
     selected best epoch; override with --checkpoint_subdir, e.g.
     phase3_trained_keyword_model for the final epoch, or sft_keyword_model)
  2. keyword_prompt(text, element=<Element>) -> that model's greedy keyword
     hint for this element only
  3. extraction_prompt(text, keyword_hints=hint, hint_element=<Element>) fed
     to the shared frozen 8B model -> a full Population/Intervention/
     Comparator/Outcome completion
  4. only THIS element's slice of that completion is kept as the prediction
     for THIS element (same convention as reward_element throughout the
     rest of the project -- each element's score comes only from the pass
     that was actually guided by its own hint)

Outputs:
  - a metrics JSON in the same shape as phase5_evaluate.py's report
    (aggregate_metrics over the four elements + macro-F1 "overall")
  - a per-document CSV for error analysis: one row per doc, with each
    element's prediction / gold / per-element F1, plus the raw 8B
    completion for that element's pass.

Usage
-----
    python cross_inference_ebm_nlp.py \\
        --input EBM_NLP_ground_truth.jsonl \\
        --checkpoint_dir_template "outputs_{element}/{subdir}" \\
        --checkpoint_subdir phase3_best_keyword_model \\
        --metrics_out outputs/cross_ebm_nlp_report.json \\
        --csv_out outputs/cross_ebm_nlp_per_doc.csv

Smoke test first:
    python cross_inference_ebm_nlp.py --input EBM_NLP_ground_truth.jsonl --limit 5
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# This script lives in a subfolder (e.g. cross_inference_EBM_NLP/) one level
# below the project root that actually holds data_utils.py, prompts.py,
# metrics.py, model_utils.py and guidelines.py -- add that parent directory
# to the import path so `python cross_inference_ebm_nlp.py` works no matter
# which directory it's launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import detokenize
from guidelines import ELEMENT_ORDER as GUIDELINE_ELEMENT_ORDER  # ["Population","Intervention","Comparator","Outcome"]
from prompts import keyword_prompt, extraction_prompt
from metrics import parse_model_output, reward_f1_element, aggregate_metrics
from model_utils import load_keyword_model, load_big_model, free_model, generate_batch, BIG_MODEL_ID

ELEMENT_ORDER = GUIDELINE_ELEMENT_ORDER  # keep the same canonical order project-wide

# EBM-NLP's own field names -> this project's element names, per the user's mapping.
_RAW_ENTITY_KEY = {"Population": "PAR", "Intervention": "INT", "Outcome": "OUT"}
# Comparator is NOT nested inside raw_entities -- it's its own top-level field.
_COMPARATOR_TOP_LEVEL_KEY = "comparator_hierarchical"


def _detok_span(span):
    """EBM-NLP spans are whitespace-tokenized ("SAR ) .", "( 243 randomized )")
    -- rebuild natural spacing with the same heuristic used for this
    project's own BIO gold spans, so exact-match scoring isn't tripped up by
    formatting alone."""
    return detokenize(span.split())


class ExternalDoc:
    __slots__ = ("doc_id", "text", "gold_spans")

    def __init__(self, doc_id, text, gold_spans):
        self.doc_id = doc_id
        self.text = text
        self.gold_spans = gold_spans


def load_ebm_nlp(path, limit=None):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw = rec.get("raw_entities", {})
            gold = {name: [] for name in ELEMENT_ORDER}
            for element, raw_key in _RAW_ENTITY_KEY.items():
                gold[element] = [_detok_span(s) for s in raw.get(raw_key, []) if s.strip()]
            gold["Comparator"] = [_detok_span(s) for s in rec.get(_COMPARATOR_TOP_LEVEL_KEY, []) if s.strip()]
            docs.append(ExternalDoc(rec["id"], rec["abstract"], gold))
    if limit:
        docs = docs[:limit]
    return docs


def _batched(seq, batch_size):
    for start in range(0, len(seq), batch_size):
        yield seq[start:start + batch_size]


def run(input_path, checkpoint_dir_template="outputs_{element}/{subdir}",
        checkpoint_subdir="phase3_best_keyword_model", big_checkpoint=BIG_MODEL_ID,
        metrics_out="outputs/cross_ebm_nlp_report.json",
        csv_out="outputs/cross_ebm_nlp_per_doc.csv",
        small_batch_size=8, big_batch_size=4,
        max_new_tokens_small=50, max_new_tokens_big=200,
        limit=None, big_model_thinks=False):
    docs = load_ebm_nlp(input_path, limit=limit)
    print(f"[cross_inference] loaded {len(docs)} documents from {input_path}")

    big_model, big_tok = load_big_model(big_checkpoint)

    # completions_by_element[element][i] = the 8B model's FULL P/I/C/O
    # completion for docs[i], guided by that element's own trained hint.
    completions_by_element = {}

    for element in ELEMENT_ORDER:
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
            print(f"[cross_inference] {element}: {len(elem_completions)}/{len(docs)} extractions done")
        completions_by_element[element] = elem_completions

    free_model(big_model, big_tok)

    # Assemble per-doc rows: for each element, keep only that element's slice
    # from the pass that was guided by its own hint (same convention as
    # reward_element elsewhere in this project).
    rows = []
    for i, doc in enumerate(docs):
        combined_pred = {}
        raw_by_element = {}
        f1_by_element = {}
        for element in ELEMENT_ORDER:
            raw = completions_by_element[element][i]
            full_pred = parse_model_output(raw)
            combined_pred[element] = full_pred.get(element, [])
            raw_by_element[element] = raw.strip()
            f1_by_element[element] = reward_f1_element(full_pred, doc.gold_spans, element)
        rows.append({
            "doc_id": doc.doc_id,
            "prediction": combined_pred,
            "gold": doc.gold_spans,
            "raw_by_element": raw_by_element,
            "f1_by_element": f1_by_element,
        })

    preds = [r["prediction"] for r in rows]
    golds = [r["gold"] for r in rows]
    results = aggregate_metrics(preds, golds)

    print(f"\n{'Element':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for elem in ELEMENT_ORDER:
        r = results[elem]
        print(f"{elem:<14}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}")
    print(f"{'Overall (macro F1)':<14}{'':>10}{'':>10}{results['overall']['f1']:>10.3f}")

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({
            "input_path": input_path,
            "checkpoint_dir_template": checkpoint_dir_template,
            "checkpoint_subdir": checkpoint_subdir,
            "n_docs": len(docs),
            "results": results,
        }, f, indent=2)
    print(f"[cross_inference] metrics saved -> {metrics_out}")

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id"]
    for element in ELEMENT_ORDER:
        low = element.lower()
        fieldnames += [f"{low}_pred", f"{low}_gold", f"{low}_f1", f"{low}_raw_completion"]
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row_out = {"doc_id": r["doc_id"]}
            for element in ELEMENT_ORDER:
                low = element.lower()
                row_out[f"{low}_pred"] = " | ".join(r["prediction"][element]) or "none"
                row_out[f"{low}_gold"] = " | ".join(r["gold"][element]) or "none"
                row_out[f"{low}_f1"] = f"{r['f1_by_element'][element]:.4f}"
                row_out[f"{low}_raw_completion"] = r["raw_by_element"][element]
            writer.writerow(row_out)
    print(f"[cross_inference] per-doc CSV saved -> {csv_out}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="EBM-NLP-style JSONL file")
    ap.add_argument("--checkpoint_dir_template", default="outputs_{element}/{subdir}",
                     help="Format string with {element} (Population/Intervention/Comparator/"
                          "Outcome) and {subdir} placeholders. Default matches the "
                          "'outputs_[element]/<checkpoint-folder>' layout you renamed your "
                          "per-run checkpoints into.")
    ap.add_argument("--checkpoint_subdir", default="phase3_best_keyword_model",
                     help="Which checkpoint folder to use under each outputs_<element>/ dir. "
                          "Default is the dev-selected BEST checkpoint. Pass "
                          "phase3_trained_keyword_model for the final epoch instead, or "
                          "sft_keyword_model for the SFT-only warm-start.")
    ap.add_argument("--big_checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--metrics_out", default="outputs/cross_ebm_nlp_report.json")
    ap.add_argument("--csv_out", default="outputs/cross_ebm_nlp_per_doc.csv")
    ap.add_argument("--small_batch_size", type=int, default=8)
    ap.add_argument("--big_batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens_small", type=int, default=50)
    ap.add_argument("--max_new_tokens_big", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="Subsample for a smoke test.")
    ap.add_argument("--big_model_thinks", action="store_true",
                     help="Let Qwen3 use its <think> reasoning mode (needs a much higher "
                          "max_new_tokens_big or the answer never gets generated). Off by default.")
    args = ap.parse_args()
    run(args.input, args.checkpoint_dir_template, args.checkpoint_subdir, args.big_checkpoint,
        args.metrics_out, args.csv_out, args.small_batch_size, args.big_batch_size,
        args.max_new_tokens_small, args.max_new_tokens_big, args.limit, args.big_model_thinks)
