import json
import csv
import re
import statistics

PREDICTION_FILE = "test_unsteered_results.jsonl"
GROUND_TRUTH_FILE = "EBM_NLP_mod_ground_truth.json"
PER_ENTRY_OUTPUT = "evaluations/test_unsteered_results_per_entry.csv"
SUMMARY_OUTPUT = "evaluations/test_unsteered_results_summary.json"


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def category_prf(pred_text, gold_phrases):
    gold_words = set()
    for phrase in gold_phrases:
        gold_words |= _tokenize(phrase)
    pred_words = _tokenize(pred_text)

    if not gold_words and not pred_words:
        return 1.0, 1.0, 1.0
    if not pred_words:
        return 0.0, 0.0, 0.0
    if not gold_words:
        return 0.0, 1.0, 0.0

    inter = len(gold_words & pred_words)
    precision = inter / len(pred_words)
    recall = inter / len(gold_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def normalize_pred_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        if value.strip().lower() in {"none extracted", "none", ""}:
            return []
        return [value]
    return [str(value)]


def main():
    predictions = []
    with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Article-level one-to-one matching by title, with normalized fallback by abstract.
    gt_by_title = {normalize_text(x.get("title", "")): x for x in ground_truth}
    gt_by_abstract = {normalize_text(x.get("abstract", "")): x for x in ground_truth if x.get("abstract", "")}

    rows = []
    matched = 0
    unmatched = 0
    used_gt_titles = set()

    mapping = {
        "Population": "population",
        "Intervention": "intervention_exposure",
        "Comparator": "comparison",
        "Outcome": "outcome",
    }

    for index, prediction in enumerate(predictions, start=1):
        title_key = normalize_text(prediction.get("title", ""))
        abstract_key = normalize_text(prediction.get("abstract", ""))
        gold = gt_by_title.get(title_key)
        if gold is None and abstract_key:
            gold = gt_by_abstract.get(abstract_key)

        if gold is None:
            unmatched += 1
            continue

        matched += 1
        used_gt_titles.add(normalize_text(gold.get("title", "")))
        generated = prediction.get("pico_directer", {}) or {}

        row = {
            "index": index,
            "title": prediction.get("title", ""),
        }

        for pred_key, gold_key in mapping.items():
            gold_phrases = gold.get(gold_key, []) or []
            pred_phrases = normalize_pred_field(generated.get(pred_key))
            pred_text = " ".join(pred_phrases)
            precision, recall, f1 = category_prf(pred_text, gold_phrases)

            prefix = {"Population": "P", "Intervention": "I", "Comparator": "C", "Outcome": "O"}[pred_key]
            row[f"{prefix}_gold"] = " | ".join(map(str, gold_phrases))
            row[f"{prefix}_pred"] = pred_text
            row[f"{prefix}_precision"] = precision
            row[f"{prefix}_recall"] = recall
            row[f"{prefix}_f1"] = f1

        rows.append(row)

    missing_gt = len(ground_truth) - len(used_gt_titles)

    fieldnames = [
        "index", "title",
        "P_gold", "P_pred", "P_precision", "P_recall", "P_f1",
        "I_gold", "I_pred", "I_precision", "I_recall", "I_f1",
        "C_gold", "C_pred", "C_precision", "C_recall", "C_f1",
        "O_gold", "O_pred", "O_precision", "O_recall", "O_f1",
    ]

    with open(PER_ENTRY_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    categories = ["P", "I", "C", "O"]
    summary = {
        "num_prediction_records": len(predictions),
        "num_ground_truth_articles": len(ground_truth),
        "num_matched_records": matched,
        "num_unmatched_prediction_records": unmatched,
        "num_ground_truth_articles_without_prediction": missing_gt,
        "ground_truth_mapping": {
            "Population": "population",
            "Intervention": "intervention_exposure",
            "Comparator": "comparison",
            "Outcome": "outcome",
        },
        "prediction_field": "pico_directer",
        "metric": "Token-overlap Precision/Recall/F1 using set-based tokenization",
        "categories": {},
    }

    for letter in categories:
        ps = [r[f"{letter}_precision"] for r in rows]
        rs = [r[f"{letter}_recall"] for r in rows]
        fs = [r[f"{letter}_f1"] for r in rows]
        summary["categories"][letter] = {
            "precision": statistics.mean(ps) if ps else 0.0,
            "recall": statistics.mean(rs) if rs else 0.0,
            "f1": statistics.mean(fs) if fs else 0.0,
        }

    all_p, all_r, all_f = [], [], []
    for r in rows:
        for letter in categories:
            all_p.append(r[f"{letter}_precision"])
            all_r.append(r[f"{letter}_recall"])
            all_f.append(r[f"{letter}_f1"])

    summary["overall"] = {
        "precision": statistics.mean(all_p) if all_p else 0.0,
        "recall": statistics.mean(all_r) if all_r else 0.0,
        "f1": statistics.mean(all_f) if all_f else 0.0,
    }

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Loaded prediction records : {len(predictions)}")
    print(f"Loaded ground-truth       : {len(ground_truth)}")
    print(f"Matched records           : {matched}")
    print(f"Unmatched predictions     : {unmatched}")
    print(f"GT without prediction     : {missing_gt}")
    print("\nResults")
    for letter, name in [("P", "Population"), ("I", "Intervention"), ("C", "Comparator"), ("O", "Outcome")]:
        m = summary["categories"][letter]
        print(f"{name:12s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"Overall      P={summary['overall']['precision']:.4f} R={summary['overall']['recall']:.4f} F1={summary['overall']['f1']:.4f}")
    print(f"\nSaved: {PER_ENTRY_OUTPUT}\nSaved: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
