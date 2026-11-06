"""Phase 5 -- Evaluation Metrics.

Computes precision/recall/F1 per PICO element (set-based tp/fp/fn over
normalized predicted vs. gold spans, see metrics.py) for up to three runs
on the TEST split:
  - "before"  -- Phase 2 predictions guided by the *untrained* (base) 0.5B
    model's keywords.
  - "after"   -- Phase 4 predictions guided by the *final-epoch* trained
    0.5B model's keywords.
  - "best"    -- Phase 4 predictions guided by the checkpoint Phase 3
    tracked as having the best training-time mean reward (may differ from
    the final epoch). Optional -- pass best_path=None or a nonexistent
    path to skip this comparison (e.g. if Phase 3 was run with
    --no_track_best).
"""
import argparse
import json
from pathlib import Path

from metrics import aggregate_metrics, ELEMENT_ORDER


def _load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    preds = [r["prediction"] for r in rows]
    golds = [r["gold"] for r in rows]
    return preds, golds


def _print_table(name, results):
    print(f"\n== {name} ==")
    print(f"{'Element':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for elem in ELEMENT_ORDER:
        r = results[elem]
        print(f"{elem:<14}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}")
    print(f"{'Overall (macro F1)':<14}{'':>10}{'':>10}{results['overall']['f1']:>10.3f}")


def run(before_path="outputs/phase2_test_predictions.json",
        after_path="outputs/phase4_test_predictions.json",
        best_path="outputs/phase_best_test_predictions.json",
        report_out="outputs/phase5_report.json"):
    before_preds, before_golds = _load(before_path)
    after_preds, after_golds = _load(after_path)
    before_results = aggregate_metrics(before_preds, before_golds)
    after_results = aggregate_metrics(after_preds, after_golds)

    _print_table("BEFORE GRPO (Phase 2, base 0.5B keywords)", before_results)
    _print_table("AFTER GRPO (Phase 4, final-epoch trained 0.5B keywords)", after_results)

    report = {"before": before_results, "after": after_results,
              "delta_after_vs_before_f1": after_results["overall"]["f1"] - before_results["overall"]["f1"]}
    print(f"\nAfter vs. before overall macro-F1 change: {report['delta_after_vs_before_f1']:+.4f}")

    have_best = best_path and Path(best_path).exists()
    if have_best:
        best_preds, best_golds = _load(best_path)
        best_results = aggregate_metrics(best_preds, best_golds)
        _print_table("BEST (Phase 4, best-tracked-epoch checkpoint)", best_results)

        report["best"] = best_results
        report["delta_best_vs_before_f1"] = best_results["overall"]["f1"] - before_results["overall"]["f1"]
        report["delta_best_vs_after_f1"] = best_results["overall"]["f1"] - after_results["overall"]["f1"]
        print(f"Best vs. before overall macro-F1 change: {report['delta_best_vs_before_f1']:+.4f}")
        print(f"Best vs. after (final epoch) overall macro-F1 change: {report['delta_best_vs_after_f1']:+.4f}")
    else:
        print("\n(No 'best' checkpoint predictions found -- skipping that comparison. "
              "Run the phase4_best stage first, or pass --best explicitly.)")

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[phase5] saved -> {report_out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="outputs/phase2_test_predictions.json")
    ap.add_argument("--after", default="outputs/phase4_test_predictions.json")
    ap.add_argument("--best", default="outputs/phase_best_test_predictions.json")
    ap.add_argument("--report_out", default="outputs/phase5_report.json")
    args = ap.parse_args()
    run(args.before, args.after, args.best, args.report_out)
