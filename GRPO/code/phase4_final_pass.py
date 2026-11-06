"""Phase 4 -- Final Production Pass.

Reload the Phase 3 fine-tuned 0.5B model, let it generate a refined (greedy,
group_size=1) set of keywords for the test split, feed those into the 8B
model one last time, and save the final optimized predictions. Thin wrapper
around phase1/phase2 so there's a single source of truth for both models'
generation logic.
"""
import argparse

import phase1_keyword_gen as phase1
import phase2_predict as phase2
from model_utils import BIG_MODEL_ID


def run(test_split="data/test.txt",
        trained_checkpoint="outputs/phase3_trained_keyword_model",
        big_checkpoint=BIG_MODEL_ID,
        keywords_out="outputs/phase4_test_keywords.json",
        predictions_out="outputs/phase4_test_predictions.json",
        batch_size=8, big_batch_size=4, limit=None):
    print("[phase4] generating refined keywords with the trained 0.5B model ...")
    phase1.run(test_split, keywords_out, checkpoint=trained_checkpoint,
               batch_size=batch_size, group_size=1, temperature=0.0, limit=limit)

    print("[phase4] generating final predictions with the 8B model ...")
    phase2.run(test_split, keywords_out, predictions_out, checkpoint=big_checkpoint,
               batch_size=big_batch_size)

    print(f"[phase4] done -> {predictions_out}")
    return predictions_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_split", default="data/test.txt")
    ap.add_argument("--trained_checkpoint", default="outputs/phase3_trained_keyword_model")
    ap.add_argument("--big_checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--keywords_out", default="outputs/phase4_test_keywords.json")
    ap.add_argument("--predictions_out", default="outputs/phase4_test_predictions.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--big_batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.test_split, args.trained_checkpoint, args.big_checkpoint,
        args.keywords_out, args.predictions_out, args.batch_size, args.big_batch_size, args.limit)
