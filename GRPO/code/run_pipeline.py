"""Command-line orchestrator for the full PICO SFT+GRPO pipeline -- run this
directly on a server (ideally inside `tmux` or with `nohup ... &`), not as
notebook cells.

Pipeline stages, in order:
  1. phase2_test_baseline -- greedy, BASE (untrained) 0.5B keywords on
     test.txt, fed to the 8B model. This is the "before" reference Phase 5
     compares against.
  2. sft -- supervised warm-start of the 0.5B policy on train.txt's gold
     Population spans (plain cross-entropy, no 8B model, no reward). See
     sft_train.py's docstring.
  3. phase3 -- TRUE online GRPO (see phase3_grpo_train.py's docstring): loads
     the SFT-warm-started 0.5B policy AND the 8B model together and trains
     directly off train.txt, generating and scoring fresh every document,
     every epoch. Uses an adaptive KL coefficient (DSP Eq. 7-8) against the
     SFT checkpoint as the frozen reference.
  4. phase4 / phase4_best -- final greedy pass on test.txt with the
     final-epoch and best-epoch (by held-out dev reward) checkpoints.
  5. phase5 -- before/after/best comparison report.

Put a held-out dev.txt at data/dev.txt (or point --dev_split elsewhere) for
genuine held-out best-epoch selection; without it, Phase 3 warns and falls
back to a training-set proxy.

Examples
--------
    # Full run, defaults:
    python run_pipeline.py --stage all

    # Timing/smoke test before committing to a full run -- IMPORTANT given
    # Phase 3 now queries the 8B model epochs x docs x group_size times:
    python run_pipeline.py --stage all --limit 10 --dev_limit 10 --test_limit 10 \\
        --sft_limit 10 --sft_epochs 1 --group_size 3 --epochs 2 --grad_accum_steps 2

    # Re-run only from GRPO onward (e.g. after tweaking lr/kl_target),
    # reusing an SFT checkpoint already on disk:
    python run_pipeline.py --stage phase3,phase4,phase4_best,phase5

    # Detached, logged to a file you can tail from another shell:
    nohup python run_pipeline.py --stage all > /dev/null 2>&1 &
    tail -f outputs/run.log
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import phase1_keyword_gen as phase1
import phase2_predict as phase2
import sft_train
import phase3_grpo_train as phase3
import phase4_final_pass as phase4
import phase5_evaluate as phase5

STAGES = ["phase2_test_baseline", "sft", "phase3", "phase4", "phase4_best", "phase5"]


def _setup_logging(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def _banner(name):
    logging.info("=" * 72)
    logging.info("STAGE: %s", name)
    logging.info("=" * 72)


def run_stage(name, args):
    t0 = time.time()
    _banner(name)

    if name == "phase2_test_baseline":
        # Greedy, base (untrained) 0.5B keywords on TEST -- the "before GRPO"
        # reference Phase 5 compares against.
        phase1.run(args.test_split, "outputs/phase1_test_keywords_base.json",
                   group_size=1, limit=args.test_limit)
        phase2.run(args.test_split, "outputs/phase1_test_keywords_base.json",
                   "outputs/phase2_test_predictions.json")

    elif name == "sft":
        # Supervised warm-start on train.txt's gold Population spans --
        # plain cross-entropy, no 8B model involved. Produces the checkpoint
        # both the GRPO policy and its frozen KL-reference load from.
        sft_train.run(train_split=args.train_split, out_dir=args.sft_out_dir,
                      epochs=args.sft_epochs, lr=args.sft_lr, batch_size=args.sft_batch_size,
                      limit=args.sft_limit, seed=args.seed, log_every=args.log_every)

    elif name == "phase3":
        # True online GRPO -- loads both models, trains directly off
        # train_split, starting from the SFT checkpoint. See
        # phase3_grpo_train.py's docstring for the cost tradeoff (roughly
        # epochs x the 8B-model calls of a single offline pass).
        keyword_checkpoint = args.sft_out_dir if not args.skip_sft else phase3.KEYWORD_MODEL_ID
        phase3.run(train_split=args.train_split, dev_split=args.dev_split,
                   dev_limit=args.dev_limit, use_dev_validation=not args.no_dev_validation,
                   keyword_checkpoint=keyword_checkpoint,
                   epochs=args.epochs, lr=args.lr,
                   use_kl_penalty=not args.no_kl, use_adaptive_kl=not args.no_adaptive_kl,
                   beta0=args.beta0, kl_target=args.kl_target, k_beta=args.k_beta,
                   reward_shaping_weight=args.reward_shaping_weight,
                   group_size=args.group_size, temperature=args.temperature,
                   grad_accum_steps=args.grad_accum_steps, limit=args.limit,
                   seed=args.seed, log_every=args.log_every,
                   track_best=not args.no_track_best,
                   out_dir=args.out_dir, best_out_dir=args.best_out_dir)

    elif name == "phase4":
        # Final pass on TEST with the FINAL-epoch trained checkpoint.
        phase4.run(test_split=args.test_split, limit=args.test_limit,
                   trained_checkpoint=args.out_dir,
                   keywords_out="outputs/phase4_test_keywords.json",
                   predictions_out="outputs/phase4_test_predictions.json")

    elif name == "phase4_best":
        # Same, but with the BEST-epoch checkpoint tracked during training.
        phase4.run(test_split=args.test_split, limit=args.test_limit,
                   trained_checkpoint=args.best_out_dir,
                   keywords_out="outputs/phase_best_test_keywords.json",
                   predictions_out="outputs/phase_best_test_predictions.json")

    elif name == "phase5":
        phase5.run(before_path="outputs/phase2_test_predictions.json",
                   after_path="outputs/phase4_test_predictions.json",
                   best_path="outputs/phase_best_test_predictions.json")

    else:
        raise ValueError(f"Unknown stage: {name}")

    logging.info("STAGE %s done in %.1fs", name, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                     help="comma-separated subset of {" + ",".join(STAGES) + "}, or 'all'")
    ap.add_argument("--train_split", default="data/train.txt")
    ap.add_argument("--dev_split", default="data/dev.txt")
    ap.add_argument("--dev_limit", type=int, default=None, help="subsample dev.txt for validation")
    ap.add_argument("--no_dev_validation", action="store_true",
                     help="Phase 3: skip dev-set validation even if dev.txt exists; falls back "
                          "to training-set mean reward for best-epoch tracking")
    ap.add_argument("--test_split", default="data/test.txt")
    ap.add_argument("--group_size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None, help="subsample train.txt for Phase 3 (GRPO)")
    ap.add_argument("--test_limit", type=int, default=None, help="subsample test.txt for eval stages")
    ap.add_argument("--epochs", type=int, default=5, help="GRPO epochs")
    ap.add_argument("--lr", type=float, default=1e-5, help="GRPO learning rate")
    ap.add_argument("--no_kl", action="store_true", help="disable the KL penalty entirely")
    ap.add_argument("--beta0", type=float, default=0.02, help="initial KL coefficient")
    ap.add_argument("--kl_target", type=float, default=0.2,
                     help="target KL for the adaptive-beta controller (DSP Eq. 7-8)")
    ap.add_argument("--k_beta", type=float, default=0.1, help="adaptive-beta reaction rate")
    ap.add_argument("--no_adaptive_kl", action="store_true",
                     help="use a single fixed KL coefficient (--beta0) instead of adapting it")
    ap.add_argument("--reward_shaping_weight", type=float, default=0.1,
                     help="Phase 3: weight (0-1) on a dense has-any-span bonus blended with "
                          "exact-match F1 for the TRAINING reward only -- reduces all-zero-"
                          "variance sampled groups. Does not affect eval or best-epoch selection. "
                          "Pass 0 to disable.")
    ap.add_argument("--grad_accum_steps", type=int, default=8,
                     help="Phase 3: step the optimizer every this many docs; default 8 matches "
                          "DSP's own batch_size=8. Raise/lower to trade noise for staleness.")
    ap.add_argument("--out_dir", default="outputs/phase3_trained_keyword_model")
    ap.add_argument("--best_out_dir", default="outputs/phase3_best_keyword_model")
    ap.add_argument("--no_track_best", action="store_true")
    ap.add_argument("--log_every", type=int, default=20, help="print progress every N docs/batches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_file", default="outputs/run.log")

    # SFT warm-start options.
    ap.add_argument("--sft_out_dir", default="outputs/sft_keyword_model")
    ap.add_argument("--sft_epochs", type=int, default=3)
    ap.add_argument("--sft_lr", type=float, default=2e-5)
    ap.add_argument("--sft_batch_size", type=int, default=8)
    ap.add_argument("--sft_limit", type=int, default=None, help="subsample train.txt for SFT smoke test")
    ap.add_argument("--skip_sft", action="store_true",
                     help="Skip the 'sft' stage's effect on Phase 3: use the raw base checkpoint "
                          "as Phase 3's keyword_checkpoint instead of --sft_out_dir. Use this if "
                          "you deliberately did not run the 'sft' stage.")
    args = ap.parse_args()

    _setup_logging(args.log_file)

    stages = STAGES if args.stage == "all" else args.stage.split(",")
    for s in stages:
        if s not in STAGES:
            raise ValueError(f"Unknown stage '{s}' -- choose from {STAGES}")

    logging.info("Running stages: %s", stages)
    logging.info("Config: group_size=%s temperature=%s epochs=%s lr=%s beta0=%s kl_target=%s "
                 "limit=%s grad_accum_steps=%s sft_epochs=%s sft_lr=%s seed=%s",
                 args.group_size, args.temperature, args.epochs, args.lr, args.beta0, args.kl_target,
                 args.limit, args.grad_accum_steps, args.sft_epochs, args.sft_lr, args.seed)
    if "phase3" in stages:
        logging.warning("Phase 3 is TRUE online GRPO: the 8B model is queried epochs x docs x "
                        "group_size times now, not once. Expect roughly %sx the wall-clock time "
                        "of a single Phase-1+Phase-2-style pass over the same documents.",
                        args.epochs)

    t_start = time.time()
    for s in stages:
        run_stage(s, args)
    logging.info("ALL DONE in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
