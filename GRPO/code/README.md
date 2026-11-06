# PICO GRPO Pipeline

Trains a small **Qwen2.5-0.5B-Instruct** "keyword policy" -- first with a
supervised warm-start, then with online GRPO -- so that its suggested
keyword hints make **Qwen3-8B**'s PICO extraction (Population /
Intervention / Comparator / Outcome) more accurate, using `guidelines.py`'s
annotation rules and the BIO-tagged `train.txt` / `dev.txt` / `test.txt`
dataset (`dev.txt` is used only for held-out best-epoch selection during
GRPO, never trained on; `test.txt` is used only for the final
before/after/best evaluation). Currently trains on the **Population**
element only (`prompts.KEYWORD_ELEMENT`).

This follows the two-stage recipe from **Directional Stimulus Prompting**
(DSP, Li et al., NeurIPS 2023): a small tunable policy model generates a
"directional stimulus" (here, keyword hints) that steers a frozen black-box
LLM's output, trained first with supervised fine-tuning (SFT) on labeled
pseudo-stimulus data and then with reinforcement learning against the
downstream LLM's output quality. Deliberate differences from the paper:
GRPO with a group baseline instead of PPO+value-network, a discrete F1
reward instead of continuous ROUGE, and one 8B sample per query at scoring
time instead of their 4-sample average -- the overall SFT-then-RL shape and
the adaptive-KL controller (Eq. 7-8) are matched directly.

## Files
- `guidelines.py` -- your annotation rules (unchanged).
- `data_utils.py` -- parses the CoNLL/BIO files into documents with raw text
  + gold PICO spans (`load_conll`, `PicoDoc`).
- `prompts.py` -- builds both models' prompts from `guidelines.py`.
  `keyword_prompt` asks the 0.5B model for `KEYWORD_ELEMENT` spans only;
  `extraction_prompt` asks the 8B model for the full P/I/C/O extraction,
  weighing the 0.5B model's hints alongside its own reading rather than
  defaulting heavily to either.
- `metrics.py` -- parses model output into spans and scores it. Spans are
  normalized before comparison (lowercased, trailing punctuation stripped,
  spaced-out hyphens collapsed) so surface formatting differences between
  the gold spans and the model's natural phrasing aren't scored as errors.
- `model_utils.py` -- load/unload helpers; both models placed explicitly on
  one GPU (no multi-GPU/CPU-offload sharding); silences noisy HF/transformers
  logging and progress bars.
- `run_pipeline.py` -- CLI orchestrator (`tmux`/`nohup`-friendly, timestamped
  logging to file + stdout).
- `phase1_keyword_gen.py` / `phase2_predict.py` -- used for the one-shot
  "before" baseline and the final "after"/"best" passes on `test.txt`. No
  longer used to build a training buffer (see below).
- `sft_train.py` -- **supervised warm-start**, see next section.
- `phase3_grpo_train.py` -- **true online GRPO training** with an adaptive
  KL coefficient, starting from the SFT checkpoint. See next section.
- `phase4_final_pass.py` -- final pass on `test.txt` with a trained
  checkpoint (called twice: once for the final-epoch model, once for the
  best-tracked-epoch model).
- `phase5_evaluate.py` -- before / after / best 3-way comparison.

## Stage 1: SFT warm-start (`sft_train.py`)

Before any RL, the policy is fine-tuned directly on `train.txt`'s **gold**
Population spans: for every doc, `keyword_prompt(doc.text)` is the input and
`"Population: <gold spans joined by ' | '>"` (or `"...: none"`) is the
target, trained with plain next-token cross-entropy (prompt tokens masked
out of the loss). This removes the burden GRPO would otherwise carry alone
-- learning output *format* and span *quality* simultaneously from a cold,
generic-instruct start with only a sparse F1 reward. Prints are tagged
plainly `SFT`, e.g. `SFT epoch 2/3 batch 40/50 -- running mean loss 0.83`.

The resulting checkpoint (`--sft_out_dir`, default
`outputs/sft_keyword_model`) is what GRPO loads as **both** its trainable
policy's starting point and its frozen KL-reference -- so the KL penalty
during GRPO measures drift from the warm-started policy, not the raw base
model, matching DSP's own pi_0 = SFT policy setup.

## Stage 2: true online GRPO (`phase3_grpo_train.py`)

For every training document, every epoch, strictly in sequence:
1. Sample `group_size` keyword completions from the **current** 0.5B policy.
2. Score every one of them with a **fresh** 8B-model extraction (no reuse
   across epochs -- the 8B model sees each document once per epoch, not
   once total).
3. Compute the group-relative advantage from those rewards.
4. Compute each completion's log-probability under the same policy that
   just generated it -- since there's no lag between sampling and training,
   this is on-policy REINFORCE with a group baseline (no PPO ratio/clip
   needed, unlike an earlier buffer-based design that trained multiple
   epochs against a fixed, aging rollout).
5. Penalize KL(policy || frozen SFT reference), weighted by an **adaptive**
   coefficient beta (see below), not a fixed one.
6. Backprop and step the optimizer every `--grad_accum_steps` documents
   (default **8**, matching DSP's own `batch_size=8`), averaging that many
   documents' gradients together before each update. The loop itself stays
   strictly sequential/on-policy regardless of this value -- it only changes
   how often the optimizer moves, never how the data is generated or scored.

Both the trainable 0.5B policy, a frozen SFT-checkpoint reference copy, and
the 8B model stay loaded on the GPU for the **entire** run -- no
unload/reload between documents or epochs. Prints are tagged plainly by
epoch, e.g. `[epoch 1] doc 40/400 -- running mean reward 0.31, beta 0.02104`.

**Cost, explicit:** this queries the 8B model `epochs x docs x group_size`
times. A 4-epoch run over the full 400-doc train set at `group_size=5`
means ~8,000 8B-model generations, not ~2,000 -- expect roughly `epochs`x
the wall-clock time of a single pass over the same documents. **Always test
with `--limit` on a small doc count first** to get a real per-epoch timing
before committing GPU hours to a full run.

### Adaptive KL coefficient (DSP Eq. 7-8)

Instead of one fixed `--kl_coef` for the whole run, beta is re-tuned after
every optimizer step:

```
e_t      = clip((KL_t - kl_target) / kl_target, -0.2, 0.2)
beta_t+1 = beta_t * (1 + k_beta * e_t)
```

`KL_t` is the mean k3-estimator KL over the documents in that step's
accumulation window, measured against the frozen SFT reference. If KL runs
above `--kl_target`, beta grows and pulls harder toward the reference; if it
runs below target, beta shrinks and lets the policy explore more.
`--k_beta` (default 0.1) sets the reaction rate. Pass `--no_adaptive_kl` to
fall back to a single fixed coefficient (`--beta0`) for the whole run, or
`--no_kl` to disable the KL term entirely.

"Best epoch" tracking uses a held-out `data/dev.txt` validation pass at the
end of each epoch (one greedy completion per dev doc -> fresh 8B extraction
-> reward, batched since it's read-only). This is a genuine held-out
signal -- dev docs are never trained on -- unlike the training-set reward
used elsewhere in the epoch loop. If `data/dev.txt` isn't found, GRPO
warns and falls back to the (weaker) training-set mean reward instead;
pass `--no_dev_validation` to force that fallback deliberately. Whichever
epoch's dev reward is highest gets saved to `--best_out_dir`, separately
from the final epoch's checkpoint in `--out_dir`.

## How Phase 3 actually works: true online GRPO

For every training document, every epoch, strictly in sequence:
1. Sample `group_size` keyword completions from the **current** 0.5B policy.
2. Score every one of them with a **fresh** 8B-model extraction (no reuse
   across epochs -- the 8B model sees each document once per epoch, not
   once total).
3. Compute the group-relative advantage from those rewards.
4. Compute each completion's log-probability under the same policy that
   just generated it -- since there's no lag between sampling and training,
   this is on-policy REINFORCE with a group baseline (no PPO ratio/clip
   needed, unlike an earlier buffer-based design that trained multiple
   epochs against a fixed, aging rollout).
5. Optionally penalize KL(policy || frozen reference-at-epoch-0).
6. Backprop and step the optimizer -- by default after *every single
   document* (`--grad_accum_steps 1`), matching "one sample in, one update,
   next sample" exactly. Raise `--grad_accum_steps` to batch a few
   documents' gradients together if per-sample updates prove too noisy,
   without changing the sequential, one-doc-at-a-time nature of the loop.

Both the trainable 0.5B policy, a frozen reference copy of it, and the 8B
model stay loaded on the GPU for the **entire** run -- no unload/reload
between documents or epochs.

**Cost, explicit:** this queries the 8B model `epochs x docs x group_size`
times. A 5-epoch run over the full 400-doc train set at `group_size=5`
means ~10,000 8B-model generations, not ~2,000 -- expect roughly `epochs`x
the wall-clock time of a single pass over the same documents. **Always test
with `--limit` on a small doc count first** to get a real per-epoch timing
before committing GPU hours to a full run.

"Best epoch" tracking uses a held-out `data/dev.txt` validation pass at the
end of each epoch (one greedy completion per dev doc -> fresh 8B extraction
-> reward, batched since it's read-only). This is a genuine held-out
signal -- dev docs are never trained on -- unlike the training-set reward
used elsewhere in the epoch loop. If `data/dev.txt` isn't found, Phase 3
warns and falls back to the (weaker) training-set mean reward instead;
pass `--no_dev_validation` to force that fallback deliberately. Whichever
epoch's dev reward is highest gets saved to `--best_out_dir`, separately
from the final epoch's checkpoint in `--out_dir`.

## Run-to-run variance

Two runs with identical config can still differ: GPU backward passes
(especially through attention) aren't bit-for-bit deterministic even with
`torch.manual_seed()` set, and this was compounded badly in an earlier
per-document-update design where two identical-config runs produced
opposite-sign results. `torch.backends.cudnn.deterministic = True` removes
one source of that variance; `--grad_accum_steps > 1` removes more of it by
averaging several documents' gradients before each update (at the cost of
delayed, less literal per-sample updates). Neither guarantees bit-identical
runs. Don't trust a single run's before/after numbers as strong evidence --
rerun with a different `--seed` and look at the spread before writing up a
result, especially at `--grad_accum_steps 1`.

## Running it

```bash
pip install -r requirements.txt   # see it for CUDA-version notes
```

```bash
tmux new -s grpo
# Smoke test FIRST -- confirms the whole SFT+GRPO loop works end to end and
# gives you a real per-doc timing before you commit to a full run:
python run_pipeline.py --stage all --limit 10 --dev_limit 10 --test_limit 10 \
    --sft_limit 10 --sft_epochs 1 --group_size 3 --epochs 2 --grad_accum_steps 2

# Full run:
python run_pipeline.py --stage all --sft_epochs 3 --group_size 5 --epochs 4 \
    --grad_accum_steps 8
# detach: Ctrl-b d ; reattach later: tmux attach -t grpo
```
or fully detached:
```bash
nohup python run_pipeline.py --stage all --sft_epochs 3 --group_size 5 \
    --epochs 4 --grad_accum_steps 8 > /dev/null 2>&1 &
tail -f outputs/run.log
```

Re-run GRPO only, reusing an SFT checkpoint already on disk:
```bash
python run_pipeline.py --stage phase3,phase4,phase4_best,phase5 \
    --epochs 4 --lr 1e-5 --seed 1
```

Run `python run_pipeline.py --help` for the full flag list. Key ones:
`--sft_epochs` / `--sft_lr` / `--sft_batch_size` / `--sft_limit` /
`--sft_out_dir` / `--skip_sft`, `--group_size`, `--temperature`, `--epochs`,
`--lr`, `--beta0` / `--kl_target` / `--k_beta` / `--no_adaptive_kl` /
`--no_kl`, `--grad_accum_steps`, `--limit` / `--test_limit`, `--seed`,
`--out_dir` / `--best_out_dir`, `--no_track_best`, `--log_file`,
`--dev_split` / `--dev_limit` / `--no_dev_validation` (held-out best-epoch
selection -- put your dev set at `data/dev.txt`, or point `--dev_split`
elsewhere).

Model weights download once via Hugging Face Hub into `~/.cache/huggingface`
(or wherever `HF_HOME` points) and are reused across runs.

## Notes / simplifications
- `data_utils.load_conll` uses a heuristic detokenizer to rebuild readable
  text from the token stream for the prompts; spot-check a few documents if
  exact surface form matters for your writeup.
- `metrics.py`'s scoring is independent of AlpaPICO's `metric.py` (no
  dependency on that repo), same set-based P/R/F1-over-spans family, plus
  the normalization described above.
- To train a different single element instead of Population, change
  `prompts.KEYWORD_ELEMENT`.
- To move to training multiple/separate elements (e.g. 4 policies, or 4
  LoRA adapters on one base model), see the discussion earlier in this
  project's history -- not implemented here.
