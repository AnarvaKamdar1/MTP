"""GRPO Policy Optimization -- TRUE online, run AFTER sft_train.py.

For every training document, every epoch, strictly in sequence:
  1. Sample `group_size` keyword completions from the CURRENT small-model
     policy (the model's weights as they stand at this exact moment).
  2. Score each completion with a FRESH extraction from the 8B model.
  3. Compute the group-relative advantage from those `group_size` rewards.
  4. Compute each sampled completion's log-probability under the same
     (current) policy that just generated it -- no staleness, so this is
     on-policy REINFORCE with a group baseline; no PPO ratio/clip is needed
     (unlike a buffer-based design that trains multiple epochs against
     samples collected once and therefore needs importance-sampling
     correction).
  5. Add a KL(pi || pi_SFT) penalty (k3 estimator) against a frozen
     reference copy of the SFT-warm-started policy, weighted by an
     ADAPTIVE coefficient beta that is itself updated every optimizer step
     (see "Adaptive KL" below) -- this is the DSP (Directional Stimulus
     Prompting, NeurIPS 2023) paper's Eq. 7-8 scheme, applied here directly
     instead of the earlier fixed --kl_coef.
  6. Backprop; step the optimizer every `grad_accum_steps` documents (default
     8, matching DSP's own batch_size=8), averaging those documents'
     gradients together before each step. The loop itself stays strictly
     sequential/on-policy either way -- grad_accum_steps only changes how
     often the optimizer moves, not how the data is generated or scored.

Both the trainable 0.5B policy, its frozen reference copy, and the 8B model
stay loaded on the single GPU for the ENTIRE run -- no unload/reload between
documents or epochs.

Cost tradeoff, explicit: this queries the 8B model `epochs * num_docs *
group_size` times, versus `num_docs * group_size` (once, total) if you only
ever ran a single offline pass. Test with --limit on a small doc count first
to get a real per-epoch time estimate before committing to a full run.

Adaptive KL (DSP Eq. 7-8)
-------------------------
Instead of one fixed KL coefficient for the whole run, beta is adjusted after
every optimizer step based on how far the OBSERVED KL this step landed from a
--kl_target you choose:

    e_t      = clip((KL_t - kl_target) / kl_target, -0.2, 0.2)
    beta_t+1 = beta_t * (1 + k_beta * e_t)

KL_t is the mean k3-estimator KL over the documents in that optimizer step's
accumulation window (measured against the frozen SFT reference). If KL runs
above target, beta grows and pulls harder toward the reference; if it runs
below target, beta shrinks and lets the policy explore more. k_beta (default
0.1) sets how fast beta reacts, matching the rate used in the RLHF work DSP
itself cites for this scheme. Pass --no_adaptive_kl to fall back to a single
fixed coefficient (--beta0, used unchanged for the whole run).

The k3 term itself is computed PER TOKEN then averaged (not once on the
sequence-mean log-ratio -- that variant underestimates the true KL by
Jensen's inequality), and both the policy's and the reference's log-probs
are computed with logits divided by the same `temperature` used to actually
sample the tokens (`_token_logprobs`), so neither the REINFORCE gradient nor
the KL estimate is silently scored under the wrong (temperature=1) sampling
distribution.

Precision: the trainable policy and its frozen reference both load with
fp32 MASTER weights (model_utils.load_keyword_model), with forward/generate
calls run under bf16_autocast() for speed/memory. Training bf16 weights
directly with a learning rate this small (1e-5) silently rounds most AdamW
updates away -- bf16 can't represent a step that small at typical weight
magnitudes, so a run can look like it trained while barely moving. The
frozen 8B model is unaffected (never trained, stays plain bf16).

"Best epoch" tracking uses a held-out `dev.txt` validation pass, run once at
the end of each epoch: the current policy generates ONE greedy completion
per dev doc, scored by the 8B model the same way as training, and averaged
into a mean dev reward -- a genuine held-out signal (dev docs are never
trained on), not a training-set proxy. Whichever epoch's dev reward is
highest gets saved to `--best_out_dir`, separately from the final epoch's
checkpoint in `--out_dir`. If no dev split is available, this falls back to
the (weaker, training-set) mean training reward instead -- see `--dev_split`.
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data_utils import load_conll
from prompts import keyword_prompt, extraction_prompt, KEYWORD_ELEMENT
from metrics import parse_model_output, reward_f1_element, reward_f1_element_shaped
from model_utils import (
    load_keyword_model, load_big_model, free_model, generate_batch, bf16_autocast,
    KEYWORD_MODEL_ID, BIG_MODEL_ID,
)


def _sample_group(policy, tokenizer, prompt, device, group_size, max_new_tokens, temperature, top_p=0.95):
    """Sample group_size completions from the CURRENT policy for one prompt.
    Works in raw token ids throughout (no text round-trip), so the later
    log-prob computation tokenizes exactly what was generated -- avoids a
    subtle tokenization mismatch that a decode-then-re-encode round trip
    could otherwise introduce."""
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_ids = enc.input_ids
    with torch.no_grad(), bf16_autocast():
        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[:, prompt_ids.shape[1]:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return prompt_ids, gen_ids, texts


def _token_logprobs(model, prompt_ids, gen_ids, eos_id, device, temperature=1.0):
    """Per-TOKEN log-prob (G, T) of gen_ids given the shared prompt_ids (1, P),
    under `model`, plus a validity mask (0 past each sequence's own EOS).
    Gradient-enabled iff called outside torch.no_grad() and model's params
    require grad -- callers control that at the call site (used with grad
    for the trainable policy, under no_grad for the frozen reference).

    Logits are divided by `temperature` before log_softmax, matching
    whatever temperature `_sample_group` actually sampled with -- computing
    log-probs at the implicit default (temperature=1) while having sampled
    at e.g. 0.8 would score every token under the WRONG distribution from
    the one that generated it, biasing both the REINFORCE gradient and the
    KL estimate. Returning per-token values (rather than already reducing to
    one number per sequence) is what lets the k3 KL estimator be applied
    per-token before averaging, instead of once to the sequence-mean
    log-ratio -- the two are not equivalent (Jensen's inequality means the
    mean-then-exp version systematically underestimates the true KL)."""
    G = gen_ids.shape[0]
    prompt_rep = prompt_ids.repeat(G, 1).to(device)
    gen_ids = gen_ids.to(device)
    full = torch.cat([prompt_rep, gen_ids], dim=1)
    with bf16_autocast():
        out = model(input_ids=full)
    logits = out.logits[:, prompt_rep.shape[1] - 1:-1, :].float() / temperature
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (G, T)

    # Mask out padding generated after each sequence's own EOS.
    mask = torch.ones_like(gen_ids, dtype=torch.float32)
    for i in range(G):
        eos_pos = (gen_ids[i] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            mask[i, eos_pos[0].item() + 1:] = 0

    return token_logp, mask


def _seq_mean(token_logp, mask):
    """Reduce (G, T) per-token values to one mean-per-sequence value (G,),
    respecting the EOS validity mask."""
    lengths = mask.sum(dim=1).clamp(min=1)
    return (token_logp * mask).sum(dim=1) / lengths


@torch.no_grad()
def _validate(policy, small_tok, big_model, big_tok, dev_docs, small_batch_size=8, big_batch_size=4,
              max_new_tokens_small=50, max_new_tokens_big=200):
    """One greedy pass over a held-out dev set: current policy -> keyword ->
    8B extraction -> reward, batched (unlike training, this is read-only so
    doesn't need the strict one-doc-at-a-time sequencing). Returns the mean
    KEYWORD_ELEMENT reward across dev_docs -- a genuine held-out signal,
    since these documents are never trained on."""
    policy.eval()
    keywords = []
    for start in range(0, len(dev_docs), small_batch_size):
        batch = dev_docs[start:start + small_batch_size]
        prompts = [keyword_prompt(d.text) for d in batch]
        with bf16_autocast():
            keywords.extend(generate_batch(policy, small_tok, prompts,
                                            max_new_tokens=max_new_tokens_small, temperature=0.0))
    policy.train()

    rewards = []
    for start in range(0, len(dev_docs), big_batch_size):
        batch_docs = dev_docs[start:start + big_batch_size]
        batch_kw = keywords[start:start + big_batch_size]
        ext_prompts = [extraction_prompt(d.text, kw) for d, kw in zip(batch_docs, batch_kw)]
        completions = generate_batch(big_model, big_tok, ext_prompts,
                                      max_new_tokens=max_new_tokens_big, temperature=0.0,
                                      chat_template_kwargs={"enable_thinking": False})
        for d, c in zip(batch_docs, completions):
            rewards.append(reward_f1_element(parse_model_output(c), d.gold_spans, KEYWORD_ELEMENT))
    return sum(rewards) / max(len(rewards), 1)


def run(train_split="data/train.txt",
        dev_split="data/dev.txt",
        keyword_checkpoint=KEYWORD_MODEL_ID,
        big_checkpoint=BIG_MODEL_ID,
        out_dir="outputs/phase3_trained_keyword_model",
        best_out_dir="outputs/phase3_best_keyword_model",
        epochs=5, lr=1e-5,
        use_kl_penalty=True, use_adaptive_kl=True,
        beta0=0.02, kl_target=0.2, k_beta=0.1,
        reward_shaping_weight=0.1,
        group_size=5, temperature=0.8, max_new_tokens_small=50, max_new_tokens_big=200,
        grad_accum_steps=8, limit=None, dev_limit=None, seed=0, log_every=20,
        track_best=True, use_dev_validation=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    docs = load_conll(train_split)
    if limit:
        docs = docs[:limit]
    total_calls = len(docs) * epochs * group_size
    print(f"GRPO: {len(docs)} docs x {epochs} epochs x group_size={group_size} = "
          f"{total_calls} total 8B-model calls (vs. {len(docs) * group_size} for a single pass)")
    if use_kl_penalty and use_adaptive_kl:
        print(f"adaptive KL enabled: beta0={beta0}, kl_target={kl_target}, k_beta={k_beta} "
              f"(DSP Eq. 7-8) -- beta will be re-tuned every optimizer step")
    if reward_shaping_weight:
        print(f"training reward shaping: {reward_shaping_weight} weight on a dense has-any-span "
              f"bonus, blended with exact-match F1 (training reward only -- eval/dev-selection "
              f"still use plain exact-match F1)")

    dev_docs = None
    if track_best and use_dev_validation:
        try:
            dev_docs = load_conll(dev_split)
            if dev_limit:
                dev_docs = dev_docs[:dev_limit]
            if not dev_docs:
                raise ValueError("dev split parsed to 0 documents")
            print(f"held-out dev validation: {len(dev_docs)} docs from {dev_split}, "
                  f"used for best-epoch selection (one greedy pass per epoch, not trained on)")
        except (FileNotFoundError, ValueError) as e:
            print(f"WARNING: couldn't load dev split ({dev_split}): {e}. "
                  f"Falling back to training-set mean reward for best-epoch tracking -- "
                  f"a weaker, non-held-out signal. Pass dev_split= or use_dev_validation=False "
                  f"to silence this.")
            dev_docs = None

    # Both the trainable policy AND its frozen reference load from the SAME
    # checkpoint -- normally the SFT warm-started checkpoint produced by
    # sft_train.py, so the KL penalty measures drift from the SFT policy,
    # not from the raw base model. Both keep fp32 master weights (policy via
    # trainable=True, reference via fp32=True) -- fp32=True on the reference
    # isn't about training it (it's never updated), it's so its log-probs are
    # computed at the same numeric precision as the policy's, rather than
    # comparing an fp32-computed value against a bf16-computed one in the KL.
    policy, small_tok = load_keyword_model(keyword_checkpoint, trainable=True)
    device = next(policy.parameters()).device

    reference = None
    if use_kl_penalty:
        reference, _ = load_keyword_model(keyword_checkpoint, trainable=False, fp32=True, device=device)

    big_model, big_tok = load_big_model(big_checkpoint)

    optimizer = AdamW(policy.parameters(), lr=lr)
    eos_id = small_tok.eos_token_id

    best_reward, best_epoch = float("-inf"), None
    beta = beta0

    for epoch in range(1, epochs + 1):
        random.shuffle(docs)
        optimizer.zero_grad()
        epoch_rewards, epoch_losses, epoch_kls = [], [], []
        n_opt_steps, n_degenerate = 0, 0
        window_kls = []
        t0 = time.time()

        for i, doc in enumerate(docs):
            prompt = keyword_prompt(doc.text)
            prompt_ids, gen_ids, texts = _sample_group(
                policy, small_tok, prompt, device, group_size, max_new_tokens_small, temperature)

            # Score every sampled completion with a FRESH 8B extraction --
            # this is the expensive, "true online" part.
            ext_prompts = [extraction_prompt(doc.text, t) for t in texts]
            completions = generate_batch(big_model, big_tok, ext_prompts,
                                          max_new_tokens=max_new_tokens_big, temperature=0.0,
                                          chat_template_kwargs={"enable_thinking": False})
            rewards = [reward_f1_element_shaped(parse_model_output(c), doc.gold_spans,
                                                 KEYWORD_ELEMENT, reward_shaping_weight)
                       for c in completions]
            epoch_rewards.extend(rewards)
            rewards_t = torch.tensor(rewards, device=device)

            std = rewards_t.std(unbiased=False)
            if std.item() < 1e-6:
                # All samples scored the same -- no gradient signal from this doc. Note this
                # does NOT `continue`: the grad_accum/last-doc boundary check below still has
                # to run every iteration regardless, otherwise a degenerate doc landing on the
                # very last document of an epoch could silently drop an already-accumulated,
                # not-yet-stepped gradient from earlier in that window (optimizer.zero_grad()
                # at the top of the next epoch would wipe it out unused).
                n_degenerate += 1
            else:
                advantages = (rewards_t - rewards_t.mean()) / (std + 1e-4)

                # Per-token, temperature-consistent with how these tokens were
                # actually sampled (see _token_logprobs docstring).
                cur_tok, mask = _token_logprobs(policy, prompt_ids, gen_ids, eos_id, device,
                                                 temperature=temperature)
                cur_logp = _seq_mean(cur_tok, mask)  # (G,) -- same reduction as before, for REINFORCE
                loss = -(advantages.detach() * cur_logp).mean()

                if use_kl_penalty:
                    with torch.no_grad():
                        ref_tok, _ = _token_logprobs(reference, prompt_ids, gen_ids, eos_id, device,
                                                      temperature=temperature)
                    # k3 estimator applied PER TOKEN, then averaged over valid
                    # tokens per sequence, then over the group -- not applied
                    # once to the sequence-mean log-ratio (that variant
                    # underestimates the true KL by Jensen's inequality).
                    log_ratio_tok = ref_tok - cur_tok
                    kl_tok = torch.exp(log_ratio_tok) - log_ratio_tok - 1
                    kl = _seq_mean(kl_tok, mask).mean()
                    loss = loss + beta * kl
                    epoch_kls.append(kl.item())
                    window_kls.append(kl.item())

                (loss / grad_accum_steps).backward()
                epoch_losses.append(loss.item())

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(docs):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                n_opt_steps += 1

                # Adaptive KL (DSP Eq. 7-8): re-tune beta from the KL observed
                # over this accumulation window, toward kl_target.
                if use_kl_penalty and use_adaptive_kl and window_kls:
                    kl_t = sum(window_kls) / len(window_kls)
                    e_t = max(-0.2, min(0.2, (kl_t - kl_target) / kl_target))
                    beta = beta * (1 + k_beta * e_t)
                    window_kls = []

            if (i + 1) % log_every == 0 or (i + 1) == len(docs):
                mean_r = sum(epoch_rewards) / len(epoch_rewards)
                beta_part = f", beta {beta:.5f}" if use_kl_penalty else ""
                print(f"[epoch {epoch}] doc {i+1}/{len(docs)} -- "
                      f"running mean reward {mean_r:.4f}{beta_part} ({time.time()-t0:.0f}s elapsed)")

        epoch_mean_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        epoch_mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        kl_part = f", mean KL {sum(epoch_kls)/len(epoch_kls):.4f}" if epoch_kls else ""
        print(f"[epoch {epoch}] DONE -- mean training reward {epoch_mean_reward:.4f}, "
              f"mean loss {epoch_mean_loss:.4f}{kl_part}, {n_opt_steps} optimizer steps, "
              f"{n_degenerate}/{len(docs)} docs had zero reward variance, "
              f"{time.time()-t0:.0f}s")

        if track_best:
            if dev_docs is not None:
                dev_reward = _validate(policy, small_tok, big_model, big_tok, dev_docs)
                print(f"[epoch {epoch}] held-out dev reward {dev_reward:.4f}")
                selection_metric, metric_name = dev_reward, "dev"
            else:
                selection_metric, metric_name = epoch_mean_reward, "training"

            if selection_metric > best_reward:
                best_reward, best_epoch = selection_metric, epoch
                Path(best_out_dir).mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(best_out_dir)
                small_tok.save_pretrained(best_out_dir)
                print(f"[epoch {epoch}] new best ({metric_name} reward {best_reward:.4f}) "
                      f"-> saved to {best_out_dir}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    small_tok.save_pretrained(out_dir)
    print(f"saved trained policy (final, epoch {epochs}) -> {out_dir}")
    if track_best:
        metric_name = "dev" if dev_docs is not None else "training-set"
        print(f"best epoch by {metric_name} reward: {best_epoch} "
              f"(reward {best_reward:.4f}) -> {best_out_dir}")

    to_free = [policy, small_tok, big_model, big_tok]
    if reference is not None:
        to_free.append(reference)
    free_model(*to_free)

    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_split", default="data/train.txt")
    ap.add_argument("--dev_split", default="data/dev.txt")
    ap.add_argument("--dev_limit", type=int, default=None, help="Subsample dev.txt for validation.")
    ap.add_argument("--no_dev_validation", action="store_true",
                     help="Skip dev-set validation even if dev.txt exists; falls back to "
                          "training-set mean reward for best-epoch tracking.")
    ap.add_argument("--keyword_checkpoint", default=KEYWORD_MODEL_ID,
                     help="Normally the SFT warm-start checkpoint from sft_train.py -- "
                          "both the trainable policy and its frozen KL-reference load from here.")
    ap.add_argument("--big_checkpoint", default=BIG_MODEL_ID)
    ap.add_argument("--out_dir", default="outputs/phase3_trained_keyword_model")
    ap.add_argument("--best_out_dir", default="outputs/phase3_best_keyword_model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--no_kl", action="store_true", help="Disable the KL penalty entirely.")
    ap.add_argument("--beta0", type=float, default=0.02, help="Initial KL coefficient.")
    ap.add_argument("--kl_target", type=float, default=0.2,
                     help="Target KL for the adaptive-beta controller (DSP Eq. 7-8).")
    ap.add_argument("--k_beta", type=float, default=0.1, help="Adaptive-beta reaction rate.")
    ap.add_argument("--no_adaptive_kl", action="store_true",
                     help="Use a single fixed KL coefficient (--beta0) for the whole run "
                          "instead of the adaptive controller.")
    ap.add_argument("--reward_shaping_weight", type=float, default=0.1,
                     help="Weight (0-1) on a dense has-any-span bonus blended with exact-match "
                          "F1 for the TRAINING reward only -- reduces all-zero-variance sampled "
                          "groups. Does not affect Phase 5 eval or dev-based best-epoch selection. "
                          "Pass 0 to disable and use plain exact-match F1 as before.")
    ap.add_argument("--group_size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens_small", type=int, default=50)
    ap.add_argument("--max_new_tokens_big", type=int, default=200)
    ap.add_argument("--grad_accum_steps", type=int, default=8,
                     help="Step the optimizer every this many documents (default 8, matching "
                          "DSP's own batch_size=8). The loop stays strictly sequential/on-policy "
                          "either way -- this only changes how often the optimizer moves.")
    ap.add_argument("--limit", type=int, default=None, help="Subsample train.txt for a timing/smoke test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--no_track_best", action="store_true")
    args = ap.parse_args()
    run(train_split=args.train_split, dev_split=args.dev_split, dev_limit=args.dev_limit,
        use_dev_validation=not args.no_dev_validation,
        keyword_checkpoint=args.keyword_checkpoint,
        big_checkpoint=args.big_checkpoint, out_dir=args.out_dir, best_out_dir=args.best_out_dir,
        epochs=args.epochs, lr=args.lr, use_kl_penalty=not args.no_kl,
        use_adaptive_kl=not args.no_adaptive_kl, beta0=args.beta0, kl_target=args.kl_target,
        k_beta=args.k_beta, reward_shaping_weight=args.reward_shaping_weight,
        group_size=args.group_size, temperature=args.temperature,
        max_new_tokens_small=args.max_new_tokens_small, max_new_tokens_big=args.max_new_tokens_big,
        grad_accum_steps=args.grad_accum_steps, limit=args.limit, seed=args.seed,
        log_every=args.log_every, track_best=not args.no_track_best)
