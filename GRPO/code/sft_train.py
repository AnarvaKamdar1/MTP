"""SFT warm-start for the 0.5B policy model, using train.txt's gold BIO-tag
spans as the supervised target -- run this BEFORE the GRPO stage.

This is the "supervised fine-tuning" half of the DSP (Directional Stimulus
Prompting, NeurIPS 2023) recipe: DSP first trains its small policy model on
(input, pseudo-stimulus) pairs before ever touching RL, because starting RL
from a cold, generic-instruct policy forces it to learn output FORMAT and
content QUALITY at the same time from a noisy, sparse reward -- slow and
unstable. We have something DSP didn't: real gold spans (not textrank-derived
pseudo-labels), so the SFT target here is exact, not heuristic.

For every training document:
    prompt  = keyword_prompt(doc.text)              # same prompt GRPO uses
    target  = "Population: span1 | span2"            # from doc.gold_spans
              (or "Population: none" if the doc has no gold spans)

Plain next-token cross-entropy on the target tokens only (the prompt tokens
are masked out of the loss with label = -100), a few epochs, small LR. The
resulting checkpoint is what phase3_grpo_train.py should load as its
--keyword_checkpoint for BOTH the trainable policy and its frozen reference
copy, so GRPO's KL penalty is measured against this warm-started policy, not
the raw base model (matching DSP: RL fine-tunes pi initialized from the SFT
policy, and the KL term keeps it near that same SFT policy).

The policy is loaded with fp32 master weights (model_utils.load_keyword_model
does this whenever trainable=True) and the forward pass runs under
bf16_autocast() -- training bf16 weights directly with a LR this small
(2e-5) silently rounds most optimizer updates away, since bf16 can't
represent a step that small at typical weight magnitudes.

max_length defaults to 1536 (raised from an earlier 768): the guideline
block + a full abstract can run long enough that 768 truncated a real slice
of training documents from the front, cutting into the chat header/
instructions rather than the (always-preserved) target.
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data_utils import load_conll
from prompts import keyword_prompt, KEYWORD_ELEMENT
from model_utils import load_keyword_model, free_model, bf16_autocast, KEYWORD_MODEL_ID


def _build_example(doc, tokenizer, element, max_length):
    """Returns (input_ids, labels) as python lists -- labels are -100 over the
    prompt span so the loss only sees the target tokens."""
    prompt = keyword_prompt(doc.text, element)
    spans = doc.gold_spans.get(element, [])
    target_text = f"{element}: " + (" | ".join(spans) if spans else "none")

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

    if len(prompt_ids) + len(target_ids) > max_length:
        # Truncate from the FRONT of the prompt (drop earliest abstract
        # tokens) so the target -- what we're actually training on -- always
        # survives intact.
        keep = max_length - len(target_ids)
        if keep <= 0:
            return None  # target alone is too long; skip (shouldn't happen in practice)
        prompt_ids = prompt_ids[-keep:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return input_ids, labels


def _collate(batch, pad_id, device):
    max_len = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, labs in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device),
            torch.tensor(attn, device=device))


def run(train_split="data/train.txt",
        checkpoint=KEYWORD_MODEL_ID,
        out_dir="outputs/sft_keyword_model",
        element=KEYWORD_ELEMENT,
        epochs=3, lr=2e-5, batch_size=8, max_length=1536,
        limit=None, seed=0, log_every=20):
    random.seed(seed)
    torch.manual_seed(seed)

    docs = load_conll(train_split)
    if limit:
        docs = docs[:limit]

    model, tokenizer = load_keyword_model(checkpoint, trainable=True)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    examples = []
    for doc in docs:
        ex = _build_example(doc, tokenizer, element, max_length)
        if ex is not None:
            examples.append(ex)
    print(f"SFT: {len(examples)}/{len(docs)} docs converted to (prompt, gold {element} span) "
          f"training pairs from {train_split}")

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        t0 = time.time()
        epoch_losses = []
        n_batches = (len(examples) + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch = examples[b * batch_size:(b + 1) * batch_size]
            input_ids, labels, attn = _collate(batch, pad_id, device)
            with bf16_autocast():
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

            if (b + 1) % log_every == 0 or (b + 1) == n_batches:
                mean_l = sum(epoch_losses) / len(epoch_losses)
                print(f"SFT epoch {epoch}/{epochs} batch {b+1}/{n_batches} -- "
                      f"running mean loss {mean_l:.4f} ({time.time()-t0:.0f}s elapsed)")

        mean_l = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"SFT epoch {epoch}/{epochs} DONE -- mean loss {mean_l:.4f}, {time.time()-t0:.0f}s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"SFT complete -- warm-started policy saved -> {out_dir}")

    free_model(model)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_split", default="data/train.txt")
    ap.add_argument("--checkpoint", default=KEYWORD_MODEL_ID)
    ap.add_argument("--out_dir", default="outputs/sft_keyword_model")
    ap.add_argument("--element", default=KEYWORD_ELEMENT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="Subsample train.txt for a smoke test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    run(train_split=args.train_split, checkpoint=args.checkpoint, out_dir=args.out_dir,
        element=args.element, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        max_length=args.max_length, limit=args.limit, seed=args.seed, log_every=args.log_every)
