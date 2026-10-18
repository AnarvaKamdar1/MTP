# %% [markdown]
# ## **Installs**

# %%
!pip install -U transformers

# %% [markdown]
# ## **Hugging Face Token**

# %%
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("my_hf_token")

# %%
from huggingface_hub import login
login(token=hf_token)

# %% [markdown]
# ## **Imports**

# %%
import json
import re
import math
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc, traceback

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
MODEL_NAME = "google/gemma-2b"
HF_TOKEN = hf_token

# %% [markdown]
# ## **Load Model**

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float16,      # NOT torch_dtype (deprecated, as you already found)
    device_map="auto",
)
model.eval()

DEVICE = next(model.parameters()).device
NUM_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
print(f"Loaded {MODEL_NAME}: {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}, device={DEVICE}")

# %%
# freeze model params

for p in model.parameters():
    p.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params in base model (should be 0): {trainable}")

# %% [markdown]
# ## **Dataset Loading**

# %%
df = pd.read_csv("/kaggle/input/datasets/anarvaaa/dataset-for-ssv/pico_dataset.csv")  # update path if running locally
# If running outside Kaggle, point this at your local copy instead, e.g.:
# df = pd.read_csv("pico_dataset.csv")
print(df.shape)
df.head(3)

# %%
def format_target(row):
    return (
        f"P: {row['P']}\n"
        f"I: {row['I']}\n"
        f"C: {row['C']}\n"
        f"O: {row['O']}"
    )

def format_prompt(row):
    # Deliberately minimal / un-instructed prompt. We are NOT telling the model
    # to produce PICO format via prompting -- the steering vector is what should
    # push the model toward that structure.
    return f"{row['research_question'].strip()}\n"

examples = []
for _, row in df.iterrows():
    examples.append({
        "prompt": format_prompt(row),
        "target": format_target(row) + tokenizer.eos_token,
    })

random.shuffle(examples)
n_val = max(5, int(0.1 * len(examples)))
val_examples = examples[:n_val]
train_examples = examples[n_val:]
print(f"train={len(train_examples)}  val={len(val_examples)}")
print("--- sample ---")
print(train_examples[0]["prompt"])
print(train_examples[0]["target"])

# %% [markdown]
# ## **S-PSR: Single-Layer Prompt Steering Replacement**
#
# Paper reference: "Steer Like the LLM" (arXiv:2605.03907), Eq. 7-8.
#
# Instead of a constant additive vector (h <- h + alpha*w1 + b1), S-PSR estimates
# a *token-specific* steering coefficient lambda(h) from the activations
# themselves, via a single-layer ReLU probe, and scales a single steering
# direction z by it:
#
#     lambda(h_i) = ReLU(h_i . w + b)            (Eq. 8, scalar per token i)
#     h_i <- h_i + alpha * lambda(h_i) * z        (Eq. 7)
#
# So there are now three learned vectors instead of two:
#   - z : the steering direction (what gets added)
#   - w, b : the probe that decides *how much* to add, per token, based on
#            that token's own (pre-steering) activation.
#
# alpha remains the global steering coefficient (a training-time/inference-time
# hyperparameter), same role as before.

# %%
class SPSR(nn.Module):
    """Single-layer Prompt Steering Replacement module (Eq. 7-8)."""
    def __init__(self, dim, dtype=torch.float32):
        super().__init__()
        self.z = nn.Parameter(torch.randn(dim, dtype=dtype) * 0.01)   # steering vector z_attr,l
        self.w = nn.Parameter(torch.randn(dim, dtype=dtype) * 0.01)   # probe weight w_attr,l
        self.b = nn.Parameter(torch.zeros(1, dtype=dtype))            # probe bias b_attr,l

    def coefficient(self, h):
        """lambda(h) = ReLU(h . w + b), computed per token. h: [..., dim] -> [..., 1]"""
        w = self.w.to(device=h.device, dtype=h.dtype)
        b = self.b.to(device=h.device, dtype=h.dtype)
        lam = F.relu(torch.matmul(h, w) + b)
        return lam.unsqueeze(-1)

    def forward(self, h, alpha):
        z = self.z.to(device=h.device, dtype=h.dtype)
        lam = self.coefficient(h)          # [..., 1], per-token scalar
        steered = h + alpha * lam * z
        return steered, lam


class SteeringHook:
    """Forward hook that injects an SPSR intervention into a decoder layer's output.

    `enabled=False` acts as a no-op passthrough, so the exact same model object
    can be used for steered vs. unsteered ("prompting-only") generation.
    Stores the last computed per-token lambda (`last_lambda`) so the training
    loop can apply the Sec. 3.5 regularization term.
    """
    def __init__(self, spsr: SPSR):
        self.sv = spsr
        self.enabled = False
        self.alpha = 0.0
        self.handle = None
        self.last_lambda = None

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states, lam = self.sv(hidden_states, self.alpha)
            self.last_lambda = lam
            return (hidden_states,) + output[1:]
        else:
            hidden_states, lam = self.sv(output, self.alpha)
            self.last_lambda = lam
            return hidden_states

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)
        return self.handle

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# %%
# Pick an intervention layer. Middle-ish layers tend to encode more abstract /
# format-level concepts (see Fig. 13 in the paper -- steerability varies by
# layer). Gemma-2B has NUM_LAYERS layers; we default to roughly the 40% mark.
STEER_LAYER = max(1, int(NUM_LAYERS * 0.4))

# With device_map="auto" the model can be sharded across multiple GPUs, so the
# layer we hook may not live on the same device as model.parameters()[0]
# (which is what DEVICE points to). Find the *actual* device of the hooked
# layer and put the S-PSR module there -- the hook itself also casts on
# every call as a safety net, but starting on the right device avoids
# unnecessary cross-GPU copies.
layer_device = next(model.model.layers[STEER_LAYER].parameters()).device
print(f"Hooking layer index {STEER_LAYER} / {NUM_LAYERS} (on {layer_device})")

# NOTE: keep S-PSR params in float32 even though the model runs in float16.
# This is the standard "fp32 master weights" pattern for mixed precision:
# AdamW's default eps=1e-8 underflows to 0 once cast into fp16 (fp16's
# smallest normal value is ~6e-5), which silently produces NaNs via
# division-by-~0 in the Adam update. fp32 params + fp32 optimizer state
# avoids this; SPSR.forward already casts down to h.dtype only for the
# actual addition into the model's fp16 hidden states.
psr = SPSR(HIDDEN_DIM, dtype=torch.float32).to(layer_device)
steering_hook = SteeringHook(psr)
steering_hook.attach(model.model.layers[STEER_LAYER])

n_trainable = sum(p.numel() for p in psr.parameters() if p.requires_grad)
print(f"Trainable S-PSR params: {n_trainable}  (z, w: {HIDDEN_DIM} each, b: 1)")

# %% [markdown]
# ## **Batch Construction**

# %%
def build_batch(example, device=DEVICE):
    prompt_ids = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=True).input_ids[0]
    target_ids = tokenizer(example["target"], return_tensors="pt", add_special_tokens=False).input_ids[0]

    input_ids = torch.cat([prompt_ids, target_ids], dim=0)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100  # only train on the PICO-formatted continuation

    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels

# %% [markdown]
# ## **Training**
#
# Training objective: loglikelihood (S-PSR_LL in the paper), i.e. plain
# cross-entropy on the PICO-formatted continuation, same as before. On top of
# that we add the Sec. 3.5 regularization term, which discourages lambda from
# collapsing into the dead ReLU region across every response token:
#
#     L_reg = max(0, 1 - sum_i lambda(A_{y'_i}))
#
# computed only over the response (labeled) tokens.

# %%
TRAIN_FACTORS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
LR = 0.04
EPOCHS = 12
GRAD_CLIP = 1.0
LAMBDA_REG = 0.01   # weight of the Sec. 3.5 regularization term; set to 0.0 to disable

optimizer = torch.optim.AdamW(psr.parameters(), lr=LR, weight_decay=0.0)

def run_forward(input_ids, attention_mask, labels, alpha, lambda_reg=LAMBDA_REG):
    steering_hook.enabled = True
    steering_hook.alpha = alpha
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss

    if lambda_reg > 0:
        lam = steering_hook.last_lambda  # [batch, seq, 1], computed on this forward pass
        resp_mask = (labels != -100).to(lam.dtype).unsqueeze(-1)  # only response tokens
        lam_sum = (lam * resp_mask).sum()
        reg = torch.clamp(1.0 - lam_sum, min=0.0)
        loss = loss + lambda_reg * reg

    return loss

@torch.no_grad()
def evaluate(examples, alpha):
    steering_hook.enabled = True
    steering_hook.alpha = alpha
    losses = []
    for ex in examples:
        input_ids, attention_mask, labels = build_batch(ex)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(out.loss.item())
    return float(np.mean(losses))

# %%
history = {"train_loss": [], "val_loss": []}
skipped_steps = 0

for epoch in range(EPOCHS):
    random.shuffle(train_examples)
    epoch_losses = []
    for ex in train_examples:
        input_ids, attention_mask, labels = build_batch(ex)
        alpha = random.choice(TRAIN_FACTORS)

        optimizer.zero_grad()
        loss = run_forward(input_ids, attention_mask, labels, alpha)

        if not torch.isfinite(loss):
            # Defensive: skip a bad step rather than let a NaN gradient poison
            # the Adam moment estimates for every subsequent step.
            skipped_steps += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(psr.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_losses.append(loss.item())

    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    # validate at the middle of the training factor range
    val_loss = evaluate(val_examples, alpha=TRAIN_FACTORS[len(TRAIN_FACTORS) // 2])
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    print(f"epoch {epoch+1:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  (skipped {skipped_steps} bad steps so far)")

# %% [markdown]
# ## **Best Steering Factor**

# %%
candidate_factors = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
factor_scores = {}
for f in candidate_factors:
    factor_scores[f] = evaluate(val_examples, alpha=f)
    print(f"alpha={f:>5.1f}  val_loss={factor_scores[f]:.4f}")

BEST_ALPHA = min(factor_scores, key=factor_scores.get)
print(f"\nBest steering factor: {BEST_ALPHA}")

# %% [markdown]
# ## **Helper to check steered vs unsteered**

# %%
@torch.no_grad()
def generate(prompt, alpha=None, max_new_tokens=80):
    steering_hook.enabled = alpha is not None
    steering_hook.alpha = alpha if alpha is not None else 0.0

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    steering_hook.enabled = False
    return text

# %%
for ex in val_examples[:3]:
    print("=" * 80)
    print("PROMPT:", ex["prompt"].strip())
    print("-" * 80)
    print("UNSTEERED (no hook, no PICO instruction):")
    print(generate(ex["prompt"], alpha=None))
    print("-" * 80)
    print(f"STEERED (S-PSR, alpha={BEST_ALPHA}):")
    print(generate(ex["prompt"], alpha=BEST_ALPHA))
    print("-" * 80)
    print("GROUND TRUTH:")
    print(ex["target"].replace(tokenizer.eos_token, ""))

# %% [markdown]
# ## **Inspect per-token lambda (sanity check)**
#
# Quick look at which tokens the probe decides to steer strongly on, for one
# validation example -- this is the S-PSR analogue of Figure 2/5 in the paper.

# %%
@torch.no_grad()
def inspect_lambda(example, alpha):
    input_ids, attention_mask, labels = build_batch(example)
    steering_hook.enabled = True
    steering_hook.alpha = alpha
    _ = model(input_ids=input_ids, attention_mask=attention_mask)
    lam = steering_hook.last_lambda[0, :, 0].detach().float().cpu().numpy()
    steering_hook.enabled = False

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    for tok, l in zip(tokens, lam):
        print(f"{tok!r:>20}  lambda={l:.4f}")

inspect_lambda(val_examples[0], alpha=BEST_ALPHA)

# %% [markdown]
# ## **Save S-PSR Module**

# %%
torch.save(
    {
        "z": psr.z.detach().cpu(),
        "w": psr.w.detach().cpu(),
        "b": psr.b.detach().cpu(),
        "layer": STEER_LAYER,
        "best_alpha": BEST_ALPHA,
        "model_name": MODEL_NAME,
        "method": "S-PSR_LL",
    },
    "pico_spsr_vector.pt",
)
print("Saved pico_spsr_vector.pt")

# %%
