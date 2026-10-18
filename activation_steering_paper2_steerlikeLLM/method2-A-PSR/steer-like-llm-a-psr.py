# %% [markdown]
# ## **Imports**

# %%
#!pip install -U transformers

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
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
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
# ## **A-PSR**

# %%
class SPSR(nn.Module):
    """Per-layer Prompt Steering Replacement module (Eq. 7-8)."""
    def __init__(self, dim, dtype=torch.float32):
        super().__init__()
        self.z = nn.Parameter(torch.randn(dim, dtype=dtype) * 0.01)
        self.w = nn.Parameter(torch.randn(dim, dtype=dtype) * 0.01)
        self.b = nn.Parameter(torch.ones(1, dtype=dtype) * 0.5)  # start biased "on"

    def coefficient(self, h):
        w = self.w.to(device=h.device, dtype=h.dtype)
        b = self.b.to(device=h.device, dtype=h.dtype)
        lam = F.relu(torch.matmul(h, w) + b)
        return lam.unsqueeze(-1)

    def forward(self, h, alpha):
        z = self.z.to(device=h.device, dtype=h.dtype)
        lam = self.coefficient(h)
        return h + alpha * lam * z, lam

class MultiLayerSteering:
    """Attaches one SPSR module + forward hook per decoder layer (A-PSR).

    Hooks fire sequentially during the model's own forward pass, so layer l's
    intervention naturally sees the already-steered output of layers < l --
    this is what A-PSR requires (Sec. 3.4).
    """
    def __init__(self, model, layer_indices, hidden_dim, dtype=torch.float32):
        self.layers = nn.ModuleList()
        self.hooks = []
        self.enabled = False
        self.alpha = 0.0

        for idx in layer_indices:
            layer_module = model.model.layers[idx]
            layer_device = next(layer_module.parameters()).device
            spsr = SPSR(hidden_dim, dtype=dtype).to(layer_device)
            spsr.last_lambda = None
            self.layers.append(spsr)

            def make_hook(spsr_module):
                def hook_fn(module, inputs, output):
                    if not self.enabled:
                        return output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        hidden_states, lam = spsr_module(hidden_states, self.alpha)
                        spsr_module.last_lambda = lam
                        return (hidden_states,) + output[1:]
                    else:
                        hidden_states, lam = spsr_module(output, self.alpha)
                        spsr_module.last_lambda = lam
                        return hidden_states
                return hook_fn

            self.hooks.append(layer_module.register_forward_hook(make_hook(spsr)))

    def parameters(self):
        return self.layers.parameters()

    def all_lambdas(self):
        return [l.last_lambda for l in self.layers if l.last_lambda is not None]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# %%
# A-PSR steers every layer, jointly optimized (Sec. 3.4).
STEER_LAYERS = list(range(NUM_LAYERS))
print(f"Hooking all {len(STEER_LAYERS)} layers (A-PSR)")

steering = MultiLayerSteering(model, STEER_LAYERS, HIDDEN_DIM, dtype=torch.float32)

n_trainable = sum(p.numel() for p in steering.parameters() if p.requires_grad)
print(f"Trainable A-PSR params: {n_trainable}  ({len(STEER_LAYERS)} layers x (2*{HIDDEN_DIM}+1))")

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

# %%
TRAIN_FACTORS = [0.25, 0.5, 0.1, 1.5 ,2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
LR = 0.02
EPOCHS = 10
GRAD_CLIP = 1.0
LAMBDA_REG = 0.01   # weight of the Sec. 3.5 regularization term; set to 0.0 to disable

optimizer = torch.optim.AdamW(steering.parameters(), lr=LR, weight_decay=0.0)

def run_forward(input_ids, attention_mask, labels, alpha, lambda_reg=LAMBDA_REG):
    steering.enabled = True
    steering.alpha = alpha
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss

    if lambda_reg > 0:
        reg_total = 0.0
        for lam in steering.all_lambdas():
            resp_mask = (labels != -100).to(device=lam.device, dtype=lam.dtype).unsqueeze(-1)
            lam_sum = (lam * resp_mask).sum()
            reg_total = reg_total + torch.clamp(1.0 - lam_sum, min=0.0).to(loss.device)
        loss = loss + lambda_reg * (reg_total / len(steering.layers))  # averaged across layers

    return loss

@torch.no_grad()
def evaluate(examples, alpha):
    steering.enabled = True
    steering.alpha = alpha
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
        torch.nn.utils.clip_grad_norm_(steering.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_losses.append(loss.item())

    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    # validate at the middle of the training factor range
    val_loss = evaluate(val_examples, alpha=TRAIN_FACTORS[len(TRAIN_FACTORS) // 2])
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    print(f"epoch {epoch+1:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  (skipped {skipped_steps} bad steps so far)")

# %% [markdown]
# ## **Best Steering Vector**

# %%
candidate_factors = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
factor_scores = {}
for f in candidate_factors:
    factor_scores[f] = evaluate(val_examples, alpha=f)
    print(f"alpha={f:>5.1f}  val_loss={factor_scores[f]:.4f}")

BEST_ALPHA = min(factor_scores, key=factor_scores.get)
print(f"\nBest steering factor: {BEST_ALPHA}")

# %% [markdown]
# ## **Generation**

# %%
@torch.no_grad()
def generate(prompt, alpha=None, max_new_tokens=80):
    steering.enabled = alpha is not None
    steering.alpha = alpha if alpha is not None else 0.0
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    steering.enabled = False
    return text

# %%
for ex in val_examples[:3]:
    print("=" * 80)
    print("PROMPT:", ex["prompt"].strip())
    print("-" * 80)
    print("UNSTEERED (no hook, no PICO instruction):")
    print(generate(ex["prompt"], alpha=None))
    print("-" * 80)
    print(f"STEERED (A-PSR, alpha={BEST_ALPHA}):")
    print(generate(ex["prompt"], alpha=BEST_ALPHA))
    print("-" * 80)
    print("GROUND TRUTH:")
    print(ex["target"].replace(tokenizer.eos_token, ""))

# %%
@torch.no_grad()
def inspect_lambda(example, alpha, layer_idx=None):
    layer_idx = STEER_LAYERS[len(STEER_LAYERS) // 2] if layer_idx is None else layer_idx
    pos = STEER_LAYERS.index(layer_idx)

    input_ids, attention_mask, labels = build_batch(example)
    steering.enabled = True
    steering.alpha = alpha
    _ = model(input_ids=input_ids, attention_mask=attention_mask)
    lam = steering.layers[pos].last_lambda[0, :, 0].detach().float().cpu().numpy()
    steering.enabled = False

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    print(f"--- layer {layer_idx} ---")
    for tok, l in zip(tokens, lam):
        print(f"{tok!r:>20}  lambda={l:.4f}")

inspect_lambda(val_examples[0], alpha=BEST_ALPHA)

# %%
torch.save(
    {
        "layers": {
            idx: {"z": spsr.z.detach().cpu(), "w": spsr.w.detach().cpu(), "b": spsr.b.detach().cpu()}
            for idx, spsr in zip(STEER_LAYERS, steering.layers)
        },
        "best_alpha": BEST_ALPHA,
        "model_name": MODEL_NAME,
        "method": "A-PSR_LL",
    },
    "pico_apsr_vectors.pt",
)
print("Saved pico_apsr_vectors.pt")

# %%



