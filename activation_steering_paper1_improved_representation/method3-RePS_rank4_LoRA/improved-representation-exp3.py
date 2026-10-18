# %% [markdown]
# ## **Installs**

# %%
#!pip install -U transformers

# %% [markdown]
# # **Common Sections**

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
import transformers
import gc, traceback

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
print("transformers version:", transformers.__version__)

# %% [markdown]
# # **Part 1: Generating positive and negative responses**

# %% [markdown]
# ## **Load Responses generating Model**

# %%
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = hf_token

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # no token needed, it's public
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

DEVICE = next(model.parameters()).device
NUM_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
print(f"Loaded {MODEL_NAME}: {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}, device={DEVICE}")

# %%
#freeze model params

for p in model.parameters():
    p.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params in base model (should be 0): {trainable}")

# %% [markdown]
# ## **Load Questions**

# %%
df = pd.read_csv("/kaggle/input/datasets/anarvaaa/dataset-for-ssv/pico_dataset_only_questions.csv")  # update path if running locally

print(df.shape)
df.head(3)

# %%
GEN_MAX_NEW_TOKENS = 100
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95

# %% [markdown]
# ## **Generate Positive Responses**

# %% [markdown]
# ### **Example Positive Prompt:**
# 
# Answer the following clinical research question by extracting it into PICO format 
# (Population, Intervention, Comparison, Outcome). Respond only with:
# P: ...
# I: ...
# C: ...
# O: ...
# 
# Question: In adults with out-of-hospital cardiac arrest (OHCA) and an initial 
# shockable rhythm, does real-time audiovisual feedback during CPR compared with 
# standard CPR without feedback improve survival to hospital discharge?
# 
# 

# %%
# code to generate positive responces from the model
POSITIVE_PROMPT_TEMPLATE = (
    "Answer the following clinical research question by extracting it into PICO format "
    "(Population, Intervention, Comparison, Outcome). Respond only with:\n"
    "P: ...\n"
    "I: ...\n"
    "C: ...\n"
    "O: ...\n\n"
    "Question: {question}"
)

def parse_pico(text):
    """Extract P/I/C/O fields from the model's raw response."""
    fields = {}
    for key in ["P", "I", "C", "O"]:
        m = re.search(rf"^{key}:\s*(.+)$", text, flags=re.MULTILINE)
        fields[key] = m.group(1).strip() if m else None
    return fields

@torch.no_grad()
def generate_positive(question, max_new_tokens=GEN_MAX_NEW_TOKENS):
    prompt = POSITIVE_PROMPT_TEMPLATE.format(question=question.strip())
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=GEN_DO_SAMPLE,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()

pico_fields = {"P": [], "I": [], "C": [], "O": []}
n_parse_failures = 0

for i, row in df.iterrows():
    raw = generate_positive(row["research_question"])
    fields = parse_pico(raw)
    for k in ["P", "I", "C", "O"]:
        pico_fields[k].append(fields[k])
    if any(v is None for v in fields.values()):
        n_parse_failures += 1
        print(f"[WARN] row {i}: could not parse all PICO fields. Raw output:\n{raw}\n")
    if (i + 1) % 5 == 0 or (i + 1) == len(df):
        print(f"[{i+1}/{len(df)}] done")

print(f"\nTotal parse failures: {n_parse_failures}/{len(df)}")

# %% [markdown]
# ## **Generate Negative Responses**

# %% [markdown]
# ### **Example Negative Prompt:**
# 
# In adults with out-of-hospital cardiac arrest (OHCA) and an initial shockable 
# rhythm, does real-time audiovisual feedback during CPR compared with standard 
# CPR without feedback improve survival to hospital discharge?

# %%
# code to generate negative responses from the model
@torch.no_grad()
def generate_negative(question, max_new_tokens=GEN_MAX_NEW_TOKENS):
    messages = [{"role": "user", "content": question.strip()}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=GEN_DO_SAMPLE,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()

negative_responses = []
for i, row in df.iterrows():
    neg = generate_negative(row["research_question"])
    negative_responses.append(neg)
    if (i + 1) % 5 == 0 or (i + 1) == len(df):
        print(f"[{i+1}/{len(df)}] done")
        print("  Q:", row["research_question"][:100])
        print("  neg:", neg[:150].replace("\n", " "))

# %% [markdown]
# ## **Save new dataset**

# %%
# code which adds the positive repsonces and negative responses and create a new dataset -> pico_dataset_positive_negative.csv
# save this in kaggle/working
# before moving to part 2, this will be uploaded into the dataset entity so it can be used later

df_out = df.copy()
df_out["P"] = pico_fields["P"]
df_out["I"] = pico_fields["I"]
df_out["C"] = pico_fields["C"]
df_out["O"] = pico_fields["O"]
df_out["negative_response"] = negative_responses

# drop any rows where PICO parsing failed, so Part 2 doesn't choke on NaNs
before = len(df_out)
df_out = df_out.dropna(subset=["P", "I", "C", "O"]).reset_index(drop=True)
print(f"Dropped {before - len(df_out)} rows with unparsed PICO fields.")

out_path = "/kaggle/working/pico_dataset_positive_negative.csv"
df_out.to_csv(out_path, index=False)
print(f"Saved {df_out.shape} to {out_path}")
df_out.head(3)

# %% [markdown]
# # **Part 2: Training Steering Vector**

# %% [markdown]
# ## **Load Steering Vector training Model**

# %%
MODEL_NAME = "google/gemma-2b-it"
HF_TOKEN = hf_token

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
#freeze model params

for p in model.parameters():
    p.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params in base model (should be 0): {trainable}")

# %% [markdown]
# ## **Loading Positive and Negative Dataset**

# %%
df = pd.read_csv("/kaggle/input/datasets/anarvaaa/dataset-for-ssv/pico_dataset_positive_negative.csv")  # update path if running locally
print(df.shape)
df.head(3)

# %% [markdown]
# ## **Train-Test Split and Prompt, Positive and Negative Triplet**

# %%
def format_positive(row):
    return (
        f"P: {row['P']}\n"
        f"I: {row['I']}\n"
        f"C: {row['C']}\n"
        f"O: {row['O']}"
    )

# Whether to wrap the raw question in Gemma's chat turn markers. Deliberately does
# NOT add any "extract this into PICO format" instruction -- that would let the
# prompt itself do the steering, confounding whether the *vector* is responsible
# for the PICO-shaped output. The only thing IT models get beyond the base-model
# prompt is the turn structure they were tuned to expect; the content is identical.
IS_INSTRUCT_MODEL = "-it" in MODEL_NAME.lower()

def format_prompt(row):
    question = row['research_question'].strip()
    if IS_INSTRUCT_MODEL:
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    return f"{question}\n"

examples = []
for _, row in df.iterrows():
    examples.append({
        "prompt": format_prompt(row),
        "positive": format_positive(row) + tokenizer.eos_token,
        "negative": str(row["negative_response"]).strip() + tokenizer.eos_token,
    })

random.shuffle(examples)
n_val = max(5, int(0.1 * len(examples)))
val_examples = examples[:n_val]
train_examples = examples[n_val:]
print(f"train={len(train_examples)}  val={len(val_examples)}")
print(f"IS_INSTRUCT_MODEL={IS_INSTRUCT_MODEL}")
print("--- sample ---")
print("PROMPT:", train_examples[0]["prompt"])
print("POSITIVE:", train_examples[0]["positive"])
print("NEGATIVE:", train_examples[0]["negative"])


# %% [markdown]
# ## **Steering Vector**

# %%
class Rank4LoRAModule(nn.Module):
    def __init__(self, dim, rank=4, dtype=torch.float32):
        super().__init__()
        self.rank = rank
        self.w1 = nn.Parameter(torch.randn(dim, rank, dtype=dtype) * 0.01)
        self.w2 = nn.Parameter(torch.zeros(dim, rank, dtype=dtype))

    def forward(self, h, alpha=1.0):
        w1 = self.w1.to(device=h.device, dtype=h.dtype)
        w2 = self.w2.to(device=h.device, dtype=h.dtype)
        # Scale by rank (1 / self.rank) to prevent activation explosion
        delta = torch.matmul(torch.matmul(h, w1), w2.T) / self.rank
        return h + alpha * delta


class MultiLayerLoRAHook:
    """Hooks Rank-4 LoRA adapters across multiple decoder layers (e.g. layers 5, 10, 15, 20)."""
    def __init__(self, lora_modules: nn.ModuleDict):
        self.lora_modules = lora_modules
        self.enabled = False
        self.alpha = 1.0
        self.handles = []

    def make_hook(self, layer_idx_str):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            mod = self.lora_modules[layer_idx_str]
            if isinstance(output, tuple):
                hidden_states = mod(output[0], alpha=self.alpha)
                return (hidden_states,) + output[1:]
            else:
                return mod(output, alpha=self.alpha)
        return hook

    def attach(self, model):
        for layer_str, mod in self.lora_modules.items():
            layer_idx = int(layer_str)
            handle = model.model.layers[layer_idx].register_forward_hook(self.make_hook(layer_str))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []




# %%
# Define 4 target layers as specified in Table 5 for Gemma-2-2B
TARGET_LAYERS = [5, 10, 15]
lora_modules_dict = nn.ModuleDict()

for l_idx in TARGET_LAYERS:
    layer_device = next(model.model.layers[l_idx].parameters()).device
    lora_modules_dict[str(l_idx)] = Rank4LoRAModule(HIDDEN_DIM, rank=4, dtype=torch.float32).to(layer_device)

lora_hook = MultiLayerLoRAHook(lora_modules_dict)
lora_hook.attach(model)

n_trainable = sum(p.numel() for p in lora_modules_dict.parameters() if p.requires_grad)
print(f"Hooked layers {TARGET_LAYERS}. Total trainable Rank-4 LoRA params: {n_trainable}")

# %% [markdown]
# ## **Preference Objective**

# %%
def seq_logprob(prompt_ids, response_ids, alpha, device, use_hook=True):
    """Length-normalized log p(response | prompt).
    use_hook=False -> reference (unsteered) model likelihood.
    """
    input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    lora_hook.enabled = use_hook
    lora_hook.alpha = alpha
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    P = prompt_ids.shape[0]
    R = response_ids.shape[0]
    resp_logp = token_logp[:, P - 1 : P - 1 + R]
    lora_hook.enabled = False   # always leave disabled after use
    return resp_logp.sum() / R


def build_pair(example, device):
    prompt_ids = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=True).input_ids[0]
    pos_ids = tokenizer(example["positive"], return_tensors="pt", add_special_tokens=False).input_ids[0]
    neg_ids = tokenizer(example["negative"], return_tensors="pt", add_special_tokens=False).input_ids[0]
    return prompt_ids, pos_ids, neg_ids

# %% [markdown]
# ## **Training**

# %%
TRAIN_FACTORS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
LR = 1e-3
EPOCHS = 8
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.00
PATIENCE = 2

optimizer = torch.optim.AdamW(lora_modules_dict.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def precompute_reference_logprobs(examples, device):
    """log p(y|x) and log p(y^c|x) under frozen reference model."""
    cache = {}
    for i, ex in enumerate(examples):
        prompt_ids, pos_ids, neg_ids = build_pair(ex, device)
        logp_pos_ref = seq_logprob(prompt_ids, pos_ids, alpha=0.0, device=device, use_hook=False)
        logp_neg_ref = seq_logprob(prompt_ids, neg_ids, alpha=0.0, device=device, use_hook=False)
        cache[i] = (logp_pos_ref.item(), logp_neg_ref.item())
    return cache

def reps_loss(example, ref_logps, alpha, device):
    prompt_ids, pos_ids, neg_ids = build_pair(example, device)
    logp_pos_ref, logp_neg_ref = ref_logps

    # beta weights from reference model (Eq. 5 / Eq. 6)
    beta_pos = max(logp_neg_ref - logp_pos_ref, 1.0)
    beta_neg = max(logp_pos_ref - logp_neg_ref, 1.0)

    # Delta+ : positive factor (+alpha) under Phi_LoRA
    logp_pos_steer = seq_logprob(prompt_ids, pos_ids, alpha=+alpha, device=device, use_hook=True)
    logp_neg_steer = seq_logprob(prompt_ids, neg_ids, alpha=+alpha, device=device, use_hook=True)
    delta_pos = beta_pos * logp_pos_steer - logp_neg_steer

    # Delta- : negative factor (-alpha) for LoRA RePS objective (Sec. 3.3)
    logp_neg_null = seq_logprob(prompt_ids, neg_ids, alpha=-alpha, device=device, use_hook=True)
    logp_pos_null = seq_logprob(prompt_ids, pos_ids, alpha=-alpha, device=device, use_hook=True)
    delta_neg = beta_neg * logp_neg_null - logp_pos_null

    loss = -F.logsigmoid(delta_pos) - F.logsigmoid(delta_neg)
    return loss, delta_pos.item(), delta_neg.item()

@torch.no_grad()
def evaluate(examples, ref_cache, alpha, device):
    losses, margins = [], []
    for i, ex in enumerate(examples):
        loss, d_pos, d_neg = reps_loss(ex, ref_cache[i], alpha, device)
        if torch.isfinite(loss) and math.isfinite(d_pos):
            losses.append(loss.item())
            margins.append(d_pos)
            
    if not losses:
        return float("nan"), float("nan"), 0.0
        
    acc = float(np.mean([m > 0 for m in margins]))
    return float(np.mean(losses)), float(np.mean(margins)), acc

# %%
main_device = next(model.model.layers[TARGET_LAYERS[0]].parameters()).device
print("Precomputing reference (unsteered) log-probs...")
train_ref_cache = precompute_reference_logprobs(train_examples, main_device)
val_ref_cache = precompute_reference_logprobs(val_examples, main_device)

history = {"train_loss": [], "val_loss": [], "val_margin": [], "val_acc": []}
best_val_loss = float("inf")
best_state = None
epochs_since_improve = 0

for epoch in range(EPOCHS):
    idx = list(range(len(train_examples)))
    random.shuffle(idx)
    epoch_losses = []
    for i in idx:
        ex = train_examples[i]
        alpha = random.choice(TRAIN_FACTORS)

        optimizer.zero_grad()
        loss, _, _ = reps_loss(ex, train_ref_cache[i], alpha, main_device)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_modules_dict.parameters(), GRAD_CLIP)
        optimizer.step()
        epoch_losses.append(loss.item())

    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    val_loss, val_margin, val_acc = evaluate(val_examples, val_ref_cache,
                                              alpha=TRAIN_FACTORS[len(TRAIN_FACTORS)//2],
                                              device=main_device)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_margin"].append(val_margin)
    history["val_acc"].append(val_acc)
    print(f"epoch {epoch+1:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"val_margin={val_margin:+.4f}  val_pref_acc={val_acc:.2f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(lora_modules_dict.state_dict())
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

if best_state is not None:
    lora_modules_dict.load_state_dict(best_state)
    print(f"Restored best checkpoint (val_loss={best_val_loss:.4f})")

# %% [markdown]
# ## **Best Steering Vector**

# %%
@torch.no_grad()
def generate(prompt, alpha=None, max_new_tokens=100):
    """Greedy-decode a continuation, optionally with LoRA adapters enabled."""
    lora_hook.enabled = alpha is not None
    lora_hook.alpha = alpha if alpha is not None else 0.0

    inputs = tokenizer(prompt, return_tensors="pt").to(main_device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    lora_hook.enabled = False
    return text


def extract_pico_fields(text):
    """Pull P:/I:/C:/O: lines out of a generation."""
    fields = {}
    for key in ["P", "I", "C", "O"]:
        m = re.search(rf"^\s*{key}\s*:\s*(.+)$", text, flags=re.MULTILINE)
        fields[key] = m.group(1).strip() if m else ""
    return fields


def word_f1(a, b):
    a_words = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    b_words = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    if not a_words or not b_words:
        return 0.0
    inter = len(a_words & b_words)
    precision = inter / len(b_words)
    recall = inter / len(a_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def repetition_penalty(text, n=3):
    tokens = text.split()
    if len(tokens) <= n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def pico_quality_score(generated, ground_truth_positive):
    gt_fields = extract_pico_fields(ground_truth_positive)
    gen_fields = extract_pico_fields(generated)

    present = sum(1 for k in gt_fields if gt_fields[k])
    if present == 0:
        return 0.0
    format_score = sum(1 for k in gt_fields if gt_fields[k] and gen_fields[k]) / present

    sims = [word_f1(gt_fields[k], gen_fields[k]) for k in gt_fields if gt_fields[k]]
    content_score = float(np.mean(sims)) if sims else 0.0

    fluency_score = 1.0 - repetition_penalty(generated)

    eps = 1e-4
    scores = [format_score + eps, content_score + eps, fluency_score + eps]
    harmonic = len(scores) / sum(1.0 / s for s in scores)
    return harmonic

# %%
random.shuffle(val_examples)
mid = max(1, len(val_examples) // 2)
sel_examples = val_examples[:mid]
hold_examples = val_examples[mid:] if len(val_examples) > mid else val_examples[:mid]

print(f"Selecting factor on {len(sel_examples)} example(s), "
      f"holding out {len(hold_examples)} example(s) for reporting.")

# Candidate inference factors specified for LoRA in Table 11
candidate_factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]

baseline_scores = [pico_quality_score(generate(ex["prompt"], alpha=None), ex["positive"])
                    for ex in sel_examples]
baseline_quality = float(np.mean(baseline_scores))
print(f"unsteered baseline pico_quality={baseline_quality:.4f}\n")

factor_scores = {}
for f in candidate_factors:
    scores = [pico_quality_score(generate(ex["prompt"], alpha=f), ex["positive"])
              for ex in sel_examples]
    factor_scores[f] = float(np.mean(scores))
    print(f"alpha={f:>5.1f}  pico_quality={factor_scores[f]:.4f}")

BEST_ALPHA = max(factor_scores, key=factor_scores.get)
print(f"\nBest steering factor (by held-out generation quality): {BEST_ALPHA} "
      f"(pico_quality={factor_scores[BEST_ALPHA]:.4f} vs. baseline={baseline_quality:.4f})")

if factor_scores[BEST_ALPHA] <= baseline_quality:
    print("WARNING: best steered score does not beat the unsteered baseline on the "
          "selection split -- inspect generations below before trusting BEST_ALPHA.")

# %% [markdown]
# ## **Generation**

# %%
for ex in hold_examples:
    print("=" * 80)
    print("PROMPT:", ex["prompt"].strip())
    print("-" * 80)

    unsteered = generate(ex["prompt"], alpha=None)
    steered = generate(ex["prompt"], alpha=BEST_ALPHA)

    print("UNSTEERED:")
    print(unsteered)
    print(f"  pico_quality={pico_quality_score(unsteered, ex['positive']):.4f}")
    print("-" * 80)
    print(f"STEERED (learned Rank-4 LoRA, alpha={BEST_ALPHA}):")
    print(steered)
    print(f"  pico_quality={pico_quality_score(steered, ex['positive']):.4f}")
    print("-" * 80)
    print("POSITIVE (ground truth):")
    print(ex["positive"].replace(tokenizer.eos_token, ""))

# %%
torch.save(
    {
        "lora_state_dict": lora_modules_dict.state_dict(),
        "target_layers": TARGET_LAYERS,
        "rank": 4,
        "best_alpha": BEST_ALPHA,
        "model_name": MODEL_NAME,
        "objective": "LoRA RePS (positive vs negative sampling)",
    },
    "/kaggle/working/pico_lora_r4_exp3.pt",
)
print("Saved /kaggle/working/pico_lora_r4_exp3.pt")


