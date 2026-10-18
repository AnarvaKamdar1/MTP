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


# %% [markdown]
# ## **Preference Objective**

# %%


# %% [markdown]
# ## **Training**

# %%


# %%


# %% [markdown]
# ## **Best Steering Vector**

# %%


# %%


# %% [markdown]
# ## **Generation**

# %%


# %%



