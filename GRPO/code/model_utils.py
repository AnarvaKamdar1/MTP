"""Load/unload helpers for the two models.

Targets a single, sufficiently large GPU -- no multi-GPU / CPU-offload
sharding. Both models are placed explicitly on one device via `.to(DEVICE)`
rather than `device_map="auto"`, so nothing is ever silently offloaded to
CPU or split across devices you didn't ask for. This also means free_model's
`.to("cpu")` step works cleanly (no "accelerate hooks" warning, which only
happens with device_map-dispatched models).

Phases still load one model at a time by default and call free_model()
between them, even though there's headroom to hold both at once -- keeps
peak VRAM low and predictable on a shared server.
"""
import os

# Must be set before huggingface_hub / transformers are imported to take
# effect. Quiets model-download progress bars and drops transformers'
# internal log level so things like the "torch_dtype is deprecated" notice
# don't print.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
# huggingface_hub's HTTP client (httpx) logs every request at INFO, which
# floods a run_pipeline.py log configured at INFO level. Silence just these
# libraries' loggers -- doesn't touch anything this project logs itself.
for _name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    if hasattr(_hf_logging, "disable_progress_bar"):
        _hf_logging.disable_progress_bar()
except Exception:
    pass  # best-effort -- exact API differs across transformers versions

KEYWORD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BIG_MODEL_ID = "Qwen/Qwen3-8B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def free_model(*objs):
    """Drop references and clear CUDA cache/allocator state. Only touches the
    objects you pass in -- never scans or clears anything else on the GPU, so
    it's safe to call on a shared server. Call this between phases (or
    between loading the 0.5B and 8B models) to actually release VRAM before
    loading the next model."""
    for obj in objs:
        try:
            obj.to("cpu")
        except Exception:
            pass
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _prep_tokenizer(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched decoder-only generation
    return tokenizer


def load_keyword_model(checkpoint=KEYWORD_MODEL_ID, device=DEVICE, trainable=False, fp32=False):
    """Load the 0.5B model on the (single) GPU.

    trainable=True (the optimized policy) or fp32=True (its frozen KL
    reference copy) keep the MASTER weights in fp32. This matters: AdamW's
    per-step update is on the order of the learning rate (1e-5 to 2e-5),
    which is smaller than a bf16 weight's representable precision at typical
    magnitudes (bf16 has ~8 bits of mantissa, so e.g. a weight around 0.02
    can only represent steps of about 8e-5) -- training bf16 weights
    directly silently rounds most optimizer updates back to where they
    started. Keeping fp32 master weights and running forward/generate calls
    under `bf16_autocast()` (below) gives bf16-speed/memory activations
    without losing the weight update itself. The frozen 8B model never gets
    fp32=True -- it's never trained, so this doesn't apply to it."""
    tokenizer = _prep_tokenizer(AutoTokenizer.from_pretrained(checkpoint))
    dtype = torch.float32 if (trainable or fp32) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
    if trainable:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tokenizer


def load_big_model(checkpoint=BIG_MODEL_ID, device=DEVICE):
    """Load the 8B model entirely on the (single) GPU. Inference only -- never
    trained, so it stays bf16 (no fp32 master-weight concern applies)."""
    tokenizer = _prep_tokenizer(AutoTokenizer.from_pretrained(checkpoint))
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def bf16_autocast():
    """Wrap forward/generate calls on an fp32-master-weight model (see
    load_keyword_model) in this so activations still run in bf16 for
    speed/memory -- standard mixed-precision pattern. Not needed around the
    frozen 8B model, which is loaded in bf16 directly."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=200, temperature=0.0,
                    num_return_sequences=1, top_p=0.95, chat_template_kwargs=None):
    """Chat-template generation for a batch of user prompts. temperature<=0 -> greedy.
    chat_template_kwargs is forwarded to apply_chat_template -- e.g. pass
    {"enable_thinking": False} for Qwen3 models, otherwise the model burns its
    whole max_new_tokens budget on a <think>...</think> block and the actual
    answer never gets generated.

    max_length=4096 (raised from an earlier 2048): the 8B extraction prompt
    (full guideline block + abstract + keyword hints) can run to roughly
    1800-2200 tokens on its own, so 2048 risked right-truncating the chat
    template's closing assistant-turn marker on longer documents -- the
    model would then just continue the prompt text instead of answering."""
    device = next(model.parameters()).device
    ctk = chat_template_kwargs or {}
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True, **ctk)
        for p in prompts
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=4096, add_special_tokens=False).to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**enc, **gen_kwargs)
    completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return completions
