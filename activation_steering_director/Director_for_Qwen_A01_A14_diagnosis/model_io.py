"""Model and tokenizer loading with greedy decoding configuration."""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def _check_qwen3_transformers_version(model_id: str) -> None:
    """Qwen3 (Qwen3ForCausalLM) is only natively supported from transformers>=4.51.0;
    on older versions AutoModelForCausalLM either fails to resolve the architecture
    or silently mis-maps it. This is a version check only -- it does not special-case
    any DIRECTER behaviour."""
    if "qwen3" not in model_id.lower():
        return
    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        return
    if (major, minor) < (4, 51):
        raise RuntimeError(
            f"Qwen3 models require transformers>=4.51.0, found {transformers.__version__}. "
            "Please upgrade: pip install -U transformers"
        )


def load_model_and_tokenizer(cfg):
    _check_qwen3_transformers_version(cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mapping in span localisation."
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map=cfg.device,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model.eval()

    # Enforce greedy decoding (DIRECTER assumes argmax distribution checks)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.update(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    return model, tokenizer