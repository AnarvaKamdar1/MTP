"""Model and tokenizer loading with greedy decoding configuration."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(cfg):
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