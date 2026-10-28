"""Configuration module for DIRECTER PICO extraction."""

from dataclasses import dataclass, field


@dataclass
class DirecterConfig:
    # --- Model Settings ---
    # NOTE: Qwen3-8B requires transformers>=4.51.0 (native Qwen3ForCausalLM support).
    model_id: str = "Qwen/Qwen3-8B"
    device: str = "cuda"
    # 'sdpa' (PyTorch native scaled dot-product attention) is fast and compatible:
    # our forward hooks read module I/O hidden states, not attention matrices.
    attn_implementation: str = "sdpa"

    # --- Chat Template ---
    # Qwen3 emits a <think>...</think> reasoning block by default (enable_thinking=True
    # in its chat template). DIRECTER's steering/plausibility loop and the pipeline's
    # JSON parser both assume the very first generated content is the answer, so we
    # keep thinking mode off to preserve the same single-shot JSON output behaviour
    # used for Llama-3.1-8B-Instruct. Tokenizers whose chat template has no such
    # switch (e.g. Llama's) simply ignore this kwarg, so it's safe to pass for
    # either model family and only takes effect for Qwen3.
    enable_thinking: bool = False

    # --- DIRECTER Hyperparameters (ICLR 2026 Paper Defaults) ---
    key_scale_factor: float = 50.0      # alpha (Key vector scaling magnitude)
    value_scale_factor: float = 1.0      # Paper fixes value scaling off for efficiency
    plausibility: float = 0.7            # beta (Plausibility acceptance threshold)
    max_trial: int = 10                  # Maximum layer halving trials per token

    # --- Generation Parameters ---
    max_new_tokens: int = 512

    # --- Instruction Span Scope ---     # steer_general_rules changed to False, default was True
    steer_general_rules: bool = False     # Include shared rules (e.g., articles, noun phrases)
    steer_output_format: bool = False    # Keep output format prompt unsteered

    # --- Ablation / Interpretability ---
    enable_steering: bool = True          # False = plain greedy decoding, DIRECTER fully bypassed
    save_activations: bool = False        # Dump per-step self-attn output activations to disk
    activation_layers: list = field(default_factory=list)   # e.g. [0, 15]
    activation_dir: str = "activations"

    # --- Diagnostics ---
    verbose: bool = False