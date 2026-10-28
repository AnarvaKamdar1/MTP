"""Configuration module for DIRECTER PICO extraction."""

from dataclasses import dataclass


@dataclass
class DirecterConfig:
    # --- Model Settings ---
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda"
    # 'sdpa' (PyTorch native scaled dot-product attention) is fast and compatible:
    # our forward hooks read module I/O hidden states, not attention matrices.
    attn_implementation: str = "sdpa"

    # --- DIRECTER Hyperparameters (ICLR 2026 Paper Defaults) ---
    key_scale_factor: float = 100.0      # alpha (Key vector scaling magnitude)
    value_scale_factor: float = 1.0      # Paper fixes value scaling off for efficiency
    plausibility: float = 0.5            # beta (Plausibility acceptance threshold)
    max_trial: int = 10                  # Maximum layer halving trials per token

    # --- Generation Parameters ---
    max_new_tokens: int = 512

    # --- Instruction Span Scope ---
    steer_general_rules: bool = True     # Include shared rules (e.g., articles, noun phrases)
    steer_output_format: bool = False    # Keep output format prompt unsteered

    # --- Diagnostics ---
    verbose: bool = False