"""DIRECTER: KV-cache steering with attention-sensitivity layer ranking and a
plausibility-guided decoding loop.

Faithful to Algorithm 1 and Algorithm 2 of Kang & Kim (ICLR 2026).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from cache_ops import SpanSteerer, crop_to, pop_last_position, push_position


class DirecterGenerator:
    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = model.device
        self.n_layers = model.config.num_hidden_layers
        self.all_layers = list(range(self.n_layers))

        # Handle both standard EOS and Llama-3's turn terminator <|eot_id|>
        stops = {tokenizer.eos_token_id}
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            stops.add(eot)
        self.stop_ids = {s for s in stops if s is not None}

        self._states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._hooks = []

    # ---------------- Attention Hooks ----------------

    def _hook_fn(self, module, args, kwargs, result):
        # H_pre  = module input hidden state
        # H_post = module output hidden state
        pre = kwargs.get("hidden_states", args[0] if args else None)
        post = result[0] if isinstance(result, tuple) else result
        if pre is not None and post is not None:
            self._states.append((pre.detach().float(), post.detach().float()))

    def _attach_hooks(self):
        self._states = []
        for name, mod in self.model.named_modules():
            if "layers." in name and name.endswith(".self_attn"):
                self._hooks.append(
                    mod.register_forward_hook(self._hook_fn, with_kwargs=True)
                )

    def _detach_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._states = []

    # ---------------- Plausibility Machinery ----------------

    def _steering_can_matter(self, logits: torch.Tensor) -> bool:
        """The paper's gating condition (Section 3.2).

        If raw top-2 probability < beta * top-1 probability, no steered distribution
        could satisfy the plausibility constraint except when top-1 is unchanged.
        """
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        top2 = probs.topk(2, dim=-1).values.flatten()
        if self.cfg.verbose:
            ratio = float(top2[1] / top2[0])
            print(f"    [GATE] top1={float(top2[0]):.4f} top2={float(top2[1]):.4f} ratio={ratio:.4f} thresh={self.cfg.plausibility}")
        return bool(top2[1] >= self.cfg.plausibility * top2[0])

    def _is_plausible(self, raw_logits: torch.Tensor, steered_logits: torch.Tensor) -> bool:
        """Accepts iff p_raw[steered_top1] >= beta * p_raw[raw_top1] (Eq. 1)."""
        raw = raw_logits[:, -1, :].float()
        steered = steered_logits[:, -1, :].float()
        p_raw = F.softmax(raw, dim=-1)

        i_steered = steered.argmax(dim=-1)
        i_raw = raw.argmax(dim=-1)
        p_of_steered = p_raw.gather(-1, i_steered.unsqueeze(-1)).squeeze(-1)
        p_of_raw = p_raw.gather(-1, i_raw.unsqueeze(-1)).squeeze(-1)
        return bool((p_of_steered >= self.cfg.plausibility * p_of_raw).item())

    @staticmethod
    def _halve(ranked: List[int], trial: int) -> List[int]:
        """Keeps the top |L| / (2**trial) most sensitive layers."""
        keep = len(ranked) // (2 ** trial)
        return ranked[:keep]

    # ---------------- Layer Ranking ----------------

    def _rank_layers(self, token_id, cache, steerer, base_len, raw_states) -> List[int]:
        """One-time attention-sensitivity ranking (Eq. 2 & Eq. 6 in the paper).

        Sensitivity(l) ∝ sum_j [ dist(H_pre^j, H_post^{j,l}) + dist(H_pre^{j,l}, H_post^j) ]
        """
        rows = []
        for probe in self.all_layers:
            steerer.apply([probe])
            self._states = []
            self.model(input_ids=token_id, past_key_values=cache, use_cache=True)
            steered_states = self._states
            steerer.restore()
            crop_to(cache, base_len)

            per_layer = []
            for j in range(self.n_layers):
                h_pre, h_post = raw_states[j]
                s_pre, s_post = steered_states[j]
                d1 = 1.0 - F.cosine_similarity(h_pre, s_post, dim=-1)
                d2 = 1.0 - F.cosine_similarity(s_pre, h_post, dim=-1)
                per_layer.append((d1 + d2).mean())
            rows.append(torch.stack(per_layer).mean())

        scores = torch.stack(rows)  # [L]
        order = torch.argsort(scores, descending=True)  # Most sensitive layers first
        ranked = [int(i) for i in order.tolist()]

        if self.cfg.verbose:
            print(f"  [DIRECTER] Top-8 sensitive layers: {ranked[:8]}")
        return ranked

    # ---------------- Autoregressive Generation ----------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        token_spans: List[Tuple[int, int]],
        doc_tag: Optional[str] = None,
        element_letter: Optional[str] = None,
    ) -> str:
        input_ids = input_ids.to(self.device)

        # Prefill prompt up to token T-1
        out = self.model(input_ids[:, :-1], use_cache=True, return_dict=True)
        cache = out.past_key_values

        # Convert legacy tuple cache to HuggingFace DynamicCache
        if isinstance(cache, tuple):
            cache = DynamicCache.from_legacy_cache(cache)

        next_token = input_ids[:, -1:]

        steerer = SpanSteerer(
            cache,
            token_spans,
            key_scale=self.cfg.key_scale_factor,
            value_scale=self.cfg.value_scale_factor,
        )

        ranked: Optional[List[int]] = None
        generated: List[int] = []
        attempt_count = 0
        accept_count = 0
        self._attach_hooks()

        try:
            for _ in range(self.cfg.max_new_tokens):
                base_len = cache.get_seq_length()

                # 1. Raw forward pass (advances KV cache)
                self._states = []
                raw_out = self.model(input_ids=next_token, past_key_values=cache, use_cache=True)
                raw_logits = raw_out.logits
                raw_states = self._states if ranked is None else None

                need_rank = ranked is None
                need_probe = self._steering_can_matter(raw_logits)
                final_logits = raw_logits

                if need_rank or need_probe:
                    # Park the newly appended KV so probes evaluate on the pre-step state
                    parked = pop_last_position(cache, base_len)

                    # Step 1: Run one-time layer ranking and immediately drop hooks
                    if need_rank:
                        ranked = self._rank_layers(next_token, cache, steerer, base_len, raw_states)
                        self._detach_hooks()

                    # Plausibility-guided halving loop
                    if need_probe:
                        attempt_count += 1
                        candidates = list(ranked)
                        trial = 0
                        while candidates and trial <= self.cfg.max_trial:
                            steerer.apply(candidates)
                            steered_logits = self.model(
                                input_ids=next_token,
                                past_key_values=cache,
                                use_cache=True,
                            ).logits
                            steerer.restore()
                            crop_to(cache, base_len)

                            if self._is_plausible(raw_logits, steered_logits):
                                accept_count += 1
                                final_logits = steered_logits
                                break

                            trial += 1
                            candidates = self._halve(ranked, trial)

                    push_position(cache, parked)

                next_token = final_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tid = int(next_token.item())
                if tid in self.stop_ids:
                    break
                generated.append(tid)
        finally:
            self._detach_hooks()

        if self.cfg.verbose and element_letter:
            rate = accept_count / attempt_count if attempt_count else 0.0
            print(f"[DIAG] doc={doc_tag} elem={element_letter} tokens={len(generated)} attempts={attempt_count} accepted={accept_count} accept_rate={rate:.2f}")

        return self.tokenizer.decode(generated, skip_special_tokens=True)