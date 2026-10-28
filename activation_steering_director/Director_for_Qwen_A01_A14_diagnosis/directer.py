"""DIRECTER: KV-cache steering with activation capture for PICO analysis."""

import os
from typing import Dict, List, Optional, Tuple

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
        unk_id = getattr(tokenizer, "unk_token_id", None)
        stops = {tokenizer.eos_token_id}
        for turn_end in ("<|eot_id|>", "<|im_end|>"):
            tid = tokenizer.convert_tokens_to_ids(turn_end)
            if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                stops.add(tid)
        self.stop_ids = {s for s in stops if s is not None}
        self._states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._hooks = []

    # ---------------- Attention Hooks ----------------

    def _hook_fn(self, module, args, kwargs, result):
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
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        top2 = probs.topk(2, dim=-1).values.flatten()
        return bool(top2[1] >= self.cfg.plausibility * top2[0])

    def _is_plausible(self, raw_logits: torch.Tensor, steered_logits: torch.Tensor) -> bool:
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
        keep = len(ranked) // (2 ** trial)
        return ranked[:keep]

    # ---------------- Layer Ranking ----------------

    def _rank_layers(self, token_id, cache, steerer, base_len, raw_states) -> List[int]:
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

        scores = torch.stack(rows)
        order = torch.argsort(scores, descending=True)
        ranked = [int(i) for i in order.tolist()]
        if self.cfg.verbose:
            print(f"  [DIRECTER] Top-8 sensitive layers: {ranked[:8]}")
        return ranked

    # ---------------- Activation Dumping ----------------

    def _dump_activation_bundle(
        self,
        bundle: Dict[str, torch.Tensor],
        meta: Dict,
        doc_tag: str,
        element_letter: str,
        do_steer: bool,
        layer: int,
    ) -> None:
        """Save A01-A14 as one layer-specific .pt bundle.

        Tensor shapes are intentionally not forced to match:
          A01-A08: [n_subtokens, hidden]
          A09-A10: [hidden]
          A11-A12: [n_tokens, hidden]
          A13-A14: [hidden]
        """
        os.makedirs(self.cfg.activation_dir, exist_ok=True)
        safe_doc = doc_tag.replace("/", "_").replace(" ", "_")
        status = "steered" if do_steer else "unsteered"
        payload = {
            "activations": bundle,
            "metadata": {
                "doc_id": str(doc_tag),
                "element": element_letter,
                "layer": layer,
                "status": status,
                "activation_ids": sorted(bundle.keys()),
                "activation_text": meta.get("text", {}),
                "activation_token_spans": meta.get("spans", {}),
                "prompt_token_ids": meta.get("token_ids", []),
                "notes": {
                    "A01-A08": "Only the task-specific name/letter pair exists in each prompt: Population uses Participant/(P), Intervention uses Intervention/(I), Comparator uses Control/(C), and Outcome uses Outcome/(O). Missing task pairs are intentionally absent.",
                    "A09": "Last prompt token fed separately immediately before first generated-token prediction.",
                    "A10": "Last token in the abstract span.",
                    "A11": "Full rendered input sequence self-attention output.",
                    "A12": "Abstract-only span self-attention output.",
                    "A13": "Steered A09 probe from the accepted DIRECTER layer subset; absent if no steered probe was accepted.",
                    "A14": "A13 - A09, same layer and hidden dimension; absent when A13 is unavailable.",
                },
            },
        }
        fname = f"{safe_doc}_{status}_L{layer}_{element_letter}.pt"
        path = os.path.join(self.cfg.activation_dir, fname)
        torch.save(payload, path)
        if self.cfg.verbose:
            present = [k for k in bundle if bundle[k] is not None]
            print(f"  [DIRECTER] Saved {present} -> {path}")

    # ---------------- Autoregressive Generation ----------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        token_spans: List[Tuple[int, int]],
        activation_meta: Optional[Dict] = None,
        doc_tag: Optional[str] = None,
        element_letter: Optional[str] = None,
    ) -> str:
        input_ids = input_ids.to(self.device)

        do_steer = self.cfg.enable_steering
        capture_layers = [l for l in self.cfg.activation_layers if 0 <= l < self.n_layers]
        do_capture = bool(self.cfg.save_activations and capture_layers and doc_tag and element_letter)

        prefill_len = input_ids.shape[1] - 1
        need_hooks = do_steer or do_capture
        if need_hooks:
            self._attach_hooks()

        # With hooks attached, this captures the distributed prompt representation used
        # for A11/A12. A09 is captured by the first raw forward below and is the last
        # prompt token, which is intentionally processed separately from prefill.
        self._states = []
        out = self.model(input_ids[:, :-1], use_cache=True, return_dict=True)
        cache = out.past_key_values
        prefill_states = list(self._states) if do_capture else None

        if isinstance(cache, tuple):
            cache = DynamicCache.from_legacy_cache(cache)

        next_token = input_ids[:, -1:]
        steerer = (
            SpanSteerer(
                cache,
                token_spans,
                key_scale=self.cfg.key_scale_factor,
                value_scale=self.cfg.value_scale_factor,
            )
            if do_steer
            else None
        )

        ranked: Optional[List[int]] = None
        generated: List[int] = []

        # Activation bundle is initialized from prefill states. A13/A14 are populated
        # after the first accepted steered probe, and are only defined for that first
        # assistant-start state so that the comparison is clean and aligned.
        layer_bundles: Dict[int, Dict[str, torch.Tensor]] = {}
        if do_capture and prefill_states:
            for l in capture_layers:
                post = prefill_states[l][1][0]  # [prefill_seq, hidden]
                spans = (activation_meta or {}).get("spans", {})
                bundle = {}
                for aid, (s, e) in spans.items():
                    if not aid.startswith("A") or aid in {"A13", "A14"}:
                        continue
                    if aid == "A11" and e > prefill_len:
                        # A11 is completed with A09 below; keep the prefill prefix here.
                        bundle[aid] = post[s:prefill_len].cpu().clone()
                    elif e <= prefill_len:
                        selected = post[s:e]
                        if selected.shape[0] == 1 and aid in {f"A{i:02d}" for i in range(1, 9)}:
                            selected = selected[0]
                        bundle[aid] = selected.cpu().clone()
                    else:
                        bundle[aid] = None
                layer_bundles[l] = bundle
        attempt_count = 0
        accept_count = 0
        try:
            for step_idx in range(self.cfg.max_new_tokens):
                base_len = cache.get_seq_length()

                # Raw forward pass. At step 0, next_token is the final prompt token,
                # producing the base assistant-start state (A09).
                self._states = []
                raw_out = self.model(input_ids=next_token, past_key_values=cache, use_cache=True)
                raw_logits = raw_out.logits
                step_states = list(self._states)

                if do_capture and step_idx == 0:
                    for l in capture_layers:
                        post = step_states[l][1][0, -1, :].cpu().clone()
                        layer_bundles[l]["A09"] = post

                        # Complete A11 with the final prompt-token state (A09).
                        current = layer_bundles[l].get("A11")
                        if current is not None:
                            layer_bundles[l]["A11"] = torch.cat([current, post.unsqueeze(0)], dim=0)

                final_logits = raw_logits

                if do_steer:
                    raw_states = step_states if ranked is None else None
                    need_rank = ranked is None
                    need_probe = self._steering_can_matter(raw_logits)
                    if need_rank or need_probe:
                        parked = pop_last_position(cache, base_len)

                        if need_rank:
                            ranked = self._rank_layers(next_token, cache, steerer, base_len, raw_states)

                        if need_probe:
                            attempt_count+=1
                            candidates = list(ranked)
                            trial = 0
                            while candidates and trial <= self.cfg.max_trial:
                                steerer.apply(candidates)
                                self._states = []
                                probe_out = self.model(
                                    input_ids=next_token,
                                    past_key_values=cache,
                                    use_cache=True,
                                )
                                steered_logits = probe_out.logits
                                steered_states = list(self._states)
                                steerer.restore()
                                crop_to(cache, base_len)

                                if self._is_plausible(raw_logits, steered_logits):
                                    accept_count+=1
                                    final_logits = steered_logits
                                    # A13/A14 are deliberately captured only for the
                                    # accepted DIRECTER probe and at step 0.
                                    if do_capture and step_idx == 0:
                                        for l in capture_layers:
                                            base_vec = layer_bundles[l]["A09"]
                                            steered_vec = steered_states[l][1][0, -1, :].cpu().clone()
                                            layer_bundles[l]["A13"] = steered_vec
                                            layer_bundles[l]["A14"] = steered_vec - base_vec
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
            if need_hooks:
                self._detach_hooks()

        if do_capture and layer_bundles:
            for l in capture_layers:
                # A13/A14 are optional by design when the first step did not run an
                # accepted steered probe (e.g. plausibility gate did not trigger).
                self._dump_activation_bundle(
                    layer_bundles[l],
                    activation_meta or {},
                    doc_tag,
                    element_letter,
                    do_steer,
                    l,
                )
        if self.cfg.verbose and do_steer and element_letter:
            rate = accept_count / attempt_count if attempt_count else 0.0
            print(f"[DIAG] doc={doc_tag} elem={element_letter} tokens={len(generated)} attempts={attempt_count} accepted={accept_count} accept_rate={rate:.2f}")
        return self.tokenizer.decode(generated, skip_special_tokens=True)
