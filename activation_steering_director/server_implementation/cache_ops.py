"""In-place KV cache steering operations.

Key insight: Instruction tokens are processed during prefill and their KV vectors
never change during decoding. We snapshot only the instruction slices once, then
scale and restore them in place. No entire cache object is ever deep-copied.
"""

from typing import List, Tuple
import torch
from transformers import DynamicCache


# --- Version-tolerant cache accessors ---

def _keys(cache, layer: int) -> torch.Tensor:
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        return cache.key_cache[layer]
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        return cache.layers[layer].keys
    if isinstance(cache, (list, tuple)):
        return cache[layer][0]
    raise ValueError(f"Unsupported cache type: {type(cache)}")


def _values(cache, layer: int) -> torch.Tensor:
    if hasattr(cache, "value_cache") and len(cache.value_cache) > 0:
        return cache.value_cache[layer]
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        return cache.layers[layer].values
    if isinstance(cache, (list, tuple)):
        return cache[layer][1]
    raise ValueError(f"Unsupported cache type: {type(cache)}")


def _set_keys(cache, layer: int, tensor: torch.Tensor) -> None:
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer] = tensor
    elif hasattr(cache, "layers"):
        cache.layers[layer].keys = tensor
    elif isinstance(cache, list):
        cache[layer] = (tensor, cache[layer][1])


def _set_values(cache, layer: int, tensor: torch.Tensor) -> None:
    if hasattr(cache, "value_cache"):
        cache.value_cache[layer] = tensor
    elif hasattr(cache, "layers"):
        cache.layers[layer].values = tensor
    elif isinstance(cache, list):
        cache[layer] = (cache[layer][0], tensor)


def num_layers(cache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if isinstance(cache, (list, tuple)):
        return len(cache)
    raise ValueError(f"Unsupported cache type: {type(cache)}")


def _sync_seen(cache, length: int) -> None:
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = length


# --- Probe Cache Management ---

def pop_last_position(cache, keep_len: int):
    """Parks newly appended KV states from the raw pass while running probe passes."""
    saved = []
    for l in range(num_layers(cache)):
        k, v = _keys(cache, l), _values(cache, l)
        saved.append((k[:, :, keep_len:, :].clone(), v[:, :, keep_len:, :].clone()))
        _set_keys(cache, l, k[:, :, :keep_len, :])
        _set_values(cache, l, v[:, :, :keep_len, :])
    _sync_seen(cache, keep_len)
    return saved


def push_position(cache, saved) -> None:
    """Restores parked KV states back to the live cache."""
    for l, (k_new, v_new) in enumerate(saved):
        _set_keys(cache, l, torch.cat([_keys(cache, l), k_new], dim=2))
        _set_values(cache, l, torch.cat([_values(cache, l), v_new], dim=2))
    _sync_seen(cache, _keys(cache, 0).shape[2])


def crop_to(cache, length: int) -> None:
    """Truncates trailing tokens appended during steering probes."""
    if hasattr(cache, "crop"):
        try:
            cache.crop(length)
            return
        except Exception:
            pass
    for l in range(num_layers(cache)):
        _set_keys(cache, l, _keys(cache, l)[:, :, :length, :])
        _set_values(cache, l, _values(cache, l)[:, :, :length, :])
    _sync_seen(cache, length)


# --- In-Place Span Steerer ---

class SpanSteerer:
    """Scales key vectors of the instruction span in-place on selected layers."""

    def __init__(
        self,
        cache,
        token_spans: List[Tuple[int, int]],
        key_scale: float,
        value_scale: float = 1.0,
    ):
        self.cache = cache
        self.key_scale = key_scale
        self.value_scale = value_scale
        self.n_layers = num_layers(cache)

        device = _keys(cache, 0).device
        idx = sorted({i for s, e in token_spans for i in range(s, e)})
        self.index = torch.tensor(idx, device=device, dtype=torch.long)

        # Snapshot the original unscaled slices from prefill
        self._orig_k = [
            _keys(cache, l)[:, :, self.index, :].clone()
            for l in range(self.n_layers)
        ]
        self._orig_v = (
            [
                _values(cache, l)[:, :, self.index, :].clone()
                for l in range(self.n_layers)
            ]
            if value_scale != 1.0
            else None
        )
        self._dirty: List[int] = []

    def apply(self, layers: List[int]) -> None:
        """Applies key scaling to the specified layers in place."""
        self.restore()
        for l in layers:
            _keys(self.cache, l)[:, :, self.index, :] = self._orig_k[l] * self.key_scale
            if self._orig_v is not None:
                _values(self.cache, l)[:, :, self.index, :] = self._orig_v[l] * self.value_scale
        self._dirty = list(layers)

    def restore(self) -> None:
        """Restores scaled layers back to their original unscaled slices."""
        for l in self._dirty:
            _keys(self.cache, l)[:, :, self.index, :] = self._orig_k[l]
            if self._orig_v is not None:
                _values(self.cache, l)[:, :, self.index, :] = self._orig_v[l]
        self._dirty = []