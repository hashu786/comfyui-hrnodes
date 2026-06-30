"""Torch runtime for masked multi-IC-LoRA.

We apply LoRA deltas via forward HOOKS on the target Linears (not by replacing
the modules). Replacing modules would move the real weight from
``...to_q.weight`` to ``...to_q.orig.weight``, which breaks ComfyUI's
weight-patch system (e.g. the distilled-model LoRA) and weight casting — that
produced blurry output. A forward hook leaves the module and its ``.weight``
path intact and just adds our masked delta to the output.

torch is imported lazily so this module (and ``mask_logic``) import without it.
"""
from __future__ import annotations

import contextvars
import logging

from . import mask_logic

logger = logging.getLogger("HRNodes.MultiICLoRA")

# Per-forward token layout, set by the diffusion wrapper before blocks run.
_LAYOUT = contextvars.ContextVar("sse_iclora_layout", default=None)

# One-shot diagnostics + a small per-(layout,guide) weight-tensor cache.
_DIAG = {"layout": False, "guides": set()}
_WCACHE = {}


def reset_diag():
    _DIAG["layout"] = False
    _DIAG["guides"] = set()
    _WCACHE.clear()


def set_layout(total_tokens, num_guide_tokens, resolved_entries):
    """Called from the diffusion wrapper. Returns a token for reset()."""
    surviving = None
    if resolved_entries:
        surviving = [int(e.get("surviving_count", e.get("pre_filter_count", 0)))
                     for e in resolved_entries]
    layout = {
        "total_tokens": int(total_tokens),
        "guide_start": int(total_tokens) - int(num_guide_tokens),
        "num_guide_tokens": int(num_guide_tokens),
        "surviving_counts": surviving,
    }
    if not _DIAG["layout"]:
        logger.info(
            "MultiICLoRA[diag]: total=%d guide_start=%d num_guide=%d surviving_counts=%s",
            layout["total_tokens"], layout["guide_start"],
            layout["num_guide_tokens"], surviving,
        )
        if surviving is None:
            logger.warning(
                "MultiICLoRA[diag]: NO resolved_guide_entries — guide tokens cannot "
                "be routed per-LoRA; LoRAs act only on noisy tokens."
            )
        _DIAG["layout"] = True
    return _LAYOUT.set(layout)


def reset_layout(token):
    try:
        _LAYOUT.reset(token)
    except Exception:
        pass


def get_layout():
    return _LAYOUT.get()


class LoRADelta:
    """A ComfyUI weight-adapter bound to one Linear, plus its guide routing.

    ``adapter.h(x, base)`` returns up(down(x)) * (alpha/rank) * multiplier;
    per-LoRA strength is folded into ``adapter.multiplier`` at load time.
    """

    __slots__ = ("adapter", "guide_index", "name")

    def __init__(self, adapter, guide_index, name=""):
        self.adapter = adapter
        self.guide_index = int(guide_index)
        self.name = name


def _build_weight_tensor(torch, layout, guide_index, noisy_blend, T, device, dtype):
    """(1, T, 1) per-token weight for one LoRA given the current layout."""
    if layout is None or layout.get("total_tokens") != T:
        return torch.full((1, T, 1), float(noisy_blend), device=device, dtype=dtype)

    guide_start = layout["guide_start"]
    surviving = layout.get("surviving_counts")
    owned = None
    if surviving is not None:
        try:
            owned = mask_logic.resolve_owned_ranges(
                T, layout["num_guide_tokens"], surviving, [guide_index]
            )[0]
        except ValueError as e:
            logger.warning("MultiICLoRA: layout mismatch (%s); noisy-only.", e)
            owned = None

    wlist = mask_logic.token_weights(T, guide_start, owned, noisy_blend)
    if guide_index not in _DIAG["guides"]:
        n_full = sum(1 for v in wlist if v == 1.0)
        logger.info(
            "MultiICLoRA[diag]: guide_index=%d -> owned_range=%s, %d full-weight "
            "tokens, %d noisy tokens @ %.2f",
            guide_index, owned, n_full, max(0, guide_start), noisy_blend,
        )
        _DIAG["guides"].add(guide_index)
    return torch.tensor(wlist, device=device, dtype=dtype).view(1, T, 1)


def _weight_for(torch, guide_index, noisy_blend, T, device, dtype):
    layout = get_layout()
    if layout is None:
        key = ("none", T, guide_index, noisy_blend, str(device), str(dtype))
    else:
        sc = layout["surviving_counts"]
        key = (layout["total_tokens"], layout["guide_start"],
               tuple(sc) if sc else None, guide_index, noisy_blend,
               T, str(device), str(dtype))
    w = _WCACHE.get(key)
    if w is None:
        w = _build_weight_tensor(torch, layout, guide_index, noisy_blend, T, device, dtype)
        _WCACHE[key] = w
    return w


def _ensure_adapter_device(torch, adapter, device):
    """Move a weight-adapter's tensors to ``device`` if needed."""
    try:
        w = getattr(adapter, "weights", None)
        if not w:
            return
        ref = w[1] if len(w) > 1 and torch.is_tensor(w[1]) else (
            w[0] if torch.is_tensor(w[0]) else None)
        if ref is None or ref.device == device:
            return
        adapter.weights = type(w)(
            t.to(device) if torch.is_tensor(t) else t for t in w
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("MultiICLoRA: could not move adapter to %s (%s)", device, e)


def make_forward_hook(deltas, noisy_blend):
    """Build a forward hook that adds masked LoRA deltas to a Linear's output."""
    import torch

    nb = float(noisy_blend)

    def hook(module, args, output):
        try:
            if not args or not torch.is_tensor(output):
                return output
            x = args[0]
            if not torch.is_tensor(x) or x.dim() != 3:
                return output
            T = x.shape[1]
            out = output
            for d in deltas:
                w = _weight_for(torch, d.guide_index, nb, T, output.device, output.dtype)
                _ensure_adapter_device(torch, d.adapter, x.device)
                delta = d.adapter.h(x, output)
                out = out + w * delta.to(output.dtype)
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("MultiICLoRA: hook error (%s); skipping delta.", e)
            return output

    return hook
