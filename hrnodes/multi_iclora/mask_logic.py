"""Framework-agnostic token-layout math for masked multi-IC-LoRA.

Kept pure (numpy / plain python) so it can be unit-tested without torch or
ComfyUI. The torch wrapper in ``masked_lora.py`` consumes these results.

LTX-2 token layout (verified against ComfyUI core
``comfy/ldm/lightricks/model.py::LTXVModel._process_input``):

    sequence = [ noisy / generation tokens | guide_0 | guide_1 | ... ]

  * Guide tokens are appended at the END of the sequence, contiguously, in the
    order the IC-LoRA guide nodes were chained.
  * ``guide_start = total_tokens - num_guide_tokens``.
  * Each guide's *surviving* (post grid-mask) token count comes from
    ``resolved_guide_entries[i]["surviving_count"]``; these partition the guide
    region in order.

This module turns that into, for one LoRA that "owns" guide index ``g``, a
per-token weight vector ``w`` of length ``T``:

    w[t] = noisy_blend          for t in [0, guide_start)      # shared path
    w[t] = 1.0                  for t in own guide range
    w[t] = 0.0                  for t in any other guide range

The LoRA's low-rank delta is then added as ``w[:, None] * delta``. On the shared
noisy tokens both LoRAs contribute (scaled by ``noisy_blend``); on a guide's own
tokens only its LoRA acts, so the two adapters never cross-contaminate each
other's reference encoding.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def guide_ranges(
    total_tokens: int,
    num_guide_tokens: int,
    surviving_counts: Sequence[int],
) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] token spans for each guide, in order.

    ``surviving_counts`` are the per-guide post-grid-mask token counts
    (``resolved_guide_entries[i]['surviving_count']``). The spans are contiguous
    and begin at ``guide_start = total_tokens - num_guide_tokens``.

    Raises ValueError if the counts don't sum to ``num_guide_tokens`` (a strong
    signal the layout assumption broke and we should NOT silently mask wrong
    tokens).
    """
    guide_start = total_tokens - num_guide_tokens
    if guide_start < 0:
        raise ValueError(
            f"num_guide_tokens ({num_guide_tokens}) > total_tokens ({total_tokens})"
        )
    if sum(surviving_counts) != num_guide_tokens:
        raise ValueError(
            f"surviving_counts sum ({sum(surviving_counts)}) != "
            f"num_guide_tokens ({num_guide_tokens})"
        )
    ranges: List[Tuple[int, int]] = []
    cursor = guide_start
    for c in surviving_counts:
        ranges.append((cursor, cursor + c))
        cursor += c
    return ranges


def token_weights(
    total_tokens: int,
    guide_start: int,
    owned_range: Optional[Tuple[int, int]],
    noisy_blend: float,
) -> List[float]:
    """Per-token scalar weights for one LoRA (length ``total_tokens``).

    ``owned_range`` is this LoRA's guide span, or ``None`` to apply the LoRA
    only on the shared noisy tokens (e.g. a guide that didn't survive grid-mask,
    or a plain style LoRA with no guide).
    """
    w = [0.0] * total_tokens
    for t in range(0, max(0, guide_start)):
        w[t] = noisy_blend
    if owned_range is not None:
        s, e = owned_range
        for t in range(s, e):
            w[t] = 1.0
    return w


def resolve_owned_ranges(
    total_tokens: int,
    num_guide_tokens: int,
    surviving_counts: Sequence[int],
    guide_indices: Sequence[int],
) -> List[Optional[Tuple[int, int]]]:
    """Map each LoRA's ``guide_index`` to its token span.

    Returns a list parallel to ``guide_indices``; an entry is ``None`` if that
    guide index is out of range or its guide produced zero surviving tokens.
    """
    ranges = guide_ranges(total_tokens, num_guide_tokens, surviving_counts)
    out: List[Optional[Tuple[int, int]]] = []
    for gi in guide_indices:
        if 0 <= gi < len(ranges):
            s, e = ranges[gi]
            out.append((s, e) if e > s else None)
        else:
            out.append(None)
    return out
