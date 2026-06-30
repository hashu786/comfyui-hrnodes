"""Offline tests for masked multi-IC-LoRA token math (no torch / ComfyUI needed).

Run:  python tests/test_masked_lora_logic.py
Validates the layout assumption that the whole approach rests on: guides live
contiguously at the END of the sequence, partitioned in order, and each LoRA's
per-token mask isolates its own guide while sharing the noisy region.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrnodes.multi_iclora import mask_logic  # noqa: E402


def test_guide_ranges_partition():
    # T=10, 4 guide tokens => guide_start=6; two guides of 1 and 3 tokens.
    r = mask_logic.guide_ranges(total_tokens=10, num_guide_tokens=4,
                                surviving_counts=[1, 3])
    assert r == [(6, 7), (7, 10)], r


def test_guide_ranges_mismatch_raises():
    try:
        mask_logic.guide_ranges(10, 4, [1, 1])  # sums to 2, not 4
    except ValueError:
        return
    raise AssertionError("expected ValueError on count mismatch")


def test_token_weights_isolation():
    # guide_start=6, lora owns guide 1 span (7,10), blend 0.5
    ranges = mask_logic.guide_ranges(10, 4, [1, 3])
    w = mask_logic.token_weights(10, guide_start=6, owned_range=ranges[1],
                                 noisy_blend=0.5)
    assert w[0:6] == [0.5] * 6          # shared noisy region
    assert w[6] == 0.0                  # other guide's token -> excluded
    assert w[7:10] == [1.0, 1.0, 1.0]   # own guide tokens -> full


def test_resolve_owned_ranges():
    owned = mask_logic.resolve_owned_ranges(
        10, 4, [1, 3], guide_indices=[0, 1, 5])
    assert owned[0] == (6, 7)
    assert owned[1] == (7, 10)
    assert owned[2] is None             # out-of-range guide index


def test_masked_combine_separates_loras():
    """Numpy emulation of MaskedLoRALinear: prove guide-A tokens see only LoRA A,
    guide-B tokens only LoRA B, noisy tokens a blend of both."""
    rng = np.random.default_rng(0)
    T, C, R = 10, 8, 4
    num_guide, surviving = 4, [1, 3]
    guide_start = T - num_guide
    x = rng.standard_normal((1, T, C)).astype(np.float32)

    # Two LoRAs (up:(C,R), down:(R,C)); A owns guide 0, B owns guide 1.
    def lora():
        return (rng.standard_normal((C, R)).astype(np.float32),
                rng.standard_normal((R, C)).astype(np.float32))
    upA, downA = lora()
    upB, downB = lora()

    def delta(up, down):  # (1,T,C)
        return (x @ down.T) @ up.T

    dA, dB = delta(upA, downA), delta(upB, downB)

    blend = 0.6
    ranges = mask_logic.guide_ranges(T, num_guide, surviving)
    wA = np.array(mask_logic.token_weights(T, guide_start, ranges[0], blend)).reshape(1, T, 1)
    wB = np.array(mask_logic.token_weights(T, guide_start, ranges[1], blend)).reshape(1, T, 1)

    out = wA * dA + wB * dB

    # guide 0 token (idx 6): only A
    assert np.allclose(out[0, 6], dA[0, 6]) and not np.allclose(out[0, 6], dB[0, 6])
    # guide 1 tokens (7..9): only B
    for t in (7, 8, 9):
        assert np.allclose(out[0, t], dB[0, t])
        assert not np.allclose(out[0, t], dA[0, t])
    # noisy token (idx 0): blended sum of both
    assert np.allclose(out[0, 0], blend * (dA[0, 0] + dB[0, 0]))


def test_lora_delta_dataclass_importable_without_torch():
    from hrnodes.multi_iclora.masked_lora import LoRADelta
    d = LoRADelta(adapter=None, guide_index=1, name="x")
    assert d.guide_index == 1 and d.name == "x"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
