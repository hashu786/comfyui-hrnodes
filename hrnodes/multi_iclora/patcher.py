"""Apply masked multi-IC-LoRA to a ComfyUI ModelPatcher via forward hooks.

We do NOT replace modules (object_patch). Doing so moves the real weight to
``...orig.weight`` and breaks ComfyUI's weight-patch system (distilled-model
LoRA) + weight casting, which caused blurry output. Instead, for the duration
of each diffusion forward we:

  * monkeypatch ``diffusion_model._process_input`` to capture the guide token
    layout (published via ``set_layout``), and
  * register a forward hook on each target Linear that adds our masked LoRA
    delta to its output.

Both are installed at forward entry and removed in ``finally`` — so the shared
base model is left untouched between runs.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from . import masked_lora

logger = logging.getLogger("HRNodes.MultiICLoRA")


def _install_wrapper(model, targets, noisy_blend):
    """targets: list of (module_path, [LoRADelta, ...])."""
    import comfy.patcher_extension as pe

    dm = model.get_model_object("diffusion_model")

    def wrapper(executor, *args, **kwargs):
        orig_pi = dm._process_input
        state = {"tok": None, "handles": []}

        def capturing_process_input(x, keyframe_idxs, denoise_mask, **kw):
            out = orig_pi(x, keyframe_idxs, denoise_mask, **kw)
            try:
                xo, _pix, add = out
                x0 = xo[0] if isinstance(xo, (list, tuple)) else xo
                ngt = int(add.get("num_guide_tokens", 0) or 0)
                resolved = add.get("resolved_guide_entries", None)
                state["tok"] = masked_lora.set_layout(x0.shape[1], ngt, resolved)
            except Exception as e:  # noqa: BLE001
                logger.warning("MultiICLoRA: failed to capture layout (%s)", e)
            return out

        # Register forward hooks on the target Linears.
        for module_path, deltas in targets:
            try:
                mod = model.get_model_object(module_path)
                h = mod.register_forward_hook(
                    masked_lora.make_forward_hook(deltas, noisy_blend)
                )
                state["handles"].append(h)
            except Exception as e:  # noqa: BLE001
                logger.warning("MultiICLoRA: could not hook '%s' (%s)", module_path, e)

        dm._process_input = capturing_process_input
        try:
            return executor(*args, **kwargs)
        finally:
            for h in state["handles"]:
                try:
                    h.remove()
                except Exception:
                    pass
            try:
                del dm._process_input
            except Exception:
                pass
            if state["tok"] is not None:
                masked_lora.reset_layout(state["tok"])

    model.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "sse_multi_iclora", wrapper)


def apply_masked_iclora(model, all_deltas, noisy_blend):
    """Clone ``model``, attach masked-LoRA hooks + layout wrapper, return clone."""
    masked_lora.reset_diag()
    m = model.clone()

    by_module = defaultdict(list)
    for module_path, delta in all_deltas:
        by_module[module_path].append(delta)

    targets, missing = [], 0
    for module_path, deltas in by_module.items():
        try:
            m.get_model_object(module_path)  # validate path resolves
        except Exception as e:  # noqa: BLE001
            logger.warning("MultiICLoRA: module '%s' not found (%s)", module_path, e)
            missing += 1
            continue
        targets.append((module_path, deltas))

    if targets:
        _install_wrapper(m, targets, noisy_blend)
    logger.info(
        "MultiICLoRA: hooked %d Linear modules (%d missing), noisy_blend=%.3g",
        len(targets), missing, noisy_blend,
    )
    return m, len(targets)
