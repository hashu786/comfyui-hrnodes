"""Load an IC-LoRA safetensors and turn it into per-Linear LoRADelta objects.

Key mapping is derived from the LoRA file's OWN keys (IC-LoRAs use the native
ComfyUI format ``diffusion_model.<path>.lora_A/B.weight``), NOT from the live
model's state_dict. That matters because once we object-patch the Linears the
model's keys shift to ``...to_q.orig.weight``; deriving from the file keeps
loading correct across re-runs. We still hand the file to ``comfy.lora.load_lora``
so we inherit its adapter objects (with the ``.h`` bypass) and dtype handling.

All comfy imports are lazy.
"""
from __future__ import annotations

import logging

from .masked_lora import LoRADelta

logger = logging.getLogger("HRNodes.MultiICLoRA")

# Suffixes that mark a LoRA tensor; stripping one yields the module base path.
_LORA_SUFFIXES = (
    ".lora_A.weight", ".lora_B.weight",
    ".lora_up.weight", ".lora_down.weight",
    ".lora.up.weight", ".lora.down.weight",
    ".lora_A", ".lora_B",
)


def read_downscale_factor(lora_path):
    """reference_downscale_factor from safetensors metadata (default 1.0)."""
    try:
        import comfy.utils
        _, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        return float((metadata or {}).get("reference_downscale_factor", 1.0))
    except Exception as e:  # noqa: BLE001
        logger.warning("MultiICLoRA: could not read downscale factor (%s); using 1.0", e)
        return 1.0


def _as_adapter(patch):
    """Return a WeightAdapterBase from a comfy patch entry, or None."""
    cands = patch if isinstance(patch, (tuple, list)) else [patch]
    for c in cands:
        if callable(getattr(c, "h", None)) and hasattr(c, "weights"):
            return c
    return None


def _build_to_load(lora_sd):
    """Map each module base in the LoRA file to its model weight key.

    e.g. 'diffusion_model.transformer_blocks.0.attn1.to_q' ->
         'diffusion_model.transformer_blocks.0.attn1.to_q.weight'
    """
    bases = set()
    for k in lora_sd.keys():
        for suf in _LORA_SUFFIXES:
            if k.endswith(suf):
                bases.add(k[: -len(suf)])
                break
    return {b: b + ".weight" for b in bases}


def load_iclora_deltas(model, lora_name, strength, guide_index):
    """Return ([(module_path, LoRADelta), ...], downscale_factor)."""
    import folder_paths
    import comfy.utils
    import comfy.lora

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    downscale = read_downscale_factor(lora_path)

    to_load = _build_to_load(lora_sd)
    patches = comfy.lora.load_lora(lora_sd, to_load, log_missing=False)

    deltas = []
    skipped = 0
    for model_key, patch in patches.items():
        if not isinstance(model_key, str) or not model_key.endswith(".weight"):
            skipped += 1
            continue
        adapter = _as_adapter(patch)
        if adapter is None:
            skipped += 1
            continue
        try:
            adapter.multiplier = float(strength)
        except Exception:  # noqa: BLE001
            pass
        module_path = model_key[: -len(".weight")]
        deltas.append(
            (module_path, LoRADelta(adapter=adapter, guide_index=guide_index,
                                    name=lora_name))
        )

    logger.info(
        "MultiICLoRA: '%s' -> %d Linear deltas (guide %d, downscale %.3g)%s",
        lora_name, len(deltas), guide_index, downscale,
        (", %d skipped" % skipped) if skipped else "",
    )
    if not deltas:
        logger.warning(
            "MultiICLoRA: '%s' produced 0 deltas — no recognizable LoRA keys in file.",
            lora_name,
        )
    return deltas, downscale
