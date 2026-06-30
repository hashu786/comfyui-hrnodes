"""LTX Multi IC-LoRA — apply two IC-LoRAs at once, each routed to its own guide.

Replaces a pair of stock ``LTX IC-LoRA Loader (Model Only)`` nodes. Instead of
merging both LoRAs globally (which averages them), each LoRA's low-rank delta is
applied in activation space and masked to its guide's token span, so e.g. a
Union/pose LoRA and an Ingredients/image-reference LoRA no longer fight over the
same weights.

Wiring:
  * Chain your two ``LTX Add Video IC-LoRA Guide`` nodes as usual. The ORDER you
    chain them defines guide indices: first = guide 0, second = guide 1.
  * Feed each guide its ``latent_downscale_factor`` from this node's
    ``downscale_a`` / ``downscale_b`` outputs (read from each LoRA's metadata).
  * Do NOT also load these same LoRAs with the stock IC-LoRA loader.

``noisy_blend`` scales how strongly BOTH LoRAs act on the shared generation
tokens. 1.0 = full strength for each (recommended start); lower it only if the
two LoRAs visibly fight, knowing it also weakens conditioning.
"""
import logging

logger = logging.getLogger("HRNodes.MultiICLoRA")


def _lora_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


class LTXMultiICLoRA:
    DISPLAY_NAME = "LTX Multi IC-LoRA"
    CATEGORY = "HRNodes/IC-LoRA"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "FLOAT", "FLOAT")
    RETURN_NAMES = ("model", "downscale_a", "downscale_b")
    DESCRIPTION = (
        "Apply two IC-LoRAs simultaneously, each masked to its own guide's "
        "tokens so they don't average together. Guide index = order the IC-LoRA "
        "guide nodes are chained (first=0, second=1)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        loras = _lora_list()
        return {
            "required": {
                "model": ("MODEL",),
                "lora_a": (loras,),
                "strength_a": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "guide_index_a": ("INT", {"default": 0, "min": 0, "max": 7}),
                "lora_b": (loras,),
                "strength_b": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "guide_index_b": ("INT", {"default": 1, "min": 0, "max": 7}),
                "noisy_blend": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "How strongly BOTH LoRAs act on the shared generation "
                                "tokens. 1.0 = full; lower = less fighting but weaker control."},
                ),
            },
            "optional": {
                "enable_a": ("BOOLEAN", {"default": True}),
                "enable_b": ("BOOLEAN", {"default": True}),
            },
        }

    def apply(self, model, lora_a, strength_a, guide_index_a,
              lora_b, strength_b, guide_index_b, noisy_blend,
              enable_a=True, enable_b=True):
        from .lora_io import load_iclora_deltas
        from .patcher import apply_masked_iclora

        all_deltas = []
        downscale_a = downscale_b = 1.0

        if enable_a and strength_a != 0.0:
            da, downscale_a = load_iclora_deltas(model, lora_a, strength_a, guide_index_a)
            all_deltas.extend(da)

        if enable_b and strength_b != 0.0:
            db, downscale_b = load_iclora_deltas(model, lora_b, strength_b, guide_index_b)
            all_deltas.extend(db)

        if not all_deltas:
            logger.warning("MultiICLoRA: nothing to apply; returning model unchanged.")
            return (model, downscale_a, downscale_b)

        patched_model, n = apply_masked_iclora(model, all_deltas, noisy_blend)
        if n == 0:
            logger.warning("MultiICLoRA: 0 modules hooked; returning model unchanged.")
            return (model, downscale_a, downscale_b)

        return (patched_model, downscale_a, downscale_b)
