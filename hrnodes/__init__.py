"""HR Nodes — a small ComfyUI node pack. Register new nodes in NODE_CLASS_MAPPINGS."""
from .multi_iclora.node import LTXMultiICLoRA

NODE_CLASS_MAPPINGS = {
    "LTXMultiICLoRA": LTXMultiICLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXMultiICLoRA": getattr(LTXMultiICLoRA, "DISPLAY_NAME", "LTXMultiICLoRA"),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
