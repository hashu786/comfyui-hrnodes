"""comfyui-hrnodes — ComfyUI custom node pack.

Exposes NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS for ComfyUI to load.
"""
from .hrnodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
