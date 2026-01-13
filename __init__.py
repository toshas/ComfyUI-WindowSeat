"""
ComfyUI WindowSeat Reflection Removal

A ComfyUI custom node plugin for removing reflections from images using
the WindowSeat model.

Nodes:
- WindowSeat Model Loader: Load the reflection removal model
- WindowSeat Reflection Removal: Process images to remove reflections
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
