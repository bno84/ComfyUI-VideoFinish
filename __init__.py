"""Video Finish: video upscaling, resizing, interpolation, and configurable export.

Node IDs are stable once released; saved workflows refer to them.
"""
from . import models
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

models.register()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
