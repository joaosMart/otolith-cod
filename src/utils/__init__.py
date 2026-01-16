"""
Utilities module for otolith age prediction.

Provides device detection and configuration loading.
"""

from .device import get_device, get_device_info, print_device_info
from .config import load_config, get_model_config, get_output_paths

__all__ = [
    "get_device",
    "get_device_info",
    "print_device_info",
    "load_config",
    "get_model_config",
    "get_output_paths",
]
