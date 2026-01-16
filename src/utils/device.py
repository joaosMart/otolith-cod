"""
Device Detection and Configuration Utilities.

Handles device selection (MPS, CUDA, CPU) for Apple Silicon and NVIDIA GPUs.
"""

import torch
from typing import Optional


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the best available compute device.

    Priority: preferred > MPS > CUDA > CPU

    Args:
        preferred: Optional preferred device ("mps", "cuda", "cpu")

    Returns:
        torch.device for computation
    """
    if preferred:
        return torch.device(preferred)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device availability and selected device
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(get_device()),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    return info


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    print(f"Selected device: {info['selected_device']}")
    print(f"  MPS available: {info['mps_available']}")
    print(f"  CUDA available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  CUDA device: {info['cuda_device_name']}")
