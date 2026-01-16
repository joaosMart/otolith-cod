"""
Configuration Loading Utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        config: Full configuration dictionary
        model_name: Model identifier

    Returns:
        Model-specific configuration
    """
    return config["models"].get(model_name, {})


def get_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get output directory paths from config.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary of output paths as Path objects
    """
    output_config = config.get("output", {})
    return {
        "embeddings": Path(output_config.get("embeddings_dir", "outputs/embeddings")),
        "models": Path(output_config.get("models_dir", "outputs/models")),
        "results": Path(output_config.get("results_dir", "outputs/results")),
        "figures": Path(output_config.get("figures_dir", "outputs/figures")),
    }
