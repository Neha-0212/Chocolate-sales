"""
Config loader — reads config/config.yaml and returns a dict.
"""

from pathlib import Path
import yaml
from typing import Any


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to the config YAML file

    Returns:
        Dictionary with all config values

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_path(config: dict, key: str) -> Path:
    """
    Get a path from config and return it as a Path object.

    Args:
        config: Loaded config dictionary
        key: Key under 'paths' section

    Returns:
        Path object
    """
    return Path(config["paths"][key])
