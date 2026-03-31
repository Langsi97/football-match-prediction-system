"""
Configuration utilities.

This module loads YAML configuration files used across the project.
"""

import yaml
from pathlib import Path

# Root project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Config folder
CONFIG_DIR = BASE_DIR / "config"


def load_yaml(file_path: Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    file_path : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)