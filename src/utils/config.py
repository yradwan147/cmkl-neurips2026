"""Experiment configuration management.

Loads YAML config files and provides a unified configuration interface.
Supports config merging (base + experiment-specific) and CLI overrides.

Usage:
    from src.utils.config import load_config
    config = load_config('configs/base.yaml', overrides={'training.lr': 0.0005})
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Dict of dot-separated keys to override values.
            e.g., {'training.lr': 0.0005, 'model.embedding_dim': 128}.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    logger.info(f"Loaded config from {config_path}")
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dicts (override takes precedence).

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration dict.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def save_config(config: dict, output_path: str) -> None:
    """Save configuration to YAML file (for experiment reproducibility).

    Args:
        config: Configuration dictionary.
        output_path: Path to save YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {output_path}")


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key.

    Args:
        d: Dictionary to modify in place.
        key: Dot-separated key path (e.g., 'training.lr').
        value: Value to set.
    """
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
