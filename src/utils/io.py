"""Data I/O utilities for loading and saving experiment artifacts.

Handles reading/writing of various formats: CSV, JSON, PyTorch tensors,
NumPy arrays, and pickle files.

Usage:
    from src.utils.io import save_json, load_json, save_tensor, load_tensor
    save_json(data, 'results/metrics.json')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation level.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Any:
    """Load data from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Deserialized data.
    """
    with open(path) as f:
        data = json.load(f)
    return data


def save_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save PyTorch tensor to disk.

    Args:
        tensor: Tensor to save.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, p)
    logger.info(f"Saved tensor {tensor.shape} to {path}")


def load_tensor(path: str, device: str = "cpu") -> torch.Tensor:
    """Load PyTorch tensor from disk.

    Args:
        path: Path to saved tensor.
        device: Device to load tensor onto.

    Returns:
        Loaded tensor.
    """
    tensor = torch.load(path, map_location=device, weights_only=True)
    logger.info(f"Loaded tensor {tensor.shape} from {path}")
    return tensor


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
