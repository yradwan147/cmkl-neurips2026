"""Load pre-computed multimodal features for CMKL.

Shared by run_cmkl.py, run_nc.py, and run_ablations.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_features(
    features_dir: str = "data/benchmark/features",
) -> dict:
    """Load pre-computed features for CMKL.

    Uses torch.load(mmap=True) for large tensors to avoid loading entirely
    into RAM before needed. Falls back gracefully if files don't exist.

    Args:
        features_dir: Path to directory with .pt feature files.

    Returns:
        Dict with keys: edge_index, edge_type, text_embeddings,
        node_has_text, mol_fingerprints, node_has_mol, mol_input_dim.
    """
    fdir = Path(features_dir)
    features: dict = {
        "edge_index": None,
        "edge_type": None,
        "text_embeddings": None,
        "node_has_text": None,
        "mol_fingerprints": None,
        "node_has_mol": None,
    }
    mol_input_dim = 1024  # default

    # Edge features (critical for R-GCN)
    edge_index_path = fdir / "edge_index.pt"
    edge_type_path = fdir / "edge_type.pt"
    if edge_index_path.exists() and edge_type_path.exists():
        features["edge_index"] = torch.load(edge_index_path, weights_only=True)
        features["edge_type"] = torch.load(edge_type_path, weights_only=True)
        logger.info(f"Loaded edges: {features['edge_index'].shape}")
    else:
        logger.warning("No edge_index.pt/edge_type.pt found — R-GCN will NOT "
                       "do message passing (flat embeddings only)!")

    # Text embeddings (large, use mmap)
    text_path = fdir / "text_embeddings.pt"
    text_mask_path = fdir / "node_has_text.pt"
    if text_path.exists() and text_mask_path.exists():
        features["text_embeddings"] = torch.load(
            text_path, weights_only=True, mmap=True
        )
        features["node_has_text"] = torch.load(
            text_mask_path, weights_only=True
        )
        n_text = features["node_has_text"].sum().item()
        logger.info(f"Loaded text embeddings: {features['text_embeddings'].shape}, "
                     f"{n_text} nodes with text")
    else:
        logger.warning("No text_embeddings.pt found — no textual features")

    # Molecular features
    mol_path = fdir / "mol_features.pt"
    mol_mask_path = fdir / "node_has_mol.pt"
    if mol_path.exists() and mol_mask_path.exists():
        features["mol_fingerprints"] = torch.load(
            mol_path, weights_only=True
        )
        features["node_has_mol"] = torch.load(
            mol_mask_path, weights_only=True
        )
        mol_input_dim = features["mol_fingerprints"].shape[1]
        n_mol = features["node_has_mol"].sum().item()
        logger.info(f"Loaded mol features: {features['mol_fingerprints'].shape}, "
                     f"{n_mol} nodes with mol, dim={mol_input_dim}")
    else:
        logger.warning("No mol_features.pt found — no molecular features")

    # Read mol_dim if available
    mol_dim_path = fdir / "mol_dim.txt"
    if mol_dim_path.exists():
        mol_input_dim = int(mol_dim_path.read_text().strip())

    features["mol_input_dim"] = mol_input_dim

    # Read vocab sizes (saved by precompute_features.py)
    # These represent the full entity/relation counts from ALL tasks,
    # needed when running with a task subset to size the model correctly.
    vocab_path = fdir / "vocab_sizes.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        features["num_entities"] = vocab["num_entities"]
        features["num_relations"] = vocab["num_relations"]
        logger.info(f"Feature vocab: {vocab['num_entities']} entities, "
                     f"{vocab['num_relations']} relations")

    return features
