"""Node classification dataset construction for continual learning.

PrimeKG has 10 node types (anatomy, biological_process, cellular_component,
disease, drug, effect/phenotype, exposure, gene/protein, molecular_function,
pathway). This module builds node classification datasets aligned with
the CL task sequence by identifying emerged entities per task and using
their node types as labels.

Usage:
    from src.data.node_classification import build_nc_dataset, get_label_map
    label_map = get_label_map()
    nc_tasks = build_nc_dataset(task_sequence, entity_to_id, node_types)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PrimeKG 10 node types -> integer labels
NODE_TYPE_LABELS: dict[str, int] = {
    "anatomy": 0,
    "biological_process": 1,
    "cellular_component": 2,
    "disease": 3,
    "drug": 4,
    "effect/phenotype": 5,
    "exposure": 6,
    "gene/protein": 7,
    "molecular_function": 8,
    "pathway": 9,
}


def get_label_map() -> dict[str, int]:
    """Return the node type -> integer label mapping.

    Returns:
        Dict mapping node type strings to integer labels (0-9).
    """
    return dict(NODE_TYPE_LABELS)


def load_node_types(
    node_index_path: str | Path | None = None,
    kg_csv_path: str | Path | None = None,
) -> dict[str, str]:
    """Load entity -> node_type mapping.

    Tries node_index_map.csv first (has node_id, node_type columns),
    then falls back to extracting from the KG CSV (x_type, y_type).

    Args:
        node_index_path: Path to node_index_map.csv.
        kg_csv_path: Path to KG CSV file (fallback).

    Returns:
        Dict mapping entity_id (str) -> node_type (str).
    """
    import pandas as pd

    node_types = {}

    if node_index_path and Path(node_index_path).exists():
        df = pd.read_csv(node_index_path)
        for _, row in df.iterrows():
            node_types[str(row["node_id"])] = str(row["node_type"])
        logger.info(f"Loaded {len(node_types)} node types from {node_index_path}")
        return node_types

    if kg_csv_path and Path(kg_csv_path).exists():
        df = pd.read_csv(kg_csv_path)
        # Extract from x_type/y_type columns
        for col_id, col_type in [("x_id", "x_type"), ("y_id", "y_type")]:
            if col_id in df.columns and col_type in df.columns:
                for _, row in df[[col_id, col_type]].drop_duplicates().iterrows():
                    node_types[str(row[col_id])] = str(row[col_type])
        # Also try x_name/y_name with x_type/y_type
        for col_name, col_type in [("x_name", "x_type"), ("y_name", "y_type")]:
            if col_name in df.columns and col_type in df.columns:
                for _, row in df[[col_name, col_type]].drop_duplicates().iterrows():
                    node_types[str(row[col_name])] = str(row[col_type])
        logger.info(f"Extracted {len(node_types)} node types from KG CSV")
        return node_types

    logger.warning("No node type source found")
    return node_types


def build_nc_dataset(
    task_sequence: OrderedDict[str, dict[str, np.ndarray]],
    entity_to_id: dict[str, int],
    node_types: dict[str, str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> OrderedDict[str, dict]:
    """Build node classification datasets aligned with CL tasks.

    For each CL task, identifies all entities that appear in that task's
    triples, maps them to node type labels, and creates train/val/test splits.

    Args:
        task_sequence: CL task sequence from load_task_sequence().
        entity_to_id: Global entity -> int mapping.
        node_types: Entity -> node_type string mapping.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for splits.

    Returns:
        OrderedDict mapping task_name -> {
            'node_ids': int array of entity IDs,
            'labels': int array of node type labels,
            'train_mask': bool array,
            'val_mask': bool array,
            'test_mask': bool array,
            'label_names': list of label name strings,
        }
    """
    rng = np.random.RandomState(seed)
    label_map = get_label_map()
    nc_tasks: OrderedDict[str, dict] = OrderedDict()

    # Reverse mapping: int ID -> string entity name
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    for task_name, task_data in task_sequence.items():
        # Collect all entity int IDs in this task
        entity_ids: set[int] = set()
        for split_data in task_data.values():
            if len(split_data) > 0:
                entity_ids.update(split_data[:, 0].tolist())  # heads
                entity_ids.update(split_data[:, 2].tolist())  # tails

        # Map int IDs to labels via string entity names
        node_ids = []
        labels = []
        for eid in sorted(entity_ids):
            entity_name = id_to_entity.get(eid)
            if entity_name is None:
                continue
            nt = node_types.get(entity_name)
            if nt is None:
                # Benchmark uses zero-padded IDs, CSV uses raw — try stripped
                stripped = entity_name.lstrip("0") or "0"
                nt = node_types.get(stripped)
            if nt is None or nt not in label_map:
                continue
            node_ids.append(eid)
            labels.append(label_map[nt])

        if len(node_ids) < 10:
            logger.warning(f"Task {task_name}: only {len(node_ids)} labeled nodes, skipping")
            continue

        node_ids = np.array(node_ids, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)

        # Create train/val/test masks
        n = len(node_ids)
        perm = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

        nc_tasks[task_name] = {
            "node_ids": node_ids,
            "labels": labels,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "label_names": [k for k, v in sorted(label_map.items(), key=lambda x: x[1])],
            "num_classes": len(label_map),
        }

        unique_labels = np.unique(labels)
        logger.info(f"Task {task_name}: {n} nodes, {len(unique_labels)} classes, "
                     f"train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return nc_tasks
