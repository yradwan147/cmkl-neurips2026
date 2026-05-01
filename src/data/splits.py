"""Train/validation/test split creation for continual learning tasks.

Creates non-leaking splits where test triples from task i never appear
in training sets of tasks i+1, i+2, etc. This is critical for valid
continual learning evaluation.

Usage:
    from src.data.splits import create_splits_per_task
    splits = create_splits_per_task(tasks, val_ratio=0.1, test_ratio=0.2)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_splits_per_task(
    tasks: OrderedDict[str, pd.DataFrame],
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> OrderedDict[str, dict]:
    """Create train/val/test splits for each task in the sequence.

    For continual learning evaluation:
    - Training set: used for learning the current task
    - Validation set: used for hyperparameter tuning and early stopping
    - Test set: used for evaluating on current AND all previous tasks

    Critical: test triples from earlier tasks are excluded from later
    training sets to prevent leakage.

    Args:
        tasks: OrderedDict mapping task names to DataFrames of triples.
        val_ratio: Fraction of triples for validation.
        test_ratio: Fraction of triples for testing.
        seed: Random seed for reproducibility.

    Returns:
        OrderedDict mapping task_name -> {
            'train': DataFrame, 'val': DataFrame, 'test': DataFrame,
            'n_train': int, 'n_val': int, 'n_test': int
        }.
    """
    rng = np.random.RandomState(seed)
    splits = OrderedDict()

    # Collect all test triple IDs from previous tasks to exclude from future training
    all_test_ids: set[str] = set()
    all_val_ids: set[str] = set()

    for task_name, df in tasks.items():
        df = df.copy()
        n = len(df)

        if n == 0:
            logger.warning(f"Task {task_name} has 0 triples, skipping")
            continue

        # Create triple IDs
        df["_triple_id"] = (
            df["x_id"].astype(str) + "|" + df["relation"] + "|" + df["y_id"].astype(str)
        )

        # Remove any triples that appeared in previous tasks' test/val sets
        # (prevents leakage for the base task which is large)
        if all_test_ids or all_val_ids:
            exclude = all_test_ids | all_val_ids
            before = len(df)
            df = df[~df["_triple_id"].isin(exclude)].reset_index(drop=True)
            if len(df) < before:
                logger.info(
                    f"  {task_name}: removed {before - len(df)} triples that overlap "
                    f"with previous test/val sets"
                )
            n = len(df)

        if n == 0:
            logger.warning(f"Task {task_name} has 0 triples after dedup, skipping")
            continue

        # Shuffle
        perm = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Very small task: put everything in train, skip val/test
            n_train = n
            n_val = 0
            n_test = 0

        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        train_df = df.iloc[train_idx].drop(columns=["_triple_id"]).reset_index(drop=True)
        val_df = df.iloc[val_idx].drop(columns=["_triple_id"]).reset_index(drop=True)
        test_df = df.iloc[test_idx].drop(columns=["_triple_id"]).reset_index(drop=True)

        splits[task_name] = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }

        # Track test/val IDs for leakage prevention
        test_ids = set(
            test_df["x_id"].astype(str) + "|" + test_df["relation"] + "|" + test_df["y_id"].astype(str)
        )
        val_ids = set(
            val_df["x_id"].astype(str) + "|" + val_df["relation"] + "|" + val_df["y_id"].astype(str)
        )
        all_test_ids.update(test_ids)
        all_val_ids.update(val_ids)

        logger.info(
            f"  {task_name}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
        )

    return splits


def verify_no_leakage(
    splits: OrderedDict[str, dict],
) -> bool:
    """Verify no data leakage across task splits.

    Ensures test triples from task i do not appear in training
    sets of tasks i+1, i+2, etc.

    Args:
        splits: Split dictionary from create_splits_per_task.

    Returns:
        True if no leakage detected.

    Raises:
        ValueError: If leakage is detected, with details.
    """
    task_names = list(splits.keys())
    all_prior_test_ids: set[str] = set()

    for i, name in enumerate(task_names):
        s = splits[name]
        train_ids = set(
            s["train"]["x_id"].astype(str) + "|" + s["train"]["relation"] + "|" + s["train"]["y_id"].astype(str)
        )

        # Check against all prior test sets
        leak = train_ids & all_prior_test_ids
        if leak:
            raise ValueError(
                f"LEAKAGE DETECTED: {len(leak)} test triples from prior tasks "
                f"appear in training set of {name}. Examples: {list(leak)[:3]}"
            )

        # Add this task's test IDs to the set
        test_ids = set(
            s["test"]["x_id"].astype(str) + "|" + s["test"]["relation"] + "|" + s["test"]["y_id"].astype(str)
        )
        all_prior_test_ids.update(test_ids)

    logger.info(f"No leakage detected across {len(splits)} tasks")
    return True


def save_splits(
    splits: OrderedDict[str, dict],
    output_dir: str,
) -> None:
    """Save splits to disk in standard KGE format.

    Each task gets a directory with train.txt, valid.txt, test.txt.
    Each line: head_entity<tab>relation<tab>tail_entity.

    Args:
        splits: Split dictionary from create_splits_per_task.
        output_dir: Base directory for saving (e.g., data/benchmark/tasks/).
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    for task_name, s in splits.items():
        task_dir = base / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for split_name, key in [("train", "train"), ("valid", "val"), ("test", "test")]:
            df = s[key]
            if len(df) == 0:
                # Write empty file
                (task_dir / f"{split_name}.txt").touch()
                continue

            # Format: head_id\trelation\ttail_id
            lines = (
                df["x_id"].astype(str) + "\t" + df["relation"] + "\t" + df["y_id"].astype(str)
            )
            with open(task_dir / f"{split_name}.txt", "w") as f:
                f.write("\n".join(lines))

        logger.info(f"Saved splits for {task_name} to {task_dir}")

    logger.info(f"All splits saved to {output_dir}")
