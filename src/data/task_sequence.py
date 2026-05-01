"""Define continual learning task sequences from temporal KG diffs.

Groups new triples into discrete learning tasks using various strategies.
The entity_type strategy is recommended as it mirrors real-world knowledge
evolution (new drugs approved vs. new diseases characterized).

Strategies:
- entity_type: Group by entity type (drug, disease, gene/protein)
- relation_type: Group by relation type
- temporal: Use publication dates for finer-grained splits

Usage:
    from src.data.task_sequence import create_task_sequence
    tasks = create_task_sequence(kg_t0, kg_t1, strategy='entity_type')
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import pandas as pd

logger = logging.getLogger(__name__)

# Task 0 is always the base KG (t0). Subsequent tasks contain added triples.
# For entity_type strategy, we group by the "domain" of the new knowledge.
ENTITY_TYPE_GROUPS = OrderedDict([
    ("drug_related", ["drug"]),
    ("disease_related", ["disease"]),
    ("gene_protein", ["gene/protein"]),
    ("phenotype_related", ["effect/phenotype"]),
    ("biological_process", ["biological_process", "molecular_function", "cellular_component"]),
    ("anatomy_pathway", ["anatomy", "pathway", "exposure"]),
])


def create_task_sequence(
    kg_t0: pd.DataFrame,
    kg_t1: pd.DataFrame,
    strategy: str = "entity_type",
    include_base_task: bool = True,
) -> OrderedDict[str, pd.DataFrame]:
    """Create a sequence of tasks from temporal KG diffs.

    Task 0 is always the base KG (t0). Subsequent tasks contain groups
    of added triples organized by the chosen strategy.

    Args:
        kg_t0: Older KG snapshot DataFrame.
        kg_t1: Newer KG snapshot DataFrame.
        strategy: Grouping strategy - 'entity_type', 'relation_type',
            or 'temporal'.
        include_base_task: If True, include t0 as task_0.

    Returns:
        OrderedDict mapping task names to DataFrames of triples.
    """
    # Compute diff to get added triples
    logger.info(f"Computing temporal diff with strategy={strategy}")

    if isinstance(kg_t0, str) and isinstance(kg_t1, str):
        from src.data.temporal_diff import compute_kg_diff
        _, added_df, _, _ = compute_kg_diff(kg_t0, kg_t1)
        kg_t0_df = pd.read_csv(kg_t0, low_memory=False)
    else:
        kg_t0_df = kg_t0
        added_df = _compute_added_from_dfs(kg_t0, kg_t1)

    logger.info(f"Added triples to distribute into tasks: {len(added_df):,}")

    # Build task sequence
    tasks = OrderedDict()

    if include_base_task:
        tasks["task_0_base"] = kg_t0_df

    if strategy == "entity_type":
        new_tasks = _entity_type_strategy(added_df)
    elif strategy == "relation_type":
        new_tasks = _relation_type_strategy(added_df)
    elif strategy == "temporal":
        new_tasks = _temporal_strategy(added_df)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'entity_type', 'relation_type', or 'temporal'.")

    tasks.update(new_tasks)

    # Log task sizes
    for name, df in tasks.items():
        logger.info(f"  {name}: {len(df):,} triples")

    return tasks


def _compute_added_from_dfs(kg_old: pd.DataFrame, kg_new: pd.DataFrame) -> pd.DataFrame:
    """Compute added triples from two DataFrames directly."""
    old_ids = set(
        kg_old["x_id"].astype(str) + "|" + kg_old["relation"] + "|" + kg_old["y_id"].astype(str)
    )
    kg_new = kg_new.copy()
    kg_new["_triple_id"] = (
        kg_new["x_id"].astype(str) + "|" + kg_new["relation"] + "|" + kg_new["y_id"].astype(str)
    )
    added = kg_new[~kg_new["_triple_id"].isin(old_ids)].drop(columns=["_triple_id"]).reset_index(drop=True)
    return added


def _entity_type_strategy(
    added_triples: pd.DataFrame,
) -> OrderedDict[str, pd.DataFrame]:
    """Group new triples by the dominant entity type they involve.

    A triple is assigned to a group based on the x_type or y_type that
    appears in that group's entity types. If both sides match different
    groups, the triple goes to the group of the rarer entity type
    (e.g., drug-protein -> drug_related, since drugs are rarer).

    Args:
        added_triples: DataFrame of triples added between snapshots.

    Returns:
        OrderedDict of task_name -> triples_dataframe.
    """
    tasks = OrderedDict()
    assigned = set()

    # Build reverse lookup: entity_type -> group_name
    type_to_group = {}
    for group_name, types in ENTITY_TYPE_GROUPS.items():
        for t in types:
            type_to_group[t] = group_name

    # Assign each triple to a group
    group_indices: dict[str, list[int]] = {name: [] for name in ENTITY_TYPE_GROUPS}
    overflow: list[int] = []

    for idx, row in added_triples.iterrows():
        x_group = type_to_group.get(row["x_type"])
        y_group = type_to_group.get(row["y_type"])

        if x_group and y_group:
            # Both sides have a group - assign to the first (rarer/more specific) one
            # Priority order matches ENTITY_TYPE_GROUPS (drug > disease > gene > ...)
            group_indices[x_group].append(idx)
        elif x_group:
            group_indices[x_group].append(idx)
        elif y_group:
            group_indices[y_group].append(idx)
        else:
            overflow.append(idx)

    # Create task DataFrames
    task_num = 1
    for group_name, indices in group_indices.items():
        if len(indices) > 0:
            tasks[f"task_{task_num}_{group_name}"] = added_triples.loc[indices].reset_index(drop=True)
            task_num += 1

    # Add overflow to the largest task or create separate
    if overflow:
        if tasks:
            largest_task = max(tasks.keys(), key=lambda k: len(tasks[k]))
            tasks[largest_task] = pd.concat(
                [tasks[largest_task], added_triples.loc[overflow]],
                ignore_index=True,
            )
            logger.info(f"Merged {len(overflow)} unassigned triples into {largest_task}")
        else:
            tasks["task_1_other"] = added_triples.loc[overflow].reset_index(drop=True)

    return tasks


def _relation_type_strategy(
    added_triples: pd.DataFrame,
) -> OrderedDict[str, pd.DataFrame]:
    """Group new triples by relation type.

    Each relation type becomes a separate task. Small tasks (< 100 triples)
    are merged during validation.

    Args:
        added_triples: DataFrame of triples added between snapshots.

    Returns:
        OrderedDict of task_name -> triples_dataframe.
    """
    tasks = OrderedDict()
    task_num = 1
    for rel_type in sorted(added_triples["relation"].unique()):
        subset = added_triples[added_triples["relation"] == rel_type].reset_index(drop=True)
        tasks[f"task_{task_num}_{rel_type}"] = subset
        task_num += 1
    return tasks


def _temporal_strategy(
    added_triples: pd.DataFrame,
) -> OrderedDict[str, pd.DataFrame]:
    """Split new triples into chunks for temporal ordering.

    Without explicit timestamps, we split added triples into N roughly
    equal chunks. This simulates a stream of new knowledge arriving
    over time.

    Args:
        added_triples: DataFrame of triples added between snapshots.

    Returns:
        OrderedDict of task_name -> triples_dataframe.
    """
    import numpy as np

    n_tasks = max(3, min(10, len(added_triples) // 1000))
    chunks = np.array_split(added_triples, n_tasks)

    tasks = OrderedDict()
    for i, chunk in enumerate(chunks):
        tasks[f"task_{i+1}_temporal_chunk"] = chunk.reset_index(drop=True)

    return tasks


def validate_task_sequence(
    tasks: OrderedDict[str, pd.DataFrame],
    min_triples: int = 100,
) -> OrderedDict[str, pd.DataFrame]:
    """Validate and clean task sequence.

    Merges tasks with fewer than min_triples into the nearest related task
    and verifies no overlap between non-base tasks.

    Args:
        tasks: Task sequence from create_task_sequence.
        min_triples: Minimum triples per task (smaller tasks get merged).

    Returns:
        Validated and potentially merged task sequence.
    """
    # Identify small tasks (skip task_0_base)
    small_tasks = []
    valid_tasks = OrderedDict()

    for name, df in tasks.items():
        if name == "task_0_base":
            valid_tasks[name] = df
            continue
        if len(df) < min_triples:
            small_tasks.append((name, df))
            logger.warning(f"Task {name} has only {len(df)} triples (< {min_triples})")
        else:
            valid_tasks[name] = df

    # Merge small tasks into the previous valid task
    if small_tasks:
        non_base = [k for k in valid_tasks if k != "task_0_base"]
        if non_base:
            merge_target = non_base[-1]  # merge into last valid task
            for name, df in small_tasks:
                valid_tasks[merge_target] = pd.concat(
                    [valid_tasks[merge_target], df], ignore_index=True
                )
                logger.info(f"Merged {name} ({len(df)} triples) into {merge_target}")
        else:
            # All non-base tasks are too small - combine them all
            combined = pd.concat([df for _, df in small_tasks], ignore_index=True)
            if len(combined) >= min_triples:
                valid_tasks["task_1_combined"] = combined
            else:
                logger.warning(
                    f"All added triples combined ({len(combined)}) still < {min_triples}. "
                    "Keeping as single task anyway."
                )
                valid_tasks["task_1_combined"] = combined

    # Verify no overlap between non-base tasks
    non_base_tasks = {k: v for k, v in valid_tasks.items() if k != "task_0_base"}
    all_ids: list[set] = []
    for name, df in non_base_tasks.items():
        ids = set(
            df["x_id"].astype(str) + "|" + df["relation"] + "|" + df["y_id"].astype(str)
        )
        for prev_ids in all_ids:
            overlap = ids & prev_ids
            if overlap:
                logger.warning(f"Found {len(overlap)} overlapping triples in {name}")
        all_ids.append(ids)

    logger.info(f"Validated task sequence: {len(valid_tasks)} tasks")
    return valid_tasks
