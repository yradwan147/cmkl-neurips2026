"""Compute temporal diffs between PrimeKG snapshots.

Identifies added/removed/persistent triples, emerged/disappeared entities,
and new relation types between two KG snapshots. This is the core mechanism
for constructing the continual learning benchmark from real temporal evolution.

Triple identifiers are constructed as: x_id|relation|y_id

Usage:
    from src.data.temporal_diff import compute_kg_diff
    stats, added, removed, emerged = compute_kg_diff('data/kg_t0.csv', 'data/kg_t1.csv')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _make_triple_ids(kg: pd.DataFrame) -> set[str]:
    """Create canonical triple identifiers from a KG DataFrame.

    Each triple is represented as 'x_id|relation|y_id'. We also include
    the reverse 'y_id|relation|x_id' for undirected comparison if needed,
    but return only the canonical (x_id <= y_id lexicographically) form.

    Args:
        kg: PrimeKG DataFrame with x_id, relation, y_id columns.

    Returns:
        Set of triple identifier strings.
    """
    return set(
        kg["x_id"].astype(str) + "|" + kg["relation"] + "|" + kg["y_id"].astype(str)
    )


def _get_entities(kg: pd.DataFrame) -> set[str]:
    """Extract all unique entity identifiers (type:id) from a KG."""
    x_ents = set(kg["x_type"].astype(str) + ":" + kg["x_id"].astype(str))
    y_ents = set(kg["y_type"].astype(str) + ":" + kg["y_id"].astype(str))
    return x_ents | y_ents


def compute_kg_diff(
    kg_old_path: str,
    kg_new_path: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, set[str]]:
    """Compute differences between two KG snapshots.

    Creates triple identifiers (x_id|relation|y_id), then computes set
    differences to find added, removed, and persistent triples. Also
    identifies emerged/disappeared entities and new relation types.

    Args:
        kg_old_path: Path to older snapshot CSV.
        kg_new_path: Path to newer snapshot CSV.

    Returns:
        Tuple of (stats_dict, added_triples_df, removed_triples_df, emerged_entities).
        stats_dict contains counts for all categories.
    """
    logger.info(f"Loading old snapshot: {kg_old_path}")
    kg_old = pd.read_csv(kg_old_path, low_memory=False)
    logger.info(f"Loading new snapshot: {kg_new_path}")
    kg_new = pd.read_csv(kg_new_path, low_memory=False)

    # Normalize entity IDs if needed
    kg_old, kg_new = normalize_entity_ids(kg_old, kg_new)

    # Create triple identifiers
    triples_old = _make_triple_ids(kg_old)
    triples_new = _make_triple_ids(kg_new)

    # Compute set differences
    added_ids = triples_new - triples_old
    removed_ids = triples_old - triples_new
    persistent_ids = triples_old & triples_new

    logger.info(
        f"Triples: {len(added_ids):,} added, {len(removed_ids):,} removed, "
        f"{len(persistent_ids):,} persistent"
    )

    # Get added/removed as DataFrames (for downstream use)
    kg_new["_triple_id"] = (
        kg_new["x_id"].astype(str) + "|" + kg_new["relation"] + "|" + kg_new["y_id"].astype(str)
    )
    kg_old["_triple_id"] = (
        kg_old["x_id"].astype(str) + "|" + kg_old["relation"] + "|" + kg_old["y_id"].astype(str)
    )

    added_df = kg_new[kg_new["_triple_id"].isin(added_ids)].drop(columns=["_triple_id"]).reset_index(drop=True)
    removed_df = kg_old[kg_old["_triple_id"].isin(removed_ids)].drop(columns=["_triple_id"]).reset_index(drop=True)

    # Entity analysis
    entities_old = _get_entities(kg_old)
    entities_new = _get_entities(kg_new)
    emerged_entities = entities_new - entities_old
    disappeared_entities = entities_old - entities_new

    # Relation analysis
    relations_old = set(kg_old["relation"].unique())
    relations_new = set(kg_new["relation"].unique())
    emerged_relations = relations_new - relations_old
    disappeared_relations = relations_old - relations_new

    stats = {
        "old_snapshot": {
            "path": str(kg_old_path),
            "num_triples": len(kg_old),
            "num_entities": len(entities_old),
            "num_relations": len(relations_old),
        },
        "new_snapshot": {
            "path": str(kg_new_path),
            "num_triples": len(kg_new),
            "num_entities": len(entities_new),
            "num_relations": len(relations_new),
        },
        "diff": {
            "added_triples": len(added_ids),
            "removed_triples": len(removed_ids),
            "persistent_triples": len(persistent_ids),
            "emerged_entities": len(emerged_entities),
            "disappeared_entities": len(disappeared_entities),
            "emerged_relations": len(emerged_relations),
            "disappeared_relations": len(disappeared_relations),
        },
        "added_by_relation": added_df["relation"].value_counts().to_dict() if len(added_df) > 0 else {},
        "added_by_x_type": added_df["x_type"].value_counts().to_dict() if len(added_df) > 0 else {},
        "added_by_y_type": added_df["y_type"].value_counts().to_dict() if len(added_df) > 0 else {},
    }

    logger.info(f"Emerged entities: {len(emerged_entities):,}")
    logger.info(f"Disappeared entities: {len(disappeared_entities):,}")
    if emerged_relations:
        logger.info(f"New relation types: {emerged_relations}")

    return stats, added_df, removed_df, emerged_entities


def normalize_entity_ids(
    kg_old: pd.DataFrame,
    kg_new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize entity IDs across snapshots.

    Handles the PrimeKG disease ID grouping mismatch: t0 uses grouped
    MONDO IDs (e.g., "13924_12592_14672") while rebuilt t1 uses individual
    MONDO IDs (e.g., "13924"). Maps t1's individual disease IDs to t0's
    grouped super-node IDs where possible.

    Args:
        kg_old: Older snapshot DataFrame.
        kg_new: Newer snapshot DataFrame.

    Returns:
        Tuple of normalized DataFrames with consistent IDs.
    """
    # Ensure consistent string types for IDs
    for col in ["x_id", "y_id"]:
        kg_old[col] = kg_old[col].astype(str).str.strip()
        kg_new[col] = kg_new[col].astype(str).str.strip()

    # Ensure relation column is clean
    for col in ["relation"]:
        kg_old[col] = kg_old[col].astype(str).str.strip()
        kg_new[col] = kg_new[col].astype(str).str.strip()

    # Build mapping from individual MONDO IDs → grouped super-node IDs.
    # t0 has MONDO_grouped disease IDs like "13924_12592_14672".
    # t1 (rebuilt) has individual IDs like "13924".
    disease_rows = kg_old[
        (kg_old["x_source"] == "MONDO_grouped") | (kg_old["y_source"] == "MONDO_grouped")
    ] if "x_source" in kg_old.columns else pd.DataFrame()

    if len(disease_rows) > 0:
        grouped_ids = set()
        for col in ["x_id", "y_id"]:
            src_col = col.replace("id", "source")
            mask = kg_old[src_col] == "MONDO_grouped"
            grouped_ids.update(kg_old.loc[mask, col].unique())

        # Map individual → grouped
        individual_to_grouped: dict[str, str] = {}
        for gid in grouped_ids:
            parts = str(gid).split("_")
            if len(parts) > 1:
                for part in parts:
                    individual_to_grouped[part] = str(gid)

        if individual_to_grouped:
            mapped = 0
            # Map t1 disease IDs to t0's grouped IDs
            for col in ["x_id", "y_id"]:
                src_col = col.replace("id", "source")
                if src_col in kg_new.columns:
                    mask = kg_new[src_col] == "MONDO"
                    ids = kg_new.loc[mask, col]
                    new_ids = ids.map(lambda x: individual_to_grouped.get(str(x), x))
                    changed = (new_ids != ids).sum()
                    mapped += changed
                    kg_new.loc[mask, col] = new_ids
                    # Update source to MONDO_grouped for mapped IDs
                    kg_new.loc[mask & (new_ids != ids), src_col] = "MONDO_grouped"

            if mapped > 0:
                logger.info(f"Mapped {mapped} t1 disease IDs to t0 grouped super-node IDs")

    return kg_old, kg_new


def save_diff_report(
    stats: dict,
    output_path: str,
) -> None:
    """Save temporal diff statistics as JSON.

    Args:
        stats: Dictionary from compute_kg_diff.
        output_path: Path to save JSON report.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Saved diff report to {output_path}")


def create_simulated_t1(
    kg_t0_path: str,
    output_path: str,
    add_fraction: float = 0.05,
    remove_fraction: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a simulated t1 snapshot from t0 for development/testing.

    Simulates temporal evolution by:
    1. Keeping most of t0 (persistent triples)
    2. Removing a small fraction (simulating deprecated knowledge)
    3. Creating new triples by rewiring some existing edges (simulating new discoveries)

    This is a development fallback when the real t1 rebuild is not available.
    The real t1 should be built from July 2023 database sources.

    Args:
        kg_t0_path: Path to t0 snapshot CSV.
        output_path: Path to save simulated t1 CSV.
        add_fraction: Fraction of t0 edges to add as new (default 5%).
        remove_fraction: Fraction of t0 edges to remove (default 1%).
        seed: Random seed.

    Returns:
        Simulated t1 DataFrame.
    """
    import numpy as np

    logger.info(f"Creating simulated t1 from {kg_t0_path}")
    kg = pd.read_csv(kg_t0_path, low_memory=False)
    rng = np.random.RandomState(seed)

    n = len(kg)
    n_remove = int(n * remove_fraction)
    n_add = int(n * add_fraction)

    # Remove some triples (simulate deprecated knowledge)
    remove_idx = rng.choice(n, size=n_remove, replace=False)
    kg_kept = kg.drop(index=remove_idx).reset_index(drop=True)

    # Create new triples by sampling head/tail from existing entities
    # and pairing them with existing relation types
    # Focus new triples on drug-disease and drug-protein (most dynamic in biomedical KGs)
    dynamic_relations = [
        "indication", "contraindication", "off-label use",
        "drug_protein", "disease_protein", "drug_effect",
    ]

    # Filter to dynamic relations for new triple generation
    dynamic_edges = kg[kg["relation"].isin(dynamic_relations)]
    if len(dynamic_edges) < n_add:
        dynamic_edges = kg  # fallback to all edges

    # Sample and rewire: keep relation and x, replace y with a random node of same type
    sample_idx = rng.choice(len(dynamic_edges), size=n_add, replace=True)
    new_triples = dynamic_edges.iloc[sample_idx].copy().reset_index(drop=True)

    # Rewire y-side: for each new triple, pick a random entity of the same y_type
    for y_type in new_triples["y_type"].unique():
        mask = new_triples["y_type"] == y_type
        candidates = kg[kg["y_type"] == y_type]["y_id"].unique()
        if len(candidates) > 0:
            new_triples.loc[mask, "y_id"] = rng.choice(candidates, size=mask.sum(), replace=True)
            # Also update y_name and y_source from the KG
            y_lookup = kg[kg["y_type"] == y_type].drop_duplicates("y_id").set_index("y_id")
            for idx in new_triples[mask].index:
                yid = new_triples.at[idx, "y_id"]
                if yid in y_lookup.index:
                    new_triples.at[idx, "y_name"] = y_lookup.at[yid, "y_name"]
                    new_triples.at[idx, "y_source"] = y_lookup.at[yid, "y_source"]

    # Remove duplicates (new triples that happen to match existing ones)
    kg_t1 = pd.concat([kg_kept, new_triples], ignore_index=True)
    kg_t1 = kg_t1.drop_duplicates(
        subset=["x_id", "relation", "y_id"], keep="first"
    ).reset_index(drop=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    kg_t1.to_csv(output_path, index=False)

    logger.info(
        f"Simulated t1: {len(kg_t1):,} triples "
        f"(removed {n_remove:,}, added up to {n_add:,} new)"
    )

    return kg_t1
