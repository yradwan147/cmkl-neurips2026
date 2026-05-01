"""Stratified evaluation: compute filtered MRR separately on persistent,
removed, and added subsets of the final test set.

The stratification is derived from the benchmark's $t_0 \to t_1$ partition:

  * persistent : triples present in both t_0 and t_1 snapshots
  * removed    : triples present in t_0 but deprecated in t_1
  * added      : triples new in t_1 (introduced in tasks 1..9)

An ideal continual model should remember persistent triples, forget removed
triples, and acquire added triples. This module evaluates all three on a
trained final-state model and returns a dict of per-stratum metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from pykeen.triples import TriplesFactory

from src.baselines._base import evaluate_link_prediction, make_triples_factory

logger = logging.getLogger(__name__)


def load_stratification(
    stratification_path: str | Path = "data/benchmark/test_stratification.json",
) -> dict:
    """Load the precomputed test-set stratification counts.

    Returns the JSON object with per-task triple counts in each stratum.
    """
    with open(stratification_path) as f:
        return json.load(f)


def build_stratum_masks(
    task_0_test_factory: TriplesFactory,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    persistent_triples: set[tuple[str, str, str]],
    removed_triples: set[tuple[str, str, str]],
    id_to_entity: dict[int, str] | None = None,
    id_to_relation: dict[int, str] | None = None,
) -> dict[str, np.ndarray]:
    """Partition task-0 test triples into persistent / removed boolean masks.

    Parameters
    ----------
    task_0_test_factory : TriplesFactory
        PyKEEN TriplesFactory built from task_0_base/test.txt (int-mapped).
    entity_to_id, relation_to_id : dict[str, int]
        Forward vocab maps.
    persistent_triples, removed_triples : set of (head, relation, tail) str tuples
        Stratum membership as string triples (from kg_t0.csv ∩/∖ kg_t1.csv).

    Returns
    -------
    dict[str, np.ndarray]
        Boolean masks of length == number of task-0 test triples, one per stratum.
    """
    if id_to_entity is None:
        id_to_entity = {v: k for k, v in entity_to_id.items()}
    if id_to_relation is None:
        id_to_relation = {v: k for k, v in relation_to_id.items()}

    test_mapped = task_0_test_factory.mapped_triples.cpu().numpy()
    n = test_mapped.shape[0]
    persist_mask = np.zeros(n, dtype=bool)
    remove_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        h = id_to_entity.get(int(test_mapped[i, 0]))
        r = id_to_relation.get(int(test_mapped[i, 1]))
        t = id_to_entity.get(int(test_mapped[i, 2]))
        if h is None or r is None or t is None:
            continue
        key = (h, r, t)
        if key in persistent_triples:
            persist_mask[i] = True
        elif key in removed_triples:
            remove_mask[i] = True

    logger.info(
        "stratum masks: total=%d, persistent=%d (%.1f%%), removed=%d (%.1f%%), "
        "unassigned=%d",
        n, persist_mask.sum(), 100 * persist_mask.sum() / n,
        remove_mask.sum(), 100 * remove_mask.sum() / n,
        n - persist_mask.sum() - remove_mask.sum(),
    )
    return {"persistent": persist_mask, "removed": remove_mask}


def _subset_factory(base: TriplesFactory, mask: np.ndarray) -> TriplesFactory:
    """Slice a TriplesFactory by a boolean mask, preserving vocab."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    sub = base.mapped_triples[torch.as_tensor(idx, dtype=torch.long)]
    return TriplesFactory(
        mapped_triples=sub,
        entity_to_id=base.entity_to_id,
        relation_to_id=base.relation_to_id,
        create_inverse_triples=False,
    )


def load_snapshot_triple_sets(
    t0_csv: str | Path = "data/benchmark/snapshots/kg_t0.csv",
    t1_csv: str | Path = "data/benchmark/snapshots/kg_t1.csv",
) -> tuple[set, set, set]:
    """Return (persistent, added, removed) sets of (h,r,t) string tuples."""
    import pandas as pd

    t0 = pd.read_csv(t0_csv, low_memory=False, usecols=["relation", "x_id", "y_id"])
    t1 = pd.read_csv(t1_csv, low_memory=False, usecols=["relation", "x_id", "y_id"])

    t0_trips = set(zip(t0.x_id.astype(str), t0.relation.astype(str), t0.y_id.astype(str)))
    t1_trips = set(zip(t1.x_id.astype(str), t1.relation.astype(str), t1.y_id.astype(str)))

    persistent = t0_trips & t1_trips
    added = t1_trips - t0_trips
    removed = t0_trips - t1_trips
    logger.info(
        "snapshot sets: persistent=%d, added=%d, removed=%d",
        len(persistent), len(added), len(removed),
    )
    return persistent, added, removed


def stratified_eval(
    model: torch.nn.Module,
    task_seq: dict,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    device: str = "cpu",
    batch_size: int = 64,
    persistent_triples: set | None = None,
    removed_triples: set | None = None,
    max_per_stratum: int = 30_000,
) -> dict[str, dict[str, float]]:
    """Evaluate the final-state model on persistent vs removed task-0 test
    triples and on the added-strata of tasks 1..9.

    Parameters
    ----------
    model : torch.nn.Module
        Trained final-task KGE model (PyKEEN or CMKL-compatible).
    task_seq : OrderedDict
        Task sequence dict (task_name -> {train, val, test int arrays}).
    entity_to_id, relation_to_id : dict[str, int]
        Vocab maps matching task_seq.
    device : str
        Torch device.
    batch_size : int
        Eval batch size.
    persistent_triples, removed_triples : set of (h, r, t) str tuples
        Stratum membership, precomputed from kg_t0 ∩/∖ kg_t1. If None, loads
        from default snapshot CSVs.
    max_per_stratum : int
        Max test triples per stratum to evaluate (for speed).

    Returns
    -------
    dict
        {
          "persistent": {"MRR": ..., "Hits@1": ..., "Hits@3": ..., "Hits@10": ..., "n": int},
          "removed":    {...},
          "added":      {...},  # mean across tasks 1..9
        }
    """
    if persistent_triples is None or removed_triples is None:
        persistent_triples, _, removed_triples = load_snapshot_triple_sets()

    # Build all-known-triples filter across the full sequence (required for filtered MRR)
    all_known_lists = []
    for name, splits in task_seq.items():
        for split_name in ("train", "val", "test"):
            arr = splits.get(split_name)
            if arr is not None and len(arr) > 0:
                all_known_lists.append(arr)
    if all_known_lists:
        all_known = torch.as_tensor(
            np.concatenate(all_known_lists, axis=0), dtype=torch.long
        )
    else:
        all_known = None

    # === Stratify task-0 test: persistent vs removed ===
    task_0_name = "task_0_base"
    t0_splits = task_seq.get(task_0_name)
    results: dict[str, dict[str, float]] = {}

    if t0_splits is not None and len(t0_splits.get("test", [])) > 0:
        t0_test_factory = make_triples_factory(
            t0_splits["test"], entity_to_id, relation_to_id
        )
        masks = build_stratum_masks(
            t0_test_factory,
            entity_to_id,
            relation_to_id,
            persistent_triples,
            removed_triples,
        )
        for stratum_name, mask in masks.items():
            sub = _subset_factory(t0_test_factory, mask)
            if sub is None:
                results[stratum_name] = {"MRR": float("nan"), "n": 0}
                continue
            logger.info("Evaluating stratum '%s': %d triples", stratum_name, mask.sum())
            metrics = evaluate_link_prediction(
                model, sub,
                device=device,
                batch_size=batch_size,
                all_known_mapped_triples=all_known,
                max_test_triples=max_per_stratum,
            )
            metrics["n"] = int(mask.sum())
            results[stratum_name] = metrics

    # === Added stratum: average over tasks 1..T-1 ===
    added_metrics_list = []
    for name in list(task_seq.keys())[1:]:  # skip task_0
        splits = task_seq[name]
        test_arr = splits.get("test")
        if test_arr is None or len(test_arr) == 0:
            continue
        tf = make_triples_factory(test_arr, entity_to_id, relation_to_id)
        logger.info("Evaluating added stratum task '%s': %d triples", name, len(test_arr))
        m = evaluate_link_prediction(
            model, tf,
            device=device,
            batch_size=batch_size,
            all_known_mapped_triples=all_known,
            max_test_triples=max_per_stratum,
        )
        m["n"] = int(len(test_arr))
        m["task"] = name
        added_metrics_list.append(m)

    if added_metrics_list:
        # Per-task values kept; mean of task MRR over tasks 1..T-1
        mean_mrr = float(np.mean([m["MRR"] for m in added_metrics_list]))
        results["added"] = {
            "MRR": mean_mrr,
            "Hits@1": float(np.mean([m["Hits@1"] for m in added_metrics_list])),
            "Hits@3": float(np.mean([m["Hits@3"] for m in added_metrics_list])),
            "Hits@10": float(np.mean([m["Hits@10"] for m in added_metrics_list])),
            "n": int(sum(m["n"] for m in added_metrics_list)),
            "per_task": added_metrics_list,
        }

    return results
