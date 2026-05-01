"""Evaluation metrics for continual knowledge graph learning.

Link prediction metrics: MRR, Hits@K, AUPRC
Continual learning metrics: AP, AF, BWT, FWT, REM

The results matrix R[i][j] represents performance on task j's test set
after training on task i. This matrix is the core data structure for
computing all continual learning metrics.

Usage:
    from src.evaluation.metrics import evaluate_continual_learning, compute_mrr
    cl_metrics = evaluate_continual_learning(results_matrix, task_names)
    mrr = compute_mrr(ranks)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Link prediction metrics
# ---------------------------------------------------------------------------

def compute_mrr(ranks: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of integer ranks for correct entities (1-based).

    Returns:
        MRR score in [0, 1].
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(1.0 / ranks))


def compute_hits_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Compute Hits@K metric.

    Args:
        ranks: Array of integer ranks for correct entities (1-based).
        k: Cutoff value (1, 3, or 10).

    Returns:
        Hits@K score (proportion of ranks <= k).
    """
    ranks = np.asarray(ranks)
    return float(np.mean(ranks <= k))


def compute_auprc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Compute Area Under Precision-Recall Curve.

    Used for drug repurposing evaluation following TxGNN protocol.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted scores.

    Returns:
        AUPRC score.
    """
    return float(average_precision_score(y_true, y_scores))


def compute_link_prediction_metrics(
    ranks: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float]:
    """Compute all standard link prediction metrics from ranks.

    Args:
        ranks: Array of integer ranks (1-based) for correct entities.
        ks: Tuple of K values for Hits@K.

    Returns:
        Dict with 'MRR' and 'Hits@K' for each K.
    """
    ranks = np.asarray(ranks)
    metrics = {"MRR": compute_mrr(ranks)}
    for k in ks:
        metrics[f"Hits@{k}"] = compute_hits_at_k(ranks, k)
    return metrics


def compute_multihop_metrics(
    ranks: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict[str, float]:
    """Compute link prediction metrics with 'multihop_' prefix.

    Same metrics as standard LP but prefixed for multi-hop evaluation.
    Reuses compute_mrr and compute_hits_at_k.

    Args:
        ranks: Array of integer ranks (1-based).
        ks: Tuple of K values for Hits@K.

    Returns:
        Dict with 'multihop_MRR', 'multihop_Hits@K', 'multihop_mean_rank'.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    metrics = {
        "multihop_MRR": compute_mrr(ranks),
        "multihop_mean_rank": float(np.mean(ranks)),
    }
    for k in ks:
        metrics[f"multihop_Hits@{k}"] = compute_hits_at_k(ranks, k)
    return metrics


# ---------------------------------------------------------------------------
# KGQA metrics
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match after normalization.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    import re
    pred = re.sub(r"[^\w\s]", "", prediction.lower().strip())
    gold = re.sub(r"[^\w\s]", "", ground_truth.lower().strip())
    pred = " ".join(pred.split())
    gold = " ".join(gold.split())
    return 1.0 if pred == gold else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        Token F1 score in [0, 1].
    """
    import re
    from collections import Counter
    pred = " ".join(re.sub(r"[^\w\s]", "", prediction.lower().strip()).split())
    gold = " ".join(re.sub(r"[^\w\s]", "", ground_truth.lower().strip()).split())
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Node classification metrics
# ---------------------------------------------------------------------------

def compute_nc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute node classification metrics.

    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dict with 'accuracy', 'macro_f1', 'weighted_f1'.
    """
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Continual learning metrics
# ---------------------------------------------------------------------------

def evaluate_continual_learning(
    results_matrix: np.ndarray,
    task_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute all continual learning metrics from results matrix R.

    R[i][j] = performance on task j's test set after training on task i.
    The matrix is lower-triangular: R[i][j] is only valid when i >= j
    (can only evaluate on tasks seen so far).

    Args:
        results_matrix: NumPy array of shape (n_tasks, n_tasks).
        task_names: Optional list of task names for logging.

    Returns:
        Dict with AP, AF, BWT, FWT, REM metrics.
    """
    R = np.asarray(results_matrix, dtype=np.float64)
    n = R.shape[0]

    if n < 2:
        return {
            "Average Performance (AP)": float(R[0, 0]),
            "Average Forgetting (AF)": 0.0,
            "Backward Transfer (BWT)": 0.0,
            "Forward Transfer (FWT)": 0.0,
            "Remembering (REM)": 1.0,
        }

    # AP: Average Performance after training on the final task
    # AP = (1/n) * sum_{j=0}^{n-1} R[n-1, j]
    ap = float(np.mean(R[-1, :]))

    # AF: Average Forgetting
    # For each task j < n-1: forgetting_j = max_{i in [j..n-2]} R[i,j] - R[n-1,j]
    forgetting = []
    for j in range(n - 1):
        max_perf = np.max(R[j:n - 1, j])  # best perf on task j before final task
        forgetting.append(max_perf - R[-1, j])
    af = float(np.mean(forgetting))

    # BWT: Backward Transfer
    # BWT = (1/(n-1)) * sum_{j=0}^{n-2} (R[n-1, j] - R[j, j])
    bwt_vals = [R[-1, j] - R[j, j] for j in range(n - 1)]
    bwt = float(np.mean(bwt_vals))

    # FWT: Forward Transfer
    # FWT = (1/(n-1)) * sum_{j=1}^{n-1} R[j-1, j]
    # R[j-1, j] = performance on task j BEFORE training on it
    # (using model trained up to task j-1)
    fwt_vals = [R[j - 1, j] for j in range(1, n)]
    fwt = float(np.mean(fwt_vals))

    # REM: Remembering = 1 - |min(BWT, 0)|
    rem = 1.0 - abs(min(bwt, 0.0))

    metrics = {
        "Average Performance (AP)": ap,
        "Average Forgetting (AF)": af,
        "Backward Transfer (BWT)": bwt,
        "Forward Transfer (FWT)": fwt,
        "Remembering (REM)": rem,
    }

    if task_names:
        logger.info(f"CL Metrics ({len(task_names)} tasks):")
        for name, val in metrics.items():
            logger.info(f"  {name}: {val:.4f}")

    return metrics
