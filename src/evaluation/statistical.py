"""Statistical significance testing for experiment comparisons.

Uses paired t-tests or Wilcoxon signed-rank tests to compare methods
across random seeds. All experiments use 5 seeds: [42, 123, 456, 789, 1024].
Report mean +/- std with significance at p < 0.05.

Usage:
    from src.evaluation.statistical import compute_significance, summarize_results
    sig = compute_significance(results_method_a, results_method_b)
    summary = summarize_results(all_seed_results)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def compute_significance(
    results_method_a: np.ndarray,
    results_method_b: np.ndarray,
    alpha: float = 0.05,
    test: str = "paired_t",
) -> dict:
    """Test statistical significance between two methods across random seeds.

    Args:
        results_method_a: Array of metric values across seeds for method A.
        results_method_b: Array of metric values across seeds for method B.
        alpha: Significance threshold.
        test: Test type - 'paired_t' or 'wilcoxon'.

    Returns:
        Dict with test statistic, p_value, significant, effect_size.
    """
    a = np.asarray(results_method_a, dtype=np.float64)
    b = np.asarray(results_method_b, dtype=np.float64)

    if test == "paired_t":
        t_stat, p_value = sp_stats.ttest_rel(a, b)
        return {
            "test": "paired_t",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < alpha,
            "mean_diff": float(np.mean(a - b)),
        }
    elif test == "wilcoxon":
        # Wilcoxon needs non-zero differences
        diffs = a - b
        if np.all(diffs == 0):
            return {
                "test": "wilcoxon",
                "w_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "mean_diff": 0.0,
            }
        w_stat, p_value = sp_stats.wilcoxon(a, b)
        return {
            "test": "wilcoxon",
            "w_statistic": float(w_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < alpha,
            "mean_diff": float(np.mean(a - b)),
        }
    else:
        raise ValueError(f"Unknown test: {test}. Use 'paired_t' or 'wilcoxon'.")


def summarize_results(
    all_seed_results: list[dict],
) -> dict[str, str]:
    """Aggregate results across seeds into mean +/- std format.

    Args:
        all_seed_results: List of metric dicts, one per seed.
            Each dict maps metric_name -> float value.

    Returns:
        Dict mapping metric_name -> "mean +/- std" string.
    """
    if not all_seed_results:
        return {}

    metric_names = list(all_seed_results[0].keys())
    summary = {}
    for name in metric_names:
        vals = [r[name] for r in all_seed_results if name in r]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            summary[name] = f"{mean:.4f} +/- {std:.4f}"
    return summary


def summarize_results_numeric(
    all_seed_results: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate results across seeds into numeric mean and std.

    Args:
        all_seed_results: List of metric dicts, one per seed.

    Returns:
        Dict mapping metric_name -> {'mean': float, 'std': float}.
    """
    if not all_seed_results:
        return {}

    metric_names = list(all_seed_results[0].keys())
    summary = {}
    for name in metric_names:
        vals = [r[name] for r in all_seed_results if name in r]
        if vals:
            summary[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            }
    return summary


def pairwise_significance_table(
    method_results: dict[str, list[dict]],
    metric: str = "Average Performance (AP)",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pairwise significance between all methods.

    Args:
        method_results: Dict mapping method_name -> list of per-seed results.
            Each result dict should contain the specified metric.
        metric: Which metric to compare.
        alpha: Significance threshold.

    Returns:
        DataFrame with p-values for all method pairs.
    """
    methods = list(method_results.keys())
    n = len(methods)
    p_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            vals_i = np.array([r[metric] for r in method_results[methods[i]]])
            vals_j = np.array([r[metric] for r in method_results[methods[j]]])
            sig = compute_significance(vals_i, vals_j, alpha=alpha)
            p_matrix[i, j] = sig["p_value"]
            p_matrix[j, i] = sig["p_value"]

    return pd.DataFrame(p_matrix, index=methods, columns=methods)
