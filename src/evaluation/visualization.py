"""Visualization utilities for experiment results.

Generates publication-quality plots for:
- Results matrix heatmaps (R[i][j] per method)
- Bar charts comparing AP, AF, BWT, FWT across methods
- Forgetting curves over task sequence
- Per-relation-type performance breakdown
- Buffer size and lambda sweep sensitivity plots

Usage:
    from src.evaluation.visualization import plot_results_heatmap, plot_forgetting_curves
    plot_results_heatmap(results_matrix, task_names, save_path='results/figures/heatmap.pdf')
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_results_heatmap(
    results_matrix: np.ndarray,
    task_names: list[str],
    method_name: str = "",
    save_path: str | None = None,
) -> None:
    """Plot results matrix as a heatmap.

    R[i][j] = performance on task j after training on task i.
    Only lower-triangular entries are meaningful (tasks seen so far).

    Args:
        results_matrix: Shape (n_tasks, n_tasks).
        task_names: Task labels for axes.
        method_name: Title for the plot.
        save_path: If provided, save figure to this path.
    """
    R = np.asarray(results_matrix)
    n = R.shape[0]

    # Mask upper triangle (tasks not yet seen)
    mask = np.triu(np.ones_like(R, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    sns.heatmap(
        R, mask=mask, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=task_names, yticklabels=task_names,
        vmin=0, vmax=max(0.5, R[~mask].max() * 1.1) if R[~mask].size > 0 else 0.5,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Evaluated on task")
    ax.set_ylabel("After training on task")
    title = "Results Matrix"
    if method_name:
        title += f" - {method_name}"
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved heatmap to {save_path}")
    plt.close(fig)


def plot_method_comparison(
    method_metrics: dict[str, dict],
    metrics_to_plot: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing methods across CL metrics.

    Args:
        method_metrics: Dict of method_name -> {metric_name: value}.
        metrics_to_plot: Which metrics to include.
            Default: AP, AF, BWT, FWT, REM.
        save_path: If provided, save figure to this path.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            "Average Performance (AP)",
            "Average Forgetting (AF)",
            "Backward Transfer (BWT)",
            "Forward Transfer (FWT)",
            "Remembering (REM)",
        ]

    methods = list(method_metrics.keys())
    n_methods = len(methods)
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", n_methods)

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [method_metrics[m].get(metric, 0.0) for m in methods]
        bars = ax.bar(range(n_methods), vals, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        # Short label
        short = metric.split("(")[-1].rstrip(")") if "(" in metric else metric
        ax.set_title(short)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Method Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved comparison to {save_path}")
    plt.close(fig)


def plot_forgetting_curves(
    method_results: dict[str, np.ndarray],
    task_names: list[str],
    target_task: int = 0,
    save_path: str | None = None,
) -> None:
    """Plot forgetting curves over task sequence for multiple methods.

    Shows how performance on a target task (default: task 0) changes
    as more tasks are learned.

    Args:
        method_results: Dict of method_name -> results_matrix (n x n).
        task_names: Task labels.
        target_task: Which task to track forgetting for.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(method_results))

    for (method, R), color in zip(method_results.items(), colors):
        R = np.asarray(R)
        n = R.shape[0]
        # Performance on target_task after training on task 0, 1, 2, ...
        perf = [R[i, target_task] for i in range(target_task, n)]
        x = list(range(target_task, n))
        ax.plot(x, perf, marker="o", label=method, color=color, linewidth=2)

    ax.set_xlabel("After training on task")
    ax.set_ylabel(f"Performance on {task_names[target_task]}")
    ax.set_title(f"Forgetting Curve - {task_names[target_task]}")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved forgetting curves to {save_path}")
    plt.close(fig)


def plot_sensitivity_sweep(
    sweep_values: list,
    sweep_results: list[dict],
    param_name: str,
    metrics_to_plot: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot hyperparameter sensitivity (buffer size, lambda sweeps).

    Args:
        sweep_values: List of parameter values tested.
        sweep_results: List of metric dicts, one per value.
        param_name: Name of the swept parameter.
        metrics_to_plot: Which metrics to show. Default: AP, AF.
        save_path: If provided, save figure to this path.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            "Average Performance (AP)",
            "Average Forgetting (AF)",
        ]

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                              figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [r.get(metric, 0.0) for r in sweep_results]
        ax.plot(range(len(sweep_values)), vals, marker="s", linewidth=2)
        ax.set_xticks(range(len(sweep_values)))
        ax.set_xticklabels([str(v) for v in sweep_values])
        ax.set_xlabel(param_name)
        short = metric.split("(")[-1].rstrip(")") if "(" in metric else metric
        ax.set_ylabel(short)
        ax.set_title(f"{short} vs {param_name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved sweep plot to {save_path}")
    plt.close(fig)


def plot_multihop_comparison(
    method_results: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> None:
    """Grouped bar chart comparing multi-hop MRR across methods and path types.

    Args:
        method_results: Dict of method_name -> {path_description: MRR_value}.
            Each key is a path type description (e.g. "drug -> protein -> disease").
        save_path: If provided, save figure to this path.
    """
    if not method_results:
        logger.warning("No multi-hop results to plot")
        return

    methods = list(method_results.keys())
    # Collect all path types across methods
    all_path_types = sorted({
        pt for m_res in method_results.values() for pt in m_res.keys()
    })

    if not all_path_types:
        logger.warning("No path types found in multi-hop results")
        return

    n_methods = len(methods)
    n_types = len(all_path_types)
    x = np.arange(n_types)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, n_types * 1.5), 6))
    colors = sns.color_palette("Set2", n_methods)

    for i, method in enumerate(methods):
        vals = [method_results[method].get(pt, 0.0) for pt in all_path_types]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=colors[i])

    # Shorten path type labels for readability
    short_labels = [pt.replace(" -> ", "\u2192") for pt in all_path_types]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Multi-hop MRR")
    ax.set_title("Multi-Hop Reasoning: MRR by Path Type")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved multi-hop comparison to {save_path}")
    plt.close(fig)
