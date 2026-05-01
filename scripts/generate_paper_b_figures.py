"""Generate figures for Paper B (CMKL method paper).

Creates:
1. Per-task CMKL performance (diagonal of learning matrix) — CMKL-only view
2. Fusion ablation bar chart (Table 3 visualization)

Usage:
    python scripts/generate_paper_b_figures.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "papers" / "paper_b_method" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_SHORT_NAMES = [
    "Base (t0)", "Disease (r1)", "Drug (r1)", "Disease (r2)", "Gene/Prot (r2)",
    "Gene/Prot (r3)", "Phenotype (r3)", "BioProcess (r4)", "Phenotype (r4)", "Anat/Path (r5)",
]


def load_cmkl_matrices() -> list[np.ndarray]:
    """Load CMKL MoE DistMult learning matrices from results_run12."""
    results_dir = PROJECT_ROOT / "results_run12"
    files = sorted(glob.glob(str(results_dir / "cmkl_DistMult_seed*.json")))
    # Exclude score-fusion variants
    files = [f for f in files if "sf_" not in f and "a0." not in f and "a1." not in f and "a2." not in f]

    matrices = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data.get("results"), list) and data["results"]:
            mat = data["results"][0].get("results_matrix")
            if mat is not None:
                matrices.append(np.array(mat))
    return matrices


def generate_cmkl_per_task() -> None:
    """Per-task peak MRR for CMKL only (different from Paper A's multi-method fig)."""
    matrices = load_cmkl_matrices()
    if not matrices:
        print("No CMKL matrices found, skipping per-task figure")
        return

    # Also load struct-only and text-only for comparison
    ablation_configs = {
        "CMKL (MoE)": ("results_run12", "cmkl_DistMult_seed*.json", ["sf_", "a0.", "a1.", "a2."]),
        "Struct only": ("results_run8", "ablation_struct_only_seed*.json", []),
        "Text only": ("results_run8", "ablation_text_only_seed*.json", []),
    }
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    n_tasks = 10
    bar_width = 0.25
    x = np.arange(n_tasks)

    for idx, (label, (rdir, pattern, exclude)) in enumerate(ablation_configs.items()):
        results_dir = PROJECT_ROOT / rdir
        files = sorted(glob.glob(str(results_dir / pattern)))
        files = [f for f in files if not any(ex in f for ex in exclude)]

        if not files:
            continue

        all_mats = []
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data.get("results"), list) and data["results"]:
                mat = data["results"][0].get("results_matrix")
                if mat is not None:
                    m = np.array(mat)
                    # Pad to 10x10 if needed
                    if m.shape[0] < n_tasks:
                        p = np.zeros((n_tasks, n_tasks))
                        p[:m.shape[0], :m.shape[1]] = m
                        m = p
                    all_mats.append(m)

        if not all_mats:
            continue

        avg = np.mean(all_mats, axis=0)
        std = np.std(all_mats, axis=0) if len(all_mats) > 1 else np.zeros_like(avg)
        diag_mean = np.diag(avg)
        diag_std = np.diag(std)

        offset = (idx - len(ablation_configs) / 2 + 0.5) * bar_width
        ax.bar(x + offset, diag_mean, bar_width, yerr=diag_std, capsize=2,
               label=label, color=colors[idx], alpha=0.85,
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Peak filtered MRR", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT_NAMES, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = FIGURES_DIR / "cmkl_per_task.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def generate_fusion_ablation_bar() -> None:
    """Bar chart of fusion strategy comparison (Table 3 data)."""
    strategies = [
        "MoE", "Gated\nCross-Attn", "Concat",
        "Score-Level", "SF+OGM", "SF+CL", "SF+OGM+CL",
    ]
    ap_means = [0.062, 0.060, 0.063, 0.049, 0.046, 0.043, 0.044]
    ap_stds = [0.010, 0.008, 0.008, 0.003, 0.001, 0.003, 0.002]

    colors = ["#e41a1c"] * 3 + ["#377eb8"] * 4  # embedding vs score-level

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(strategies))
    bars = ax.bar(x, ap_means, yerr=ap_stds, capsize=3,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add reference lines
    ax.axhline(y=0.071, color="gray", linestyle="--", linewidth=1, label="Struct only (0.071)")
    ax.axhline(y=0.136, color="gray", linestyle=":", linewidth=1, label="Text only (0.136)")

    ax.set_xlabel("Fusion Strategy", fontsize=11)
    ax.set_ylabel("AP (filtered MRR)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.15)

    # Add labels
    ax.text(1.0, 0.145, "Embedding-level", ha="center", fontsize=9, fontstyle="italic", color="#e41a1c")
    ax.text(4.5, 0.145, "Score-level", ha="center", fontsize=9, fontstyle="italic", color="#377eb8")

    plt.tight_layout()
    out = FIGURES_DIR / "fusion_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Paper B figures...")
    generate_cmkl_per_task()
    generate_fusion_ablation_bar()
    print("Done!")
