"""Run ablation studies for CMKL.

7 ablation studies:
1. struct_only: No text, no mol encoders — tests value of multimodal features
2. text_only: No GNN, text embeddings only — tests structural vs textual contribution
3. concat_fusion: Concatenation + MLP instead of cross-attention
4. global_ewc: Single lambda for all params (standard EWC, not modality-aware)
5. random_replay: Random buffer instead of K-means diverse selection
6. buffer_size_sweep: Buffer sizes [100, 250, 500, 1000, 2000, 5000]
7. lambda_sweep: Per-modality lambda sweep

Usage:
    # Single ablation
    python scripts/run_ablations.py --ablation struct_only --quick

    # All ablations
    python scripts/run_ablations.py --ablation all --seeds 42 123 456 789 1024

    # Buffer size sweep
    python scripts/run_ablations.py --ablation buffer_size_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
ABLATIONS = [
    "struct_only",
    "text_only",
    "concat_fusion",
    "moe_fusion",
    "global_ewc",
    "random_replay",
    "buffer_size_sweep",
    "lambda_sweep",
    "distillation",
]

# Buffer sizes to sweep
BUFFER_SIZES = [100, 250, 500, 1000, 2000, 5000]

# Lambda values to sweep (applied independently per modality)
LAMBDA_VALUES = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]


def get_base_config(args: argparse.Namespace) -> dict:
    """Build CMKL base config from args."""
    from src.baselines._base import load_task_sequence

    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )

    config = {
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "embedding_dim": args.embedding_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "num_attention_heads": args.num_attention_heads,
        "fusion_type": "cross_attention",
        "decoder_type": "DistMult",
        "lambda_struct": 10.0,
        "lambda_text": 5.0,
        "lambda_mol": 1.0,
        "replay_buffer_size": 1000,
        "replay_weight": 0.5,
        "samples_per_epoch": 50000,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
    }
    return config, task_seq, entity_to_id, relation_to_id


def run_single_config(
    config: dict,
    task_seq: dict,
    entity_to_id: dict,
    relation_to_id: dict,
    seeds: list[int],
    device: str,
    label: str,
    features: dict | None = None,
) -> list[dict]:
    """Run CMKL with a specific config across multiple seeds."""
    from src.models.cmkl import CMKL
    from src.evaluation.metrics import evaluate_continual_learning

    if features is None:
        features = {}

    task_names = list(task_seq.keys())
    all_results = []

    for seed in seeds:
        logger.info(f"  [{label}] seed={seed}")
        start = time.time()

        model = CMKL(config)
        results = model.train_continually(
            task_sequence=task_seq,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            device=device,
            seed=seed,
            edge_index=features.get("edge_index"),
            edge_type=features.get("edge_type"),
            text_embeddings=features.get("text_embeddings"),
            mol_fingerprints=features.get("mol_fingerprints"),
            node_has_text=features.get("node_has_text"),
            node_has_mol=features.get("node_has_mol"),
        )

        elapsed = time.time() - start
        R = np.array(results["results_matrix"])
        cl = evaluate_continual_learning(R, task_names)
        cl["seed"] = seed
        cl["results_matrix"] = results["results_matrix"]
        cl["training_time_s"] = elapsed
        all_results.append(cl)

        logger.info(f"    AP={cl['Average Performance (AP)']:.4f}, "
                    f"AF={cl['Average Forgetting (AF)']:.4f}, "
                    f"REM={cl['Remembering (REM)']:.4f} ({elapsed:.0f}s)")

    return all_results


def run_struct_only(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 1: Structural encoder only, no text or molecular."""
    c = {**config, "lambda_text": 0.0, "lambda_mol": 0.0}
    # Pass features but exclude text/mol by zeroing lambdas
    # (model ignores text/mol when they have zero EWC weight)
    feat = dict(features) if features else {}
    feat.pop("text_embeddings", None)
    feat.pop("node_has_text", None)
    feat.pop("mol_fingerprints", None)
    feat.pop("node_has_mol", None)
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "struct_only", feat)


def run_text_only(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 2: Text only (no GNN message passing)."""
    c = {**config, "num_gnn_layers": 0, "lambda_struct": 0.0, "lambda_mol": 0.0}
    feat = dict(features) if features else {}
    feat.pop("edge_index", None)
    feat.pop("edge_type", None)
    feat.pop("mol_fingerprints", None)
    feat.pop("node_has_mol", None)
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "text_only", feat)


def run_concat_fusion(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 3: Concatenation fusion instead of cross-attention."""
    c = {**config, "fusion_type": "concatenation"}
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "concat_fusion", features)


def run_global_ewc(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 4: Global EWC (same lambda for all modalities)."""
    global_lambda = 10.0
    c = {**config,
         "lambda_struct": global_lambda,
         "lambda_text": global_lambda,
         "lambda_mol": global_lambda}
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "global_ewc", features)


def run_random_replay(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 5: Random replay instead of K-means diverse."""
    c = {**config, "replay_strategy": "random"}
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "random_replay", features)


def run_buffer_size_sweep(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 6: Buffer size sweep."""
    all_sweep_results = {}
    for buf_size in BUFFER_SIZES:
        logger.info(f"Buffer size: {buf_size}")
        c = {**config, "replay_buffer_size": buf_size}
        results = run_single_config(c, task_seq, e2id, r2id, seeds, device,
                                    f"buf_{buf_size}", features)
        all_sweep_results[buf_size] = results
    return all_sweep_results


def run_lambda_sweep(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 7: Per-modality lambda sweep.

    Sweeps each modality's lambda independently while keeping others at default.
    """
    all_sweep_results = {}

    # Sweep lambda_struct
    for lam in LAMBDA_VALUES:
        key = f"struct_{lam}"
        logger.info(f"Lambda sweep: struct={lam}")
        c = {**config, "lambda_struct": lam}
        all_sweep_results[key] = run_single_config(
            c, task_seq, e2id, r2id, seeds, device, key, features)

    # Sweep lambda_text
    for lam in LAMBDA_VALUES:
        key = f"text_{lam}"
        logger.info(f"Lambda sweep: text={lam}")
        c = {**config, "lambda_text": lam}
        all_sweep_results[key] = run_single_config(
            c, task_seq, e2id, r2id, seeds, device, key, features)

    # Sweep lambda_mol
    for lam in LAMBDA_VALUES:
        key = f"mol_{lam}"
        logger.info(f"Lambda sweep: mol={lam}")
        c = {**config, "lambda_mol": lam}
        all_sweep_results[key] = run_single_config(
            c, task_seq, e2id, r2id, seeds, device, key, features)

    return all_sweep_results


def run_moe_fusion(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 9: MoE fusion instead of gated cross-attention."""
    c = {**config, "fusion_type": "moe", "load_balance_weight": 0.01}
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "moe_fusion", features)


def run_distillation(config, task_seq, e2id, r2id, seeds, device, features=None):
    """Ablation 8: Add knowledge distillation to CMKL."""
    c = {**config,
         "use_distillation": True,
         "distillation_temperature": 2.0,
         "distillation_alpha": 0.5}
    return run_single_config(c, task_seq, e2id, r2id, seeds, device,
                             "distillation", features)


ABLATION_DISPATCH = {
    "struct_only": run_struct_only,
    "text_only": run_text_only,
    "concat_fusion": run_concat_fusion,
    "moe_fusion": run_moe_fusion,
    "global_ewc": run_global_ewc,
    "random_replay": run_random_replay,
    "buffer_size_sweep": run_buffer_size_sweep,
    "lambda_sweep": run_lambda_sweep,
    "distillation": run_distillation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CMKL ablation studies")
    parser.add_argument(
        "--ablation", choices=ABLATIONS + ["all"], required=True,
        help="Which ablation to run",
    )
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix for output filename (e.g. _seed42)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: dim=64, epochs=3",
    )
    args = parser.parse_args()

    if args.quick:
        args.embedding_dim = 64
        args.num_epochs = 3
        args.num_gnn_layers = 1
        args.num_attention_heads = 2
        logger.info("Quick mode: dim=64, epochs=3")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, task_seq, e2id, r2id = get_base_config(args)
    ablations = ABLATIONS if args.ablation == "all" else [args.ablation]

    # Load pre-computed multimodal features
    from src.utils.feature_loader import load_features
    features = load_features()
    config["mol_input_dim"] = features.get("mol_input_dim", 1024)

    # Use feature vocab sizes to handle task subsets correctly
    if "num_entities" in features:
        config["num_entities"] = max(config["num_entities"],
                                      features["num_entities"])
    if "num_relations" in features:
        config["num_relations"] = max(config["num_relations"],
                                       features["num_relations"])

    for ablation_name in ablations:
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")

        start = time.time()
        fn = ABLATION_DISPATCH[ablation_name]
        results = fn(config, task_seq, e2id, r2id, args.seeds, args.device,
                     features)
        elapsed = time.time() - start

        # Save results
        result_path = output_dir / f"ablation_{ablation_name}{args.output_suffix}.json"
        # Handle sweep results (dict of lists) vs single ablation (list)
        save_data = {
            "ablation": ablation_name,
            "config": config,
            "task_names": list(task_seq.keys()),
            "seeds": args.seeds,
            "total_time_s": elapsed,
        }

        if isinstance(results, dict):
            # Sweep results: convert numpy for JSON serialization
            save_data["sweep_results"] = {
                str(k): v for k, v in results.items()
            }
        else:
            save_data["results"] = results

        with open(result_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Saved to {result_path} ({elapsed:.0f}s total)")


if __name__ == "__main__":
    main()
