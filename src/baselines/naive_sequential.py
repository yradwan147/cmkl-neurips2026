"""Baseline 1: Naive Sequential Training (Lower Bound).

Trains a KGE model (TransE/ComplEx/DistMult/RotatE) on each task sequentially
without any continual learning mechanism. Establishes the maximum-forgetting
reference point. Uses PyKEEN for KGE training and evaluation.

Expected outcome: High performance on the most recent task but severe
degradation on older tasks.

Usage:
    from src.baselines.naive_sequential import NaiveSequentialTrainer
    trainer = NaiveSequentialTrainer(model_name='TransE')
    results_matrix = trainer.train(task_sequence, entity_to_id, relation_to_id)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch

from src.baselines._base import (
    _log_mem,
    create_model,
    evaluate_link_prediction,
    get_device,
    make_triples_factory,
    train_epoch,
)

logger = logging.getLogger(__name__)


class NaiveSequentialTrainer:
    """Train KGE model sequentially without any CL mechanism.

    Args:
        model_name: KGE model type - 'TransE', 'ComplEx', 'DistMult', or 'RotatE'.
        embedding_dim: Dimension of entity/relation embeddings.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        batch_size: Training batch size.
        device: 'cuda', 'mps', 'cpu', or 'auto'.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device(device)
        self.seed = seed
        self.model = None

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Train sequentially on all tasks and build results matrix.

        Args:
            task_sequence: OrderedDict of {task_name: {'train': int64_array, ...}}.
            entity_to_id: Global entity → int mapping.
            relation_to_id: Global relation → int mapping.

        Returns:
            results_matrix: R[i][j] = MRR on task j's test set
                after training on task i. Shape (n_tasks, n_tasks).
        """
        task_names = list(task_sequence.keys())
        n_tasks = len(task_names)

        # Create TriplesFactories from pre-mapped int arrays
        _log_mem("before creating TriplesFactories")
        task_factories = {}
        for name, data in task_sequence.items():
            task_factories[name] = {
                split: make_triples_factory(arr, entity_to_id, relation_to_id)
                for split, arr in data.items()
                if len(arr) > 0
            }
            _log_mem(f"after TriplesFactory for {name}")

        # Initialize model
        first_tf = task_factories[task_names[0]]["train"]
        self.model = create_model(
            self.model_name, first_tf,
            embedding_dim=self.embedding_dim,
            random_seed=self.seed,
        )
        self.model = self.model.to(self.device)

        # Results matrix
        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== Training on task {i}: {task_name} ===")

            train_tf = task_factories[task_name]["train"]
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            # Train
            for epoch in range(self.num_epochs):
                loss = train_epoch(
                    self.model, train_tf, optimizer,
                    device=self.device, batch_size=self.batch_size,
                )
                if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                    logger.info(f"  Epoch {epoch + 1}/{self.num_epochs}, loss={loss:.4f}")

            # Build filter triples: all known triples from tasks seen so far
            all_known = torch.cat([
                torch.cat([
                    task_factories[task_names[k]][split].mapped_triples
                    for split in ("train", "val", "test")
                    if split in task_factories[task_names[k]]
                ])
                for k in range(i + 1)
            ])

            # Evaluate on all tasks seen so far
            for j in range(i + 1):
                test_name = task_names[j]
                test_tf = task_factories[test_name]["test"]
                metrics = evaluate_link_prediction(
                    self.model, test_tf,
                    device=self.device, batch_size=self.batch_size,
                    all_known_mapped_triples=all_known,
                )
                R[i, j] = metrics["MRR"]
                logger.info(f"  Eval {test_name}: MRR={metrics['MRR']:.4f}, "
                           f"H@10={metrics['Hits@10']:.4f}")

        return R
