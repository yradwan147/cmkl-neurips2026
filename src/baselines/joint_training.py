"""Baseline 2: Joint Training (Upper Bound).

Trains on all tasks simultaneously by concatenating all training data.
This is the upper bound for performance since there is no forgetting
by definition. The gap between joint and naive sequential quantifies
how much forgetting matters on this benchmark.

Usage:
    from src.baselines.joint_training import JointTrainer
    trainer = JointTrainer(model_name='TransE')
    per_task_results = trainer.train(task_sequence, entity_to_id, relation_to_id)
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


class JointTrainer:
    """Train KGE model on all tasks simultaneously (upper bound reference).

    Args:
        model_name: KGE model type - 'TransE', 'ComplEx', 'DistMult', or 'RotatE'.
        embedding_dim: Dimension of entity/relation embeddings.
        num_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        device: 'cuda', 'mps', 'cpu', or 'auto'.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 200,
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
    ) -> dict[str, dict]:
        """Train on concatenated data from all tasks.

        Args:
            task_sequence: OrderedDict of {task_name: {'train': int64_array, ...}}.
            entity_to_id: Global entity → int mapping.
            relation_to_id: Global relation → int mapping.

        Returns:
            Dict mapping task_name -> metrics dict for each task's test set.
            Also includes 'results_matrix' key with n_tasks x n_tasks matrix
            (all rows identical since single training run).
        """
        task_names = list(task_sequence.keys())
        n_tasks = len(task_names)

        # Concatenate all training data (int arrays — very cheap)
        _log_mem("joint: before concatenating train data")
        all_train = np.concatenate([
            task_sequence[name]["train"] for name in task_names
            if len(task_sequence[name]["train"]) > 0
        ], axis=0)
        logger.info(f"Joint training data: {len(all_train):,} triples "
                    f"from {n_tasks} tasks")
        _log_mem("joint: after concatenation")

        # Create factories
        train_tf = make_triples_factory(all_train, entity_to_id, relation_to_id)
        del all_train  # free the numpy copy
        _log_mem("joint: after train TriplesFactory")

        test_factories = {}
        for name in task_names:
            test_data = task_sequence[name]["test"]
            if len(test_data) > 0:
                test_factories[name] = make_triples_factory(
                    test_data, entity_to_id, relation_to_id
                )
        _log_mem("joint: after all test TriplesFactories")

        # Create and train model
        model = create_model(
            self.model_name, train_tf,
            embedding_dim=self.embedding_dim,
            random_seed=self.seed,
        )
        model = model.to(self.device)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        logger.info("=== Joint Training ===")
        for epoch in range(self.num_epochs):
            loss = train_epoch(
                model, train_tf, optimizer,
                device=self.device, batch_size=self.batch_size,
            )
            if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.num_epochs}, loss={loss:.4f}")

        # Build filter triples: all known triples from all tasks
        all_known = torch.cat([
            torch.cat([
                tf.mapped_triples for tf in test_factories.values()
            ]),
            train_tf.mapped_triples,
        ])

        # Evaluate on each task's test set
        per_task = {}
        R = np.zeros((n_tasks, n_tasks))

        for j, name in enumerate(task_names):
            if name in test_factories:
                metrics = evaluate_link_prediction(
                    model, test_factories[name],
                    device=self.device, batch_size=self.batch_size,
                    all_known_mapped_triples=all_known,
                )
                per_task[name] = metrics
                logger.info(f"  {name}: MRR={metrics['MRR']:.4f}, "
                           f"H@10={metrics['Hits@10']:.4f}")
                # Fill all rows with same value (no sequential training)
                for i in range(n_tasks):
                    R[i, j] = metrics["MRR"]

        per_task["results_matrix"] = R
        return per_task
