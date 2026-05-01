"""Baseline 4: Experience Replay for continual KGE (BER-style).

Maintains a memory buffer of representative triples from previous tasks.
During training on new tasks, replays old triples to prevent forgetting.
Supports multiple exemplar selection strategies.

Usage:
    from src.baselines.experience_replay import ReplayTrainer
    trainer = ReplayTrainer(model_name='TransE', buffer_size_per_task=500)
    results_matrix = trainer.train(task_sequence, entity_to_id, relation_to_id)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch

from src.baselines._base import (
    create_model,
    evaluate_link_prediction,
    get_device,
    make_triples_factory,
    train_epoch,
)

logger = logging.getLogger(__name__)


class ExperienceReplayKGE:
    """Experience replay buffer for continual KGE.

    Stores representative triples from completed tasks and provides
    them for mixed training during subsequent tasks.

    Works with int64 mapped triples (column 1 = relation ID).

    Args:
        buffer_size_per_task: Max exemplars to store per task.
        selection_strategy: How to select exemplars - 'random' or
            'relation_balanced'.
    """

    def __init__(
        self,
        buffer_size_per_task: int = 500,
        selection_strategy: str = "relation_balanced",
    ) -> None:
        self.buffer_size_per_task = buffer_size_per_task
        self.selection_strategy = selection_strategy
        self.buffer: list[np.ndarray] = []  # List of int64 triple arrays per task

    def select_exemplars(
        self,
        triples: np.ndarray,
        task_id: int,
    ) -> np.ndarray:
        """Select representative triples for the memory buffer.

        Args:
            triples: int64 array of shape (n, 3) with [head_id, rel_id, tail_id].
            task_id: Identifier for the current task.

        Returns:
            Selected triples array of shape (<=buffer_size, 3).
        """
        n = len(triples)
        k = min(self.buffer_size_per_task, n)

        if self.selection_strategy == "random":
            indices = np.random.choice(n, k, replace=False)
            return triples[indices]

        elif self.selection_strategy == "relation_balanced":
            # Equal samples per relation type
            relations = triples[:, 1]
            unique_rels = np.unique(relations)
            per_rel = max(1, k // len(unique_rels))

            selected = []
            for rel in unique_rels:
                rel_mask = relations == rel
                rel_triples = triples[rel_mask]
                n_sel = min(per_rel, len(rel_triples))
                indices = np.random.choice(len(rel_triples), n_sel, replace=False)
                selected.append(rel_triples[indices])

            result = np.concatenate(selected, axis=0)
            # If we have too many, subsample
            if len(result) > k:
                indices = np.random.choice(len(result), k, replace=False)
                result = result[indices]
            return result

        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")

    def add_task(self, triples: np.ndarray, task_id: int) -> None:
        """Add exemplars from a completed task to the buffer."""
        exemplars = self.select_exemplars(triples, task_id)
        self.buffer.append(exemplars)
        total = sum(len(b) for b in self.buffer)
        logger.info(f"  Buffer: added {len(exemplars)} exemplars for task {task_id} "
                    f"(total: {total})")

    def get_replay_triples(self) -> np.ndarray | None:
        """Get all triples in the replay buffer.

        Returns:
            Concatenated int64 array of all buffered triples, or None if empty.
        """
        if not self.buffer:
            return None
        return np.concatenate(self.buffer, axis=0)


class ReplayTrainer:
    """Train KGE sequentially with experience replay.

    Args:
        model_name: KGE model type.
        embedding_dim: Embedding dimension.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        buffer_size_per_task: Exemplars to store per task.
        selection_strategy: 'random' or 'relation_balanced'.
        replay_ratio: Fraction of each batch from replay buffer.
        batch_size: Training batch size.
        device: Device string.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        lr: float = 0.001,
        buffer_size_per_task: int = 500,
        selection_strategy: str = "relation_balanced",
        replay_ratio: float = 0.3,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.buffer_size_per_task = buffer_size_per_task
        self.selection_strategy = selection_strategy
        self.replay_ratio = replay_ratio
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
        """Train sequentially with replay and build results matrix.

        Args:
            task_sequence: OrderedDict of {task_name: {'train': int64_array, ...}}.
            entity_to_id: Global entity → int mapping.
            relation_to_id: Global relation → int mapping.

        Returns:
            results_matrix: R[i][j] = MRR on task j's test set
                after training on task i.
        """
        task_names = list(task_sequence.keys())
        n_tasks = len(task_names)

        np.random.seed(self.seed)

        task_factories = {}
        for name, data in task_sequence.items():
            task_factories[name] = {
                split: make_triples_factory(arr, entity_to_id, relation_to_id)
                for split, arr in data.items()
                if len(arr) > 0
            }

        # Initialize model
        first_tf = task_factories[task_names[0]]["train"]
        model = create_model(
            self.model_name, first_tf,
            embedding_dim=self.embedding_dim,
            random_seed=self.seed,
        )
        model = model.to(self.device)
        self.model = model

        replay = ExperienceReplayKGE(
            buffer_size_per_task=self.buffer_size_per_task,
            selection_strategy=self.selection_strategy,
        )

        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== Replay Training on task {i}: {task_name} "
                       f"(buffer={self.buffer_size_per_task}, "
                       f"strategy={self.selection_strategy}) ===")

            train_data = task_sequence[task_name]["train"]

            # Mix current task with replay buffer
            replay_triples = replay.get_replay_triples()
            if replay_triples is not None:
                # Compute replay batch size
                n_replay = int(len(train_data) * self.replay_ratio)
                n_replay = min(n_replay, len(replay_triples))
                # Sample from replay buffer
                replay_idx = np.random.choice(
                    len(replay_triples), n_replay, replace=True
                )
                replay_sample = replay_triples[replay_idx]
                combined = np.concatenate([train_data, replay_sample], axis=0)
                logger.info(f"  Mixed training: {len(train_data):,} current + "
                           f"{n_replay:,} replay = {len(combined):,}")
            else:
                combined = train_data

            # Create factory for combined data
            train_tf = make_triples_factory(combined, entity_to_id, relation_to_id)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Train
            for epoch in range(self.num_epochs):
                loss = train_epoch(
                    model, train_tf, optimizer,
                    device=self.device, batch_size=self.batch_size,
                )
                if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                    logger.info(f"  Epoch {epoch + 1}/{self.num_epochs}, "
                               f"loss={loss:.4f}")

            # Add current task exemplars to buffer
            replay.add_task(train_data, i)

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
                    model, test_tf,
                    device=self.device, batch_size=self.batch_size,
                    all_known_mapped_triples=all_known,
                )
                R[i, j] = metrics["MRR"]
                logger.info(f"  Eval {test_name}: MRR={metrics['MRR']:.4f}, "
                           f"H@10={metrics['Hits@10']:.4f}")

        return R
