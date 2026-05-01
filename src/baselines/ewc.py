"""Baseline 3: Elastic Weight Consolidation (EWC) for Knowledge Graph Embeddings.

Protects important parameters learned in previous tasks by adding a quadratic
penalty based on the Fisher Information Matrix. After each task, computes the
Fisher diagonal to identify important parameters, then adds a penalty term
to the loss when training on subsequent tasks.

L_total = L_task + (lambda/2) * sum_k F_k * (theta_k - theta*_k)^2

Reference: EWC with lambda=10 reduced forgetting from 12.62% to 6.85% on FB15k-237.

Usage:
    from src.baselines.ewc import EWCTrainer
    trainer = EWCTrainer(model_name='TransE', lambda_ewc=10.0)
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
    _generate_negatives,
    _margin_loss,
    _score_triples,
)

logger = logging.getLogger(__name__)


class EWC_KGE:
    """Elastic Weight Consolidation mechanism for KGE models.

    Computes and stores Fisher Information Matrix diagonals and
    reference parameters after each task. Provides the EWC penalty
    term to add to the base KGE loss.

    Args:
        model: The KGE model (PyTorch nn.Module).
        lambda_ewc: Regularization strength.
    """

    def __init__(self, model: torch.nn.Module, lambda_ewc: float = 10.0) -> None:
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_diags: dict[str, torch.Tensor] = {}
        self.old_params: dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        train_factory: "TriplesFactory",
        device: str = "cpu",
        n_samples: int = 1000,
    ) -> None:
        """Compute diagonal of Fisher Information Matrix after training on a task.

        Uses the empirical Fisher approximation:
        F_k ~ (1/N) * sum_i (dL_i/d_theta_k)^2

        Accumulates Fisher across tasks (sum) and stores current parameters
        as reference for the EWC penalty.

        Args:
            train_factory: TriplesFactory for the task's training data.
            device: Device for computation.
            n_samples: Number of samples for Fisher estimation.
        """
        self.model.eval()
        self.model.to(device)

        # Initialize new Fisher diagonals
        new_fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        mapped_cpu = train_factory.mapped_triples
        n = min(n_samples, mapped_cpu.shape[0])
        indices = torch.randperm(mapped_cpu.shape[0])[:n]
        samples = mapped_cpu[indices]

        from pykeen.models import TransE as TransEModel
        is_transe = isinstance(self.model, TransEModel)

        for i in range(n):
            self.model.zero_grad()
            triple = samples[i:i + 1].to(device)
            neg = _generate_negatives(triple, self.model.num_entities, device)
            pos_score = _score_triples(self.model, triple, is_transe)
            neg_score = _score_triples(self.model, neg, is_transe)
            loss = _margin_loss(pos_score, neg_score)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += param.grad.data ** 2

        # Normalize
        for name in new_fisher:
            new_fisher[name] /= n

        # Accumulate Fisher across tasks
        for name in new_fisher:
            if name in self.fisher_diags:
                self.fisher_diags[name] = self.fisher_diags[name] + new_fisher[name]
            else:
                self.fisher_diags[name] = new_fisher[name].clone()

        # Store current parameters as reference
        self.old_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def ewc_loss(self) -> torch.Tensor:
        """Compute the EWC penalty term.

        L_ewc = (lambda/2) * sum_k F_k * (theta_k - theta*_k)^2

        Returns:
            Scalar tensor with EWC penalty (0 if no previous tasks).
        """
        if not self.fisher_diags:
            return torch.tensor(0.0, requires_grad=True)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_diags and name in self.old_params:
                fisher = self.fisher_diags[name]
                old_param = self.old_params[name]
                loss = loss + (fisher * (param - old_param) ** 2).sum()

        return (self.lambda_ewc / 2.0) * loss


class EWCTrainer:
    """Train KGE sequentially with EWC regularization.

    Args:
        model_name: KGE model type.
        embedding_dim: Embedding dimension.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        lambda_ewc: EWC regularization strength.
        batch_size: Training batch size.
        fisher_samples: Number of samples for Fisher estimation.
        device: Device string.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        lr: float = 0.001,
        lambda_ewc: float = 10.0,
        batch_size: int = 256,
        fisher_samples: int = 1000,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_ewc = lambda_ewc
        self.batch_size = batch_size
        self.fisher_samples = fisher_samples
        self.device = get_device(device)
        self.seed = seed
        self.model = None

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Train sequentially with EWC and build results matrix.

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

        ewc = EWC_KGE(model, lambda_ewc=self.lambda_ewc)
        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== EWC Training on task {i}: {task_name} "
                       f"(lambda={self.lambda_ewc}) ===")

            train_tf = task_factories[task_name]["train"]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Train with EWC penalty
            from pykeen.models import TransE as TransEModel
            is_transe = isinstance(model, TransEModel)
            model.train()
            mapped_cpu = train_tf.mapped_triples

            for epoch in range(self.num_epochs):
                perm = torch.randperm(mapped_cpu.shape[0])
                shuffled = mapped_cpu[perm]
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, shuffled.shape[0], self.batch_size):
                    batch = shuffled[start:start + self.batch_size].to(self.device)
                    neg = _generate_negatives(
                        batch, model.num_entities, self.device
                    )
                    pos_scores = _score_triples(model, batch, is_transe)
                    neg_scores = _score_triples(model, neg, is_transe)
                    base_loss = _margin_loss(pos_scores, neg_scores)

                    # Add EWC penalty
                    total_loss = base_loss + ewc.ewc_loss()

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    epoch_loss += total_loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                    logger.info(f"  Epoch {epoch + 1}/{self.num_epochs}, "
                               f"loss={avg_loss:.4f}")

            # Compute Fisher for this task
            ewc.compute_fisher(
                train_tf, device=self.device,
                n_samples=self.fisher_samples,
            )
            logger.info(f"  Fisher computed ({self.fisher_samples} samples)")

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
