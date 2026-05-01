"""Node classification baseline using frozen KGE embeddings + MLP classifier.

Takes entity embeddings from a trained KGE model (or CMKL), freezes them,
and trains a 2-layer MLP classifier on top for node type prediction.

Usage:
    from src.baselines.nc_baseline import NCBaseline
    nc = NCBaseline(embedding_dim=256, num_classes=10)
    metrics = nc.train_and_evaluate(embeddings, labels, train_mask, val_mask, test_mask)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NCClassifier(nn.Module):
    """2-layer MLP classifier for node classification.

    Args:
        input_dim: Dimension of input embeddings.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, input_dim] node embeddings.

        Returns:
            Logits [N, num_classes].
        """
        return self.net(x)


class NCBaseline:
    """Node classification baseline using frozen embeddings + MLP.

    Extracts entity embeddings from a trained KGE/CMKL model, then
    trains a lightweight MLP classifier for node type prediction.

    Args:
        embedding_dim: Dimension of entity embeddings.
        num_classes: Number of node type classes (default 10 for PrimeKG).
        hidden_dim: MLP hidden dimension (default: same as embedding_dim).
        lr: Learning rate for classifier.
        num_epochs: Training epochs for classifier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int | None = None,
        lr: float = 0.01,
        num_epochs: int = 100,
        dropout: float = 0.3,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.dropout = dropout

    def train_and_evaluate(
        self,
        embeddings: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        train_mask: np.ndarray | torch.Tensor,
        val_mask: np.ndarray | torch.Tensor,
        test_mask: np.ndarray | torch.Tensor,
        device: str = "cpu",
    ) -> dict[str, float]:
        """Train MLP classifier and evaluate on test set.

        Args:
            embeddings: [N, D] frozen entity embeddings.
            labels: [N] integer class labels.
            train_mask: [N] boolean training mask.
            val_mask: [N] boolean validation mask.
            test_mask: [N] boolean test mask.
            device: Device for training.

        Returns:
            Dict with 'accuracy', 'macro_f1', 'weighted_f1'.
        """
        from src.evaluation.metrics import compute_nc_metrics

        # Convert to tensors
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.long)
        if isinstance(train_mask, np.ndarray):
            train_mask = torch.tensor(train_mask, dtype=torch.bool)
        if isinstance(val_mask, np.ndarray):
            val_mask = torch.tensor(val_mask, dtype=torch.bool)
        if isinstance(test_mask, np.ndarray):
            test_mask = torch.tensor(test_mask, dtype=torch.bool)

        embeddings = embeddings.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        # Create classifier
        classifier = NCClassifier(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr)
        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.num_epochs):
            # Train
            classifier.train()
            logits = classifier(embeddings)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validate
            if (epoch + 1) % 10 == 0:
                classifier.eval()
                with torch.no_grad():
                    val_logits = classifier(embeddings)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = (val_pred == labels[val_mask]).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        # Load best model and evaluate on test
        if best_state is not None:
            classifier.load_state_dict(best_state)

        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(embeddings)
            test_pred = test_logits[test_mask].argmax(dim=1).cpu().numpy()
            test_labels = labels[test_mask].cpu().numpy()

        metrics = compute_nc_metrics(test_labels, test_pred)
        logger.info(f"NC test: acc={metrics['accuracy']:.4f}, "
                    f"macro_f1={metrics['macro_f1']:.4f}")
        return metrics

    @staticmethod
    def extract_pykeen_embeddings(
        model: torch.nn.Module,
        entity_to_id: dict[str, int],
    ) -> np.ndarray:
        """Extract entity embeddings from a trained PyKEEN model.

        Args:
            model: Trained PyKEEN KGE model.
            entity_to_id: Entity -> ID mapping.

        Returns:
            [N, D] numpy array of entity embeddings.
        """
        model.eval()
        with torch.no_grad():
            # PyKEEN models store embeddings in entity_representations
            emb = model.entity_representations[0]
            # Get all embeddings
            ids = torch.arange(len(entity_to_id))
            embeddings = emb(ids).cpu().numpy()
        return embeddings
