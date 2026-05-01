"""Multimodal memory replay buffer for CMKL.

Stores triples with their multimodal features (structural, textual, molecular
embeddings). Uses K-means clustering on structural embeddings for diverse
exemplar selection, ensuring replay covers different graph regions.

Three replay strategies:
1. Full multimodal: Store complete multimodal features for each triple
2. Partial modality: Store different modality subsets for different triples
3. Cross-modal reconstruction: Store one modality, reconstruct others

Usage:
    from src.continual.multimodal_replay import MultimodalMemoryBuffer
    buffer = MultimodalMemoryBuffer(max_size=1000)
    buffer.add_exemplars(triples, struct_embs, text_embs, mol_embs, masks)
    replay_batch = buffer.sample(batch_size=64)
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MultimodalMemoryBuffer:
    """Memory buffer that stores triples with their multimodal features.

    Each exemplar is a dict with:
        - triple: (head_id, relation_id, tail_id)
        - struct_emb_head: structural embedding of head entity
        - struct_emb_tail: structural embedding of tail entity
        - text_emb_head: text embedding of head (or None)
        - text_emb_tail: text embedding of tail (or None)
        - mol_emb_head: molecular embedding of head (or None)
        - mol_emb_tail: molecular embedding of tail (or None)
        - task_id: which task this exemplar came from

    Args:
        max_size: Maximum number of exemplars in the buffer.
            Sweep: [100, 250, 500, 1000, 2000, 5000].
        strategy: Replay strategy - 'full_multimodal', 'partial_modality',
            or 'cross_modal_reconstruction'.
    """

    def __init__(
        self,
        max_size: int = 1000,
        strategy: str = "full_multimodal",
    ) -> None:
        self.max_size = max_size
        self.strategy = strategy
        self.buffer: list[dict] = []

    def add_exemplars(
        self,
        triples: np.ndarray,
        struct_embs: torch.Tensor,
        text_embs: torch.Tensor | None,
        mol_embs: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        task_id: int = 0,
        entity_to_id: dict | None = None,
    ) -> None:
        """Add representative samples to the buffer after training on a task.

        Stores each triple with all available modality embeddings for its
        head and tail entities. When buffer exceeds max_size, uses K-means
        clustering on structural embeddings to keep the most diverse samples.

        Args:
            triples: Array of (head, relation, tail) triples, shape [num_triples, 3].
                Values are entity/relation IDs (integers indexing into embeddings).
            struct_embs: [num_entities, D] structural embeddings for all entities.
            text_embs: [num_entities, D] text embeddings (zeros for nodes without text).
                Can be None if no text features available.
            mol_embs: [num_entities, D] molecular embeddings (zeros for non-drug nodes).
                Can be None if no molecular features available.
            node_has_text: [num_entities] boolean mask for text availability.
            node_has_mol: [num_entities] boolean mask for molecular availability.
            task_id: ID of the current task (for provenance tracking).
            entity_to_id: Optional entity-to-id mapping for debugging.
        """
        new_exemplars = []
        struct_embs_cpu = struct_embs.detach().cpu()

        for i in range(len(triples)):
            h, r, t = int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2])

            exemplar = {
                "triple": (h, r, t),
                "task_id": task_id,
                "struct_emb_head": struct_embs_cpu[h].clone(),
                "struct_emb_tail": struct_embs_cpu[t].clone(),
                "text_emb_head": None,
                "text_emb_tail": None,
                "mol_emb_head": None,
                "mol_emb_tail": None,
            }

            # Add text embeddings if available
            if text_embs is not None and node_has_text is not None:
                text_embs_cpu = text_embs.detach().cpu()
                if h < len(node_has_text) and node_has_text[h]:
                    exemplar["text_emb_head"] = text_embs_cpu[h].clone()
                if t < len(node_has_text) and node_has_text[t]:
                    exemplar["text_emb_tail"] = text_embs_cpu[t].clone()

            # Add molecular embeddings if available
            if mol_embs is not None and node_has_mol is not None:
                mol_embs_cpu = mol_embs.detach().cpu()
                if h < len(node_has_mol) and node_has_mol[h]:
                    exemplar["mol_emb_head"] = mol_embs_cpu[h].clone()
                if t < len(node_has_mol) and node_has_mol[t]:
                    exemplar["mol_emb_tail"] = mol_embs_cpu[t].clone()

            new_exemplars.append(exemplar)

        self.buffer.extend(new_exemplars)
        logger.info(
            "Added %d exemplars from task %d (buffer: %d / %d)",
            len(new_exemplars), task_id, len(self.buffer), self.max_size,
        )

        # If buffer exceeds max size, select diverse exemplars
        if len(self.buffer) > self.max_size:
            self._diverse_selection()

    def _diverse_selection(self) -> None:
        """Select diverse exemplars using K-means clustering on structural embeddings.

        Clusters all buffer items by the mean of their head/tail structural
        embeddings, then keeps the exemplar closest to each cluster center
        to maximize diversity.
        """
        if len(self.buffer) <= self.max_size:
            return

        # Build feature matrix from structural embeddings (mean of head + tail)
        features = []
        for ex in self.buffer:
            feat = (ex["struct_emb_head"] + ex["struct_emb_tail"]) / 2.0
            features.append(feat)
        feature_matrix = torch.stack(features)  # [buffer_size, D]

        try:
            from sklearn.cluster import KMeans
            X = feature_matrix.numpy()
            n_clusters = self.max_size
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=50, random_state=42)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_  # [n_clusters, D]

            # For each cluster, keep the exemplar closest to the center
            selected_indices = []
            for c in range(n_clusters):
                cluster_mask = labels == c
                cluster_indices = np.where(cluster_mask)[0]
                if len(cluster_indices) == 0:
                    continue
                # Distance from each cluster member to its center
                cluster_features = X[cluster_indices]
                dists = np.linalg.norm(cluster_features - centers[c], axis=1)
                best_local = np.argmin(dists)
                selected_indices.append(cluster_indices[best_local])

            self.buffer = [self.buffer[i] for i in selected_indices]

        except ImportError:
            # Fallback: random selection if sklearn not available
            logger.warning("sklearn not available, using random selection for replay buffer")
            random.shuffle(self.buffer)
            self.buffer = self.buffer[:self.max_size]

        logger.info("Diverse selection: buffer reduced to %d exemplars", len(self.buffer))

    def sample(self, batch_size: int) -> list[dict]:
        """Sample a batch from the replay buffer.

        Args:
            batch_size: Number of exemplars to sample.

        Returns:
            List of exemplar dicts with triple and embedding data.
        """
        if len(self.buffer) == 0:
            return []
        actual_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_size)

    def get_replay_triples(self, batch_size: int) -> np.ndarray | None:
        """Get just the triples from a replay sample (for simple replay training).

        Args:
            batch_size: Number of triples to sample.

        Returns:
            Array of shape [batch_size, 3] with (head, relation, tail) triples,
            or None if buffer is empty.
        """
        samples = self.sample(batch_size)
        if not samples:
            return None
        triples = np.array([s["triple"] for s in samples])
        return triples

    def get_replay_batch(self, batch_size: int, device: str = "cpu") -> dict | None:
        """Get a full multimodal replay batch with embeddings.

        Returns a dict with batched tensors ready for model consumption:
        - triples: [B, 3]
        - struct_emb_head/tail: [B, D]
        - text_emb_head/tail: [B, D] (zeros where unavailable)
        - mol_emb_head/tail: [B, D] (zeros where unavailable)
        - has_text_head/tail: [B] boolean
        - has_mol_head/tail: [B] boolean

        Args:
            batch_size: Number of exemplars.
            device: Target device.

        Returns:
            Dict with batched tensors, or None if buffer is empty.
        """
        samples = self.sample(batch_size)
        if not samples:
            return None

        D = samples[0]["struct_emb_head"].shape[0]

        triples = torch.tensor([s["triple"] for s in samples], dtype=torch.long)
        struct_h = torch.stack([s["struct_emb_head"] for s in samples])
        struct_t = torch.stack([s["struct_emb_tail"] for s in samples])

        text_h = torch.stack([
            s["text_emb_head"] if s["text_emb_head"] is not None
            else torch.zeros(D)
            for s in samples
        ])
        text_t = torch.stack([
            s["text_emb_tail"] if s["text_emb_tail"] is not None
            else torch.zeros(D)
            for s in samples
        ])
        mol_h = torch.stack([
            s["mol_emb_head"] if s["mol_emb_head"] is not None
            else torch.zeros(D)
            for s in samples
        ])
        mol_t = torch.stack([
            s["mol_emb_tail"] if s["mol_emb_tail"] is not None
            else torch.zeros(D)
            for s in samples
        ])

        has_text_h = torch.tensor([s["text_emb_head"] is not None for s in samples])
        has_text_t = torch.tensor([s["text_emb_tail"] is not None for s in samples])
        has_mol_h = torch.tensor([s["mol_emb_head"] is not None for s in samples])
        has_mol_t = torch.tensor([s["mol_emb_tail"] is not None for s in samples])

        return {
            "triples": triples.to(device),
            "struct_emb_head": struct_h.to(device),
            "struct_emb_tail": struct_t.to(device),
            "text_emb_head": text_h.to(device),
            "text_emb_tail": text_t.to(device),
            "mol_emb_head": mol_h.to(device),
            "mol_emb_tail": mol_t.to(device),
            "has_text_head": has_text_h.to(device),
            "has_text_tail": has_text_t.to(device),
            "has_mol_head": has_mol_h.to(device),
            "has_mol_tail": has_mol_t.to(device),
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> dict:
        """Serialize buffer for checkpointing."""
        return {
            "buffer": self.buffer,
            "max_size": self.max_size,
            "strategy": self.strategy,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load buffer from checkpoint."""
        self.buffer = state["buffer"]
        self.max_size = state["max_size"]
        self.strategy = state["strategy"]
