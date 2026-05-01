"""Modality-specific encoders for the CMKL framework.

Three encoder types for different modalities in biomedical KGs:
1. StructuralEncoder: R-GCN for graph structure (all nodes)
2. TextualEncoder: Frozen BiomedBERT for text descriptions (drug/disease nodes)
3. MolecularEncoder: MLP for Morgan fingerprints (drug nodes only)

Not all nodes have all modalities - encoders must handle missing modalities
gracefully via zero vectors or learned default embeddings.

Usage:
    from src.models.encoders import StructuralEncoder, TextualEncoder, MolecularEncoder
    struct_enc = StructuralEncoder(num_nodes=129375, num_relations=30)
    text_enc = TextualEncoder()
    mol_enc = MolecularEncoder()
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StructuralEncoder(nn.Module):
    """R-GCN encoder for graph structure.

    Uses RGCNConv layers from torch_geometric.nn with LayerNorm and ReLU.
    If torch_geometric is not available, falls back to a simple embedding
    lookup (no message passing).

    When `out_dim` differs from `embedding_dim`, R-GCN runs internally at
    `embedding_dim` and a final Linear projects to `out_dim`. This keeps
    per-edge message-passing memory bound to `embedding_dim` — critical
    for RotatE, which wants out_dim = 2*embedding_dim: running R-GCN at
    the doubled dim OOMs V100 on PrimeKG-scale graphs (~47 GB intermediate).
    Keeping the projection *inside* the encoder means MA-EWC's per-modality
    Fisher (which walks `structural_encoder.named_parameters()`) protects
    it like any other R-GCN parameter.

    Args:
        num_nodes: Total number of nodes in the KG.
        num_relations: Total number of relation types.
        embedding_dim: Internal R-GCN dim.
        out_dim: Output dim. Defaults to `embedding_dim` (no projection).
        num_layers: Number of R-GCN layers.
        num_bases: Number of basis decomposition matrices for R-GCN.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 256,
        out_dim: int | None = None,
        num_layers: int = 2,
        num_bases: int = 30,
        preserve_complex_geom: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim if out_dim is not None else embedding_dim
        # For rotational/complex decoders (RotatE, ComplEx) the entity dim is
        # reinterpreted as (real, imag) pairs. LayerNorm + ReLU destroy that
        # geometry: LN rescales magnitudes per-sample (breaks |h|=|t|), ReLU
        # restricts every component to >=0 which collapses the full complex
        # plane to the first quadrant (kills rotation). For those decoders we
        # run a *linear* residual R-GCN (no LN, no ReLU) — structural message
        # passing still happens, but the complex arithmetic needed by the
        # decoder is preserved downstream.
        self.preserve_complex_geom = preserve_complex_geom

        # Node embeddings (learnable)
        self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # R-GCN layers
        try:
            from torch_geometric.nn import RGCNConv
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(RGCNConv(
                    embedding_dim, embedding_dim,
                    num_relations=num_relations * 2,  # forward + reverse
                    num_bases=min(num_bases, num_relations * 2),
                ))
                self.norms.append(nn.LayerNorm(embedding_dim))
            self.has_gnn = True
        except ImportError:
            logger.warning("torch_geometric not available, using embedding-only mode")
            self.has_gnn = False

        # Output projection (absent when out_dim == embedding_dim, for
        # bit-identical behavior to the pre-RotatE code path)
        if self.out_dim != embedding_dim:
            self.out_proj = nn.Linear(embedding_dim, self.out_dim)
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            self.out_proj = None

    def forward(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through R-GCN layers.

        Args:
            edge_index: Edge indices [2, num_edges]. If None, returns raw embeddings.
            edge_type: Edge type labels [num_edges].

        Returns:
            Node embeddings [num_nodes, out_dim].
        """
        x = self.node_emb.weight

        if self.has_gnn and edge_index is not None and edge_type is not None:
            if self.preserve_complex_geom:
                # Linear residual R-GCN: structural message passing only,
                # no LayerNorm/ReLU (which break rotational/complex geometry).
                for conv in self.convs:
                    x = x + conv(x, edge_index, edge_type)
            else:
                for conv, norm in zip(self.convs, self.norms):
                    x = x + conv(x, edge_index, edge_type)  # c-RGCN residual
                    x = norm(x)
                    x = torch.relu(x)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x


class TextualEncoder(nn.Module):
    """Frozen BiomedBERT encoder for textual node features.

    Uses a pretrained language model with frozen weights. A linear projection
    maps from the LM hidden size (768) to projection_dim.

    In practice, text embeddings are pre-computed and cached to avoid
    re-encoding during training. This module handles both on-the-fly
    encoding and cached embedding lookup.

    Optionally includes a LoRA-style low-rank adapter before the projection,
    enabling the text pathway to adapt to the margin-ranking training objective
    while keeping the core BiomedBERT representations frozen.

    Args:
        model_name: HuggingFace model identifier.
        projection_dim: Output embedding dimension.
        hidden_size: BiomedBERT hidden dimension.
        lora_rank: Low-rank adapter rank. 0 = no adapter (default).
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        projection_dim: int = 256,
        hidden_size: int = 768,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        self._lm = None
        self._tokenizer = None

        if lora_rank > 0:
            self.lora_down = nn.Linear(hidden_size, lora_rank, bias=False)
            self.lora_up = nn.Linear(lora_rank, hidden_size, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight)
            nn.init.zeros_(self.lora_up.weight)
            logger.info(f"TextualEncoder: LoRA adapter enabled (rank={lora_rank})")

    def _load_lm(self) -> None:
        """Lazy-load the language model (only when needed for encoding)."""
        if self._lm is None:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._lm = AutoModel.from_pretrained(self.model_name)
            # Freeze LM weights
            for param in self._lm.parameters():
                param.requires_grad = False
            self._lm.eval()
            logger.info(f"Loaded {self.model_name} (frozen)")

    @torch.no_grad()
    def encode_texts(self, texts: list[str], device: str = "cpu", batch_size: int = 32) -> torch.Tensor:
        """Encode raw text strings into embeddings.

        Used for pre-computing text embeddings. The projection layer
        IS included (and is the only trainable part).

        Args:
            texts: List of text descriptions.
            device: Device for computation.
            batch_size: Encoding batch size.

        Returns:
            Text embeddings [len(texts), projection_dim].
        """
        self._load_lm()
        self._lm = self._lm.to(device)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = self._tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(device)
            outputs = self._lm(**tokens)
            cls_emb = outputs.last_hidden_state[:, 0]  # [CLS] token
            all_embs.append(cls_emb.cpu())

        return torch.cat(all_embs, dim=0)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project pre-computed text embeddings.

        If LoRA is enabled, applies a low-rank residual adaptation before
        projection: x' = x + lora_up(lora_down(x)). This allows the text
        pathway to learn task-specific adjustments to frozen BiomedBERT
        embeddings without modifying the original representations.

        Args:
            text_embeddings: Pre-computed LM embeddings [batch, hidden_size].

        Returns:
            Projected text embeddings [batch, projection_dim].
        """
        if self.lora_rank > 0:
            text_embeddings = text_embeddings + self.lora_up(self.lora_down(text_embeddings))
        return self.projection(text_embeddings)


class MolecularEncoder(nn.Module):
    """Encoder for molecular fingerprints (drug nodes).

    Two-layer MLP: input_dim (1024) -> hidden_dim (512) -> projection_dim (256).

    Args:
        input_dim: Morgan fingerprint dimensionality.
        hidden_dim: Hidden layer dimension.
        projection_dim: Output embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Encode molecular fingerprints.

        Args:
            fingerprints: Morgan fingerprint vectors [batch, input_dim].

        Returns:
            Molecular embeddings [batch, projection_dim].
        """
        return self.mlp(fingerprints)
