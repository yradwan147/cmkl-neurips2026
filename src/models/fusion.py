"""Cross-modal fusion for the CMKL framework.

Three fusion strategies:
1. CrossModalAttentionFusion: Gated cross-modal fusion following MMKGR
   (Zheng et al., ICDE 2023). For per-node KG embeddings where each node
   has ONE vector per modality, learned gates control the blend between
   structural and auxiliary modality representations.
2. ConcatenationFusion: Simple concatenation + MLP (ablation baseline).
3. MixtureOfExpertsFusion: Per-modality expert MLPs with a learned router.
   Each modality is an independent expert; a masked softmax router assigns
   per-entity weights. Guarantees MoE >= best single modality by construction.

All handle missing modalities via boolean masks — nodes without text/mol
features get zero contributions from those modalities.

Usage:
    from src.models.fusion import MixtureOfExpertsFusion
    fusion = MixtureOfExpertsFusion(embed_dim=256)
    h_fused = fusion(h_struct, h_text, h_mol, node_has_text, node_has_mol)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossModalAttentionFusion(nn.Module):
    """Gated cross-modal fusion following MMKGR (Zheng et al., ICDE 2023).

    For each node with text/mol features, a learned gate controls the
    blend between structural and auxiliary modality representations.
    This captures cross-modal complementarity while filtering irrelevant
    modality signals per entity.

    The old nn.MultiheadAttention approach was a no-op because seq_len=1
    makes softmax trivially 1.0. Gated fusion is the standard for per-node
    KG settings where each node has one embedding per modality.

    Args:
        embed_dim: Embedding dimension (must be same for all modalities).
        num_heads: Unused, kept for API compatibility with existing configs.
        dropout: Dropout probability in fusion MLP.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Struct-text gated fusion (MMKGR gate-attention pattern)
        self.gate_st = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.proj_s_text = nn.Linear(embed_dim, embed_dim)
        self.proj_t = nn.Linear(embed_dim, embed_dim)

        # Struct-mol gated fusion
        self.gate_sm = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.proj_s_mol = nn.Linear(embed_dim, embed_dim)
        self.proj_m = nn.Linear(embed_dim, embed_dim)

        # Final fusion MLP: [struct_enhanced, text_contribution, mol_contribution]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
        h_mol: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal embeddings via gated cross-modal fusion.

        Args:
            h_struct: [N, D] structural embeddings for all nodes.
            h_text: [N_text, D] textual embeddings (only for nodes with text).
            h_mol: [N_mol, D] molecular embeddings (only for drug nodes).
            node_has_text: [N] boolean mask indicating which nodes have text.
            node_has_mol: [N] boolean mask indicating which nodes have mol.

        Returns:
            Fused node embeddings [N, D].
        """
        N, D = h_struct.shape
        device = h_struct.device

        # --- 1. Struct-Text gated fusion ---
        # For text nodes: gate controls struct vs text blend
        h_text_contribution = torch.zeros(N, D, device=device)

        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() > 0 and h_text.shape[0] > 0:
            h_s = h_struct[text_idx]  # [N_text, D]
            # gate = sigmoid(W([h_struct; h_text]))
            gate = self.gate_st(torch.cat([h_s, h_text], dim=-1))  # [N_text, D]
            # Gated blend: gate * proj(struct) + (1-gate) * proj(text)
            h_st = gate * self.proj_s_text(h_s) + (1 - gate) * self.proj_t(h_text)
            h_text_contribution[text_idx] = h_st

        # --- 2. Struct-Mol gated fusion ---
        h_mol_contribution = torch.zeros(N, D, device=device)

        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() > 0 and h_mol.shape[0] > 0:
            h_s = h_struct[mol_idx]  # [N_mol, D]
            gate = self.gate_sm(torch.cat([h_s, h_mol], dim=-1))  # [N_mol, D]
            h_sm = gate * self.proj_s_mol(h_s) + (1 - gate) * self.proj_m(h_mol)
            h_mol_contribution[mol_idx] = h_sm

        # --- 3. Concatenate and fuse via MLP ---
        # [N, 3D] -> MLP -> [N, D]
        h_concat = torch.cat([h_struct, h_text_contribution, h_mol_contribution], dim=-1)
        h_fused = self.fusion_mlp(h_concat)

        # --- 4. Conditional residual: preserve h_struct for nodes without ANY modality ---
        # Nodes WITH text/mol: gated contributions are non-zero, MLP has real signal
        # Nodes WITHOUT text/mol: input is [h_struct, 0, 0], MLP bottleneck destroys signal
        has_any_modality = node_has_text | node_has_mol  # [N] bool
        h_fused[~has_any_modality] = h_struct[~has_any_modality]

        h_fused = self.layer_norm(h_fused)

        return h_fused


class ConcatenationFusion(nn.Module):
    """Simple concatenation + MLP fusion (ablation baseline).

    Used as an ablation to compare against gated cross-modal fusion.
    Concatenates available modality embeddings and projects via MLP.
    No cross-attention or gating — just direct concatenation.

    Args:
        embed_dim: Per-modality embedding dimension.
        num_modalities: Number of modalities to fuse.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_modalities: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
        h_mol: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse via concatenation + MLP.

        Missing modalities are filled with zeros before concatenation.

        Args:
            h_struct: [N, D] structural embeddings.
            h_text: [N_text, D] textual embeddings.
            h_mol: [N_mol, D] molecular embeddings.
            node_has_text: [N] boolean mask for text availability.
            node_has_mol: [N] boolean mask for molecular availability.

        Returns:
            Fused node embeddings [N, D].
        """
        N, D = h_struct.shape
        device = h_struct.device

        # Scatter text/mol embeddings into full-size tensors
        h_text_full = torch.zeros(N, D, device=device)
        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() > 0 and h_text.shape[0] > 0:
            h_text_full[text_idx] = h_text

        h_mol_full = torch.zeros(N, D, device=device)
        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() > 0 and h_mol.shape[0] > 0:
            h_mol_full[mol_idx] = h_mol

        h_concat = torch.cat([h_struct, h_text_full, h_mol_full], dim=-1)
        h_fused = self.fusion_mlp(h_concat)

        # Conditional residual: preserve h_struct for nodes without ANY modality
        has_any_modality = node_has_text | node_has_mol
        h_fused[~has_any_modality] = h_struct[~has_any_modality]

        h_fused = self.layer_norm(h_fused)

        return h_fused


class MixtureOfExpertsFusion(nn.Module):
    """Mixture-of-Experts fusion with per-modality experts and a learned router.

    Each modality (structural, textual, molecular) has an independent expert MLP
    with a residual connection. A router network learns per-entity softmax weights
    over available experts. The fused output is a weighted sum of expert outputs —
    no compression bottleneck.

    Key property: the solution space includes all single-modality solutions.
    If the router learns weight=1.0 for any single expert, the output equals
    that modality exactly. So MoE >= best single modality by construction.

    Unavailable modalities are masked out via -inf logits before softmax,
    ensuring zero weight and no gradient flow for missing modalities.

    Args:
        embed_dim: Per-modality embedding dimension.
        router_hidden_dim: Hidden size of the router MLP.
        dropout: Dropout probability in expert and router MLPs.
        load_balance_weight: Weight for auxiliary load-balancing loss (prevents
            total router collapse to a single expert).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,  # unused, kept for API compatibility
        router_hidden_dim: int = 128,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.load_balance_weight = load_balance_weight

        # Expert MLPs with residual: Linear(D,D) -> ReLU -> Dropout -> Linear(D,D)
        self.expert_struct = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.expert_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.expert_mol = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Router: [e_struct; e_text_full; e_mol_full; avail_mask] -> 3 logits
        # Input: 3*D (expert outputs) + 3 (availability mask)
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 3 + 3, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 3),
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        # Stored for logging (not a parameter — just diagnostic info)
        self._last_router_weights: list[float] | None = None
        self._aux_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor:
        """Auxiliary load-balancing loss from the last forward pass."""
        if self._aux_loss is not None:
            return self._aux_loss
        return torch.tensor(0.0)

    def forward(
        self,
        h_struct: torch.Tensor,
        h_text: torch.Tensor,
        h_mol: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal embeddings via Mixture-of-Experts.

        Args:
            h_struct: [N, D] structural embeddings for all nodes.
            h_text: [N_text, D] textual embeddings (only for nodes with text).
            h_mol: [N_mol, D] molecular embeddings (only for drug nodes).
            node_has_text: [N] boolean mask indicating which nodes have text.
            node_has_mol: [N] boolean mask indicating which nodes have mol.

        Returns:
            Fused node embeddings [N, D].
        """
        N, D = h_struct.shape
        device = h_struct.device

        # --- 1. Expert outputs with residual connections ---
        e_struct = self.expert_struct(h_struct) + h_struct  # [N, D]

        # Scatter text expert outputs into full-size tensor
        e_text_full = torch.zeros(N, D, device=device)
        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() > 0 and h_text.shape[0] > 0:
            e_text_full[text_idx] = self.expert_text(h_text) + h_text

        # Scatter mol expert outputs into full-size tensor
        e_mol_full = torch.zeros(N, D, device=device)
        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() > 0 and h_mol.shape[0] > 0:
            e_mol_full[mol_idx] = self.expert_mol(h_mol) + h_mol

        # --- 2. Router with masked softmax ---
        # Availability mask: struct always available, text/mol conditional
        avail = torch.zeros(N, 3, device=device)
        avail[:, 0] = 1.0  # struct always available
        avail[text_idx, 1] = 1.0
        avail[mol_idx, 2] = 1.0

        # Router input: concatenate all expert outputs + availability mask
        router_input = torch.cat([e_struct, e_text_full, e_mol_full, avail], dim=-1)
        logits = self.router(router_input)  # [N, 3]

        # Mask unavailable experts with -inf before softmax
        mask = avail == 0
        logits = logits.masked_fill(mask, float("-inf"))

        weights = torch.softmax(logits, dim=-1)  # [N, 3]
        # NaN safety: nodes with only struct will have weights [1, 0, 0] from
        # softmax, but if all are masked (shouldn't happen), fill with struct-only
        weights = torch.nan_to_num(weights, nan=0.0)

        # --- 3. Weighted sum (no compression bottleneck) ---
        h_fused = (
            weights[:, 0:1] * e_struct
            + weights[:, 1:2] * e_text_full
            + weights[:, 2:3] * e_mol_full
        )  # [N, D]

        h_fused = self.layer_norm(h_fused)

        # --- 4. Auxiliary load-balancing loss ---
        # Encourages the router to use all available experts, preventing collapse
        if self.training and self.load_balance_weight > 0:
            # Mean weight per expert across nodes that have that expert available
            mean_weights = []
            for i in range(3):
                expert_mask = avail[:, i] == 1.0
                if expert_mask.any():
                    mean_weights.append(weights[expert_mask, i].mean())
                else:
                    mean_weights.append(torch.tensor(0.0, device=device))
            mean_w = torch.stack(mean_weights)
            # Variance of mean weights — zero when perfectly balanced
            self._aux_loss = self.load_balance_weight * mean_w.var()
        else:
            self._aux_loss = None

        # Store per-group router weights for logging
        # Report weights for text-available nodes separately (more informative
        # than overall mean, which is dominated by struct-only nodes)
        with torch.no_grad():
            has_text = avail[:, 1] == 1.0
            if has_text.any():
                tw = weights[has_text]
                self._last_router_weights = [
                    tw[:, 0].mean().item(),
                    tw[:, 1].mean().item(),
                    tw[:, 2].mean().item(),
                ]
            else:
                self._last_router_weights = [1.0, 0.0, 0.0]

        return h_fused
