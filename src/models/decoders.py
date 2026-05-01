"""Link prediction decoders for CMKL and baselines.

Score functions for predicting links in knowledge graphs:
- TransE: -||h + r - t|| (translation-based)
- DistMult: <h, r, t> (bilinear diagonal)
- Bilinear: h^T M_r t (full bilinear)

Usage:
    from src.models.decoders import TransEDecoder, DistMultDecoder
    decoder = TransEDecoder(embedding_dim=256)
    scores = decoder(head_embs, rel_embs, tail_embs)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransEDecoder(nn.Module):
    """TransE-style link prediction decoder.

    Score: -||h + r - t||_p (higher is more plausible).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        p_norm: Norm order (1 or 2).
    """

    def __init__(self, embedding_dim: int = 256, p_norm: int = 2) -> None:
        super().__init__()
        self.p_norm = p_norm

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute TransE scores: -||h + r - t||_p."""
        return -torch.norm(head + relation - tail, p=self.p_norm, dim=-1)


class DistMultDecoder(nn.Module):
    """DistMult-style link prediction decoder.

    Score: sum(h * r * t).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute DistMult scores: sum(h * r * t)."""
        return (head * relation * tail).sum(dim=-1)


class ComplExDecoder(nn.Module):
    """ComplEx-style link prediction decoder, faithful to PyKEEN.

    Score: Re(sum(h * r * conj(t))) in complex space. Entities AND relations
    are full complex vectors (unlike RotatE which uses unit-phase relations).

    PyKEEN convention: entities and relations are stored as raw real tensors
    of dim 2*embedding_dim (embedding_dim complex pairs, interleaved real/imag).

    Args:
        embedding_dim: Number of complex pairs. Internal real dim = 2 * embedding_dim.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.real_dim = 2 * embedding_dim

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute ComplEx scores: Re(sum(h * r * conj(t))).

        Args:
            head: [batch, 2*D] real-valued.
            relation: [batch, 2*D] real-valued (full complex relation).
            tail: [batch, 2*D] real-valued.

        Returns:
            Scores [batch] — higher is more plausible.
        """
        B = head.shape[0]
        h_c = torch.view_as_complex(head.reshape(B, self.embedding_dim, 2).contiguous())
        r_c = torch.view_as_complex(relation.reshape(B, self.embedding_dim, 2).contiguous())
        t_c = torch.view_as_complex(tail.reshape(B, self.embedding_dim, 2).contiguous())
        return (h_c * r_c * t_c.conj()).sum(dim=-1).real


class RotatEDecoder(nn.Module):
    """RotatE-style link prediction decoder, faithful to PyKEEN.

    Score: -||h ∘ r - t||_2 in complex space, where r has unit modulus.

    To match PyKEEN's RotatE convention exactly:
    - Entity embeddings have 2*embedding_dim REAL values internally
      (split into embedding_dim complex pairs). The decoder takes the
      RAW embeddings of dim 2*embedding_dim from CMKL's encoder.
    - Relation embeddings are stored as raw phase parameters (theta)
      of dim embedding_dim, converted to unit complex r = exp(i*theta).
    - PyKEEN initializes phases uniformly in [-pi, pi].

    Args:
        embedding_dim: Number of complex pairs (PyKEEN convention).
                       Internal real dim = 2 * embedding_dim.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim  # number of complex pairs
        self.real_dim = 2 * embedding_dim   # real-valued dim of entity embeddings

    def forward(self, head: torch.Tensor, relation_phases: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute RotatE scores: -||h ∘ r - t||_2.

        Args:
            head: [batch, 2*D] real-valued (PyKEEN convention: 2 reals per complex)
            relation_phases: [batch, D] real-valued phases (theta)
            tail: [batch, 2*D] real-valued

        Returns:
            Scores [batch] — higher is more plausible.
        """
        B = head.shape[0]
        # View entity reals as complex
        h_c = torch.view_as_complex(head.reshape(B, self.embedding_dim, 2).contiguous())
        t_c = torch.view_as_complex(tail.reshape(B, self.embedding_dim, 2).contiguous())
        # Convert phases to unit-modulus complex: r = cos(theta) + i*sin(theta)
        r_c = torch.complex(torch.cos(relation_phases), torch.sin(relation_phases))

        diff = h_c * r_c - t_c
        return -diff.abs().pow(2).sum(dim=-1).sqrt()


class BilinearDecoder(nn.Module):
    """Full bilinear link prediction decoder.

    Score: h^T M_r t (with per-relation weight matrix).

    Args:
        embedding_dim: Dimension of entity/relation embeddings.
        num_relations: Number of relation types.
    """

    def __init__(self, embedding_dim: int = 256, num_relations: int = 30) -> None:
        super().__init__()
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, embedding_dim, embedding_dim) * 0.01
        )

    def forward(self, head: torch.Tensor, relation_ids: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Compute bilinear scores: h^T M_r t.

        Args:
            head: [batch, dim].
            relation_ids: [batch] integer relation type indices.
            tail: [batch, dim].

        Returns:
            Score tensor [batch].
        """
        M = self.relation_matrices[relation_ids]  # [batch, dim, dim]
        # h^T M_r t = sum_ij h_i M_ij t_j
        return torch.einsum("bi,bij,bj->b", head, M, tail)
