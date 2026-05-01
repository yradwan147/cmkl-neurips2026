"""CMKL: Continual Multimodal Knowledge Graph Learner.

Assembles the full CMKL model from its 4 key components:
1. Modality-specific encoders (structural, textual, molecular)
2. Cross-modal attention fusion
3. Modality-aware EWC (continual learning regularization)
4. Multimodal memory replay

The core contribution: modality-aware continual learning that leverages
multimodal complementarity to reduce forgetting while handling heterogeneous
distribution shifts across modalities.

Training pipeline per task:
1. Encode: structural, textual, molecular
2. Fuse: cross-modal attention
3. Train: task loss + EWC penalty + replay loss
4. After training: compute Fisher per modality, add to replay buffer
5. Evaluate: on all tasks seen so far

Usage:
    from src.models.cmkl import CMKL
    model = CMKL(config)
    results = model.train_continually(task_sequence)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from src.models.encoders import StructuralEncoder, TextualEncoder, MolecularEncoder
from src.models.fusion import CrossModalAttentionFusion, ConcatenationFusion, MixtureOfExpertsFusion
from src.models.decoders import TransEDecoder, DistMultDecoder, BilinearDecoder, RotatEDecoder, ComplExDecoder
from src.models.ogm_ge import OGMGEModulator
from src.continual.modality_ewc import ModalityAwareEWC
from src.continual.multimodal_replay import MultimodalMemoryBuffer
from src.continual.distillation import KnowledgeDistillation
from src.baselines._base import (
    load_task_sequence,
    make_triples_factory,
    evaluate_link_prediction,
    get_device,
)

logger = logging.getLogger(__name__)

DECODER_REGISTRY = {
    "TransE": TransEDecoder,
    "RotatE": RotatEDecoder,
    "ComplEx": ComplExDecoder,
    "DistMult": DistMultDecoder,
    "Bilinear": BilinearDecoder,
}

DEFAULT_CONFIG = {
    "embedding_dim": 256,
    "num_gnn_layers": 2,
    "num_gnn_bases": 30,
    "num_attention_heads": 4,
    "fusion_type": "cross_attention",  # "cross_attention", "concatenation", or "moe"
    "decoder_type": "DistMult",  # TransE, DistMult, or Bilinear
    "router_hidden_dim": 128,
    "load_balance_weight": 0.01,
    "lambda_fusion": 5.0,
    "lambda_relation": 50.0,        # Fisher lambda for main relation_emb — high because
                                    # rotational decoders (RotatE) treat these as rotation
                                    # angles where any drift warps the whole geometry
    "lambda_struct": 10.0,
    "lambda_text": 5.0,
    "lambda_mol": 1.0,
    "text_lora_rank": 0,  # LoRA adapter rank for text encoder (0 = disabled)
    "replay_buffer_size": 1000,
    "replay_strategy": "full_multimodal",
    "replay_weight": 0.5,
    "lr": 0.001,
    "num_epochs": 50,
    "batch_size": 256,
    "dropout": 0.1,
    "margin": 1.0,
    "neg_ratio": 1,
    "use_distillation": False,
    "distillation_temperature": 2.0,
    "distillation_alpha": 0.5,
    # Score-level fusion (MoSE-style): per-modality decoders, combined at eval
    "score_fusion_alpha_text": 0.5,   # text score weight at eval
    "score_fusion_alpha_mol": 0.3,    # mol score weight at eval
    "text_loss_weight": 1.0,          # text decoder training loss weight
    "mol_loss_weight": 1.0,           # mol decoder training loss weight
    # OGM-GE gradient modulation
    "use_ogm": False,
    "ogm_alpha": 1.0,
    # Contrastive modality alignment
    "contrastive_weight": 0.0,        # 0 = disabled
    "contrastive_temp": 0.1,
}


class CMKL(nn.Module):
    """Continual Multimodal Knowledge Graph Learner.

    Combines structural (R-GCN), textual (BiomedBERT), and molecular
    (Morgan fingerprint MLP) encoders with cross-modal attention fusion
    and modality-aware continual learning mechanisms.

    Args:
        config: Configuration dict with model hyperparameters.
            See DEFAULT_CONFIG for expected keys and defaults.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **config}
        c = self.config

        # These are set during training when we know the data dimensions
        self.num_entities = c.get("num_entities", 0)
        self.num_relations = c.get("num_relations", 0)
        self.embedding_dim = c["embedding_dim"]
        # Complex-valued decoders (RotatE, ComplEx) need 2× entity dim for
        # real/imag pairs, matching PyKEEN's internal storage.
        self.is_complex_decoder = c["decoder_type"] in ("RotatE", "ComplEx")
        self.entity_dim = 2 * c["embedding_dim"] if self.is_complex_decoder else c["embedding_dim"]
        # Relation dim — RotatE uses D phases (unit modulus), ComplEx uses full
        # 2D complex relations, everything else uses entity_dim.
        if c["decoder_type"] == "RotatE":
            self.rel_dim = c["embedding_dim"]
        else:
            self.rel_dim = self.entity_dim

        # --- Component 1: Modality-specific encoders ---
        if self.num_entities > 0 and self.num_relations > 0:
            self._init_encoders()
        else:
            # Encoders will be initialized when data dimensions are known
            self.structural_encoder = None
            self.textual_encoder = None
            self.molecular_encoder = None

        # --- Component 2: Cross-modal fusion ---
        # Fusion operates on entity_dim (which is 2×D for RotatE)
        if c["fusion_type"] == "cross_attention":
            self.fusion = CrossModalAttentionFusion(
                embed_dim=self.entity_dim,
                num_heads=c["num_attention_heads"],
                dropout=c["dropout"],
            )
        elif c["fusion_type"] == "moe":
            self.fusion = MixtureOfExpertsFusion(
                embed_dim=self.entity_dim,
                router_hidden_dim=c.get("router_hidden_dim", 128),
                dropout=c["dropout"],
                load_balance_weight=c.get("load_balance_weight", 0.01),
            )
        else:
            self.fusion = ConcatenationFusion(
                embed_dim=self.entity_dim,
                dropout=c["dropout"],
            )

        # --- Component 3: Decoder ---
        decoder_cls = DECODER_REGISTRY.get(c["decoder_type"], DistMultDecoder)
        if c["decoder_type"] == "Bilinear":
            self.decoder = decoder_cls(
                embedding_dim=c["embedding_dim"],
                num_relations=max(self.num_relations, 1),
            )
        else:
            self.decoder = decoder_cls(embedding_dim=c["embedding_dim"])

        # Relation embeddings — use self.rel_dim directly so ComplEx (rel_dim
        # = 2D, full complex relations) gets the right shape on the first pass.
        # Previously this hardcoded c["embedding_dim"] and relied on
        # init_for_data() to rebuild it; but init_for_data is only called when
        # structural_encoder is None, so for configs with num_entities/num_relations
        # already set, the override never ran and ComplEx silently trained with
        # half-sized relation embeddings.
        self.relation_emb = None
        if self.num_relations > 0:
            self.relation_emb = nn.Embedding(self.num_relations, self.rel_dim)
            if c["decoder_type"] == "RotatE":
                import math
                nn.init.uniform_(self.relation_emb.weight, -math.pi, math.pi)
            else:
                nn.init.xavier_uniform_(self.relation_emb.weight)

        # --- Score-level fusion: per-modality decoders + relation embeddings ---
        self.decoder_text: nn.Module | None = None
        self.decoder_mol: nn.Module | None = None
        self.relation_emb_text: nn.Embedding | None = None
        self.relation_emb_mol: nn.Embedding | None = None
        if c["fusion_type"] == "score_fusion":
            self._init_score_fusion_decoders()

        # --- OGM-GE gradient modulation ---
        self.ogm: OGMGEModulator | None = None
        if c.get("use_ogm", False):
            self.ogm = OGMGEModulator(
                modality_names=["struct", "text", "mol"],
                alpha=c.get("ogm_alpha", 1.0),
            )

        # --- Component 4: Continual learning modules (not nn.Module, separate) ---
        # These are initialized lazily during training
        self.ewc: ModalityAwareEWC | None = None
        self.replay_buffer: MultimodalMemoryBuffer | None = None

        # --- Optional: Knowledge Distillation ---
        self.distillation: KnowledgeDistillation | None = None
        self._teacher_model: CMKL | None = None

        # --- Optional: Node Classification head ---
        self.nc_classifier: nn.Sequential | None = None
        if c.get("use_nc", False):
            num_classes = c.get("num_nc_classes", 10)
            self.nc_classifier = nn.Sequential(
                nn.Linear(self.entity_dim, self.entity_dim),
                nn.ReLU(),
                nn.Dropout(c["dropout"]),
                nn.Linear(self.entity_dim, num_classes),
            )

    def _init_encoders(self) -> None:
        """Initialize encoders once data dimensions are known.

        For RotatE: entity dim is 2× embedding_dim (real pairs for complex
        pairs). The structural encoder runs R-GCN at embedding_dim internally
        and carries its own Linear projection to entity_dim — keeping that
        projection *inside* StructuralEncoder means MA-EWC protects it like
        any other R-GCN parameter when computing the per-modality Fisher.
        """
        c = self.config
        self.structural_encoder = StructuralEncoder(
            num_nodes=self.num_entities,
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim,
            out_dim=self.entity_dim,
            num_layers=c["num_gnn_layers"],
            num_bases=c["num_gnn_bases"],
            preserve_complex_geom=self.is_complex_decoder,
        )
        self.textual_encoder = TextualEncoder(
            projection_dim=self.entity_dim,
            lora_rank=c.get("text_lora_rank", 0),
        )
        self.molecular_encoder = MolecularEncoder(
            input_dim=c.get("mol_input_dim", 1024),
            projection_dim=self.entity_dim,
            dropout=c["dropout"],
        )

    def _init_score_fusion_decoders(self) -> None:
        """Initialize per-modality decoders and relation embeddings for score-level fusion."""
        c = self.config
        decoder_cls = DECODER_REGISTRY.get(c["decoder_type"], DistMultDecoder)

        # Text decoder + relation embeddings
        if c["decoder_type"] == "Bilinear":
            self.decoder_text = BilinearDecoder(
                embedding_dim=c["embedding_dim"],
                num_relations=max(self.num_relations, 1),
            )
            self.decoder_mol = BilinearDecoder(
                embedding_dim=c["embedding_dim"],
                num_relations=max(self.num_relations, 1),
            )
        else:
            self.decoder_text = decoder_cls(embedding_dim=c["embedding_dim"])
            self.decoder_mol = decoder_cls(embedding_dim=c["embedding_dim"])

        if self.num_relations > 0:
            self.relation_emb_text = nn.Embedding(self.num_relations, c["embedding_dim"])
            nn.init.xavier_uniform_(self.relation_emb_text.weight)
            self.relation_emb_mol = nn.Embedding(self.num_relations, c["embedding_dim"])
            nn.init.xavier_uniform_(self.relation_emb_mol.weight)

    def init_for_data(
        self,
        num_entities: int,
        num_relations: int,
    ) -> None:
        """Initialize model components that depend on data dimensions.

        Called before training begins once we know the KG size.

        Args:
            num_entities: Total number of entities across all tasks.
            num_relations: Total number of relation types across all tasks.
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self._init_encoders()

        # Re-init relation embeddings
        # For RotatE: relations are phase parameters in [-pi, pi] of dim embedding_dim
        # For others: standard Glorot init of dim entity_dim
        self.relation_emb = nn.Embedding(num_relations, self.rel_dim)
        if self.config["decoder_type"] == "RotatE":
            # Initialize as uniform phases in [-pi, pi] (PyKEEN RotatE convention)
            import math
            nn.init.uniform_(self.relation_emb.weight, -math.pi, math.pi)
        else:
            nn.init.xavier_uniform_(self.relation_emb.weight)

        # Re-init bilinear decoder if needed
        if self.config["decoder_type"] == "Bilinear":
            self.decoder = BilinearDecoder(
                embedding_dim=self.embedding_dim,
                num_relations=num_relations,
            )

        # Re-init score fusion decoders if needed
        if self.config["fusion_type"] == "score_fusion":
            self._init_score_fusion_decoders()

    def encode_structural(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all nodes through the structural R-GCN encoder.

        Args:
            edge_index: [2, num_edges] edge indices.
            edge_type: [num_edges] edge type labels.

        Returns:
            Structural embeddings [num_entities, entity_dim]. R-GCN runs at
            embedding_dim internally and is projected to entity_dim by the
            encoder's own `out_proj` when they differ (RotatE only).
        """
        return self.structural_encoder(edge_index, edge_type)

    def encode_textual(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project pre-computed text embeddings through the textual encoder.

        Args:
            text_embeddings: [N_text, 768] pre-computed BiomedBERT embeddings.

        Returns:
            Projected text embeddings [N_text, embedding_dim].
        """
        return self.textual_encoder(text_embeddings)

    def encode_molecular(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Encode molecular fingerprints through the molecular encoder.

        Args:
            fingerprints: [N_mol, 1024] Morgan fingerprint vectors.

        Returns:
            Molecular embeddings [N_mol, embedding_dim].
        """
        return self.molecular_encoder(fingerprints)

    def forward(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        mol_fingerprints: torch.Tensor | None = None,
        node_has_text: torch.Tensor | None = None,
        node_has_mol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: encode all modalities and fuse.

        Args:
            edge_index: [2, E] edge indices for R-GCN.
            edge_type: [E] edge types for R-GCN.
            text_embeddings: [N, 768] pre-computed text features (zeros where missing).
            mol_fingerprints: [N, 1024] Morgan fingerprints (zeros where missing).
            node_has_text: [N] boolean mask for text availability.
            node_has_mol: [N] boolean mask for molecular availability.

        Returns:
            Fused node embeddings [N, embedding_dim].
        """
        N = self.num_entities
        D = self.entity_dim
        device = next(self.parameters()).device

        # --- Structural encoding (always available) ---
        h_struct = self.encode_structural(edge_index, edge_type)

        # --- Textual encoding ---
        if text_embeddings is not None and node_has_text is not None:
            text_idx = node_has_text.nonzero(as_tuple=True)[0]
            if text_idx.numel() > 0:
                h_text = self.encode_textual(text_embeddings[text_idx])
            else:
                h_text = torch.zeros(0, D, device=device)
        else:
            h_text = torch.zeros(0, D, device=device)
            node_has_text = torch.zeros(N, dtype=torch.bool, device=device)

        # --- Molecular encoding ---
        if mol_fingerprints is not None and node_has_mol is not None:
            mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
            if mol_idx.numel() > 0:
                h_mol = self.encode_molecular(mol_fingerprints[mol_idx])
            else:
                h_mol = torch.zeros(0, D, device=device)
        else:
            h_mol = torch.zeros(0, D, device=device)
            node_has_mol = torch.zeros(N, dtype=torch.bool, device=device)

        # --- Fusion ---
        h_fused = self.fusion(h_struct, h_text, h_mol, node_has_text, node_has_mol)

        return h_fused

    def score_triples(
        self,
        node_embeddings: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score (head, relation, tail) triples using the decoder.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            head_ids: [B] head entity indices.
            relation_ids: [B] relation type indices.
            tail_ids: [B] tail entity indices.

        Returns:
            Scores [B].
        """
        h = node_embeddings[head_ids]
        t = node_embeddings[tail_ids]

        if self.config["decoder_type"] == "Bilinear":
            return self.decoder(h, relation_ids, t)
        else:
            r = self.relation_emb(relation_ids)
            return self.decoder(h, r, t)

    def classify_nodes(
        self,
        node_embeddings: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Classify nodes using the optional NC head.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            node_ids: [B] indices of nodes to classify.

        Returns:
            Logits [B, num_classes].

        Raises:
            RuntimeError: If NC head is not initialized (use_nc=False).
        """
        if self.nc_classifier is None:
            raise RuntimeError("NC classifier not initialized. Set use_nc=True in config.")
        return self.nc_classifier(node_embeddings[node_ids])

    def compute_task_loss(
        self,
        node_embeddings: torch.Tensor,
        triples: torch.Tensor,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Compute link prediction loss with negative sampling.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            triples: [B, 3] integer triples (mapped head, relation, tail).
            entity_to_id: Entity-to-ID mapping (for num_entities).
            relation_to_id: Relation-to-ID mapping.
            margin: Margin for ranking loss.

        Returns:
            Scalar loss tensor.
        """
        device = node_embeddings.device
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]

        # Positive scores
        pos_scores = self.score_triples(node_embeddings, heads, rels, tails)

        # Negative sampling: corrupt head or tail
        neg_triples = triples.clone()
        n = neg_triples.shape[0]
        mask = torch.rand(n, device=device) < 0.5
        random_entities = torch.randint(0, self.num_entities, (n,), device=device)
        neg_triples[mask, 0] = random_entities[mask]
        neg_triples[~mask, 2] = random_entities[~mask]

        neg_heads = neg_triples[:, 0]
        neg_rels = neg_triples[:, 1]
        neg_tails = neg_triples[:, 2]
        neg_scores = self.score_triples(node_embeddings, neg_heads, neg_rels, neg_tails)

        # Margin ranking loss
        loss = torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()
        return loss

    # ================================================================
    # Score-level fusion methods
    # ================================================================

    def forward_per_modality(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        mol_fingerprints: torch.Tensor | None = None,
        node_has_text: torch.Tensor | None = None,
        node_has_mol: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode each modality separately without fusion.

        Used by score-level fusion where each modality scores independently.

        Returns:
            Dict with h_struct [N, D], h_text_full [N, D], h_mol_full [N, D],
            node_has_text [N], node_has_mol [N].
        """
        N = self.num_entities
        D = self.entity_dim
        device = next(self.parameters()).device

        h_struct = self.encode_structural(edge_index, edge_type)

        # Text: full-size tensor with zeros for missing entities
        h_text_full = torch.zeros(N, D, device=device)
        if text_embeddings is not None and node_has_text is not None:
            text_idx = node_has_text.nonzero(as_tuple=True)[0]
            if text_idx.numel() > 0:
                h_text_full[text_idx] = self.encode_textual(text_embeddings[text_idx])
        else:
            node_has_text = torch.zeros(N, dtype=torch.bool, device=device)

        # Molecular: full-size tensor with zeros for missing entities
        h_mol_full = torch.zeros(N, D, device=device)
        if mol_fingerprints is not None and node_has_mol is not None:
            mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
            if mol_idx.numel() > 0:
                h_mol_full[mol_idx] = self.encode_molecular(mol_fingerprints[mol_idx])
        else:
            node_has_mol = torch.zeros(N, dtype=torch.bool, device=device)

        return {
            "h_struct": h_struct,
            "h_text_full": h_text_full,
            "h_mol_full": h_mol_full,
            "node_has_text": node_has_text,
            "node_has_mol": node_has_mol,
        }

    def score_triples_modality(
        self,
        embeddings: torch.Tensor,
        decoder: nn.Module,
        relation_emb: nn.Embedding,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score triples using a specific modality's decoder and relation embeddings.

        Args:
            embeddings: [N, D] modality-specific node embeddings.
            decoder: Modality-specific decoder module.
            relation_emb: Modality-specific relation embeddings.
            head_ids, relation_ids, tail_ids: [B] triple component indices.

        Returns:
            Scores [B].
        """
        h = embeddings[head_ids]
        t = embeddings[tail_ids]
        if self.config["decoder_type"] == "Bilinear":
            return decoder(h, relation_ids, t)
        else:
            r = relation_emb(relation_ids)
            return decoder(h, r, t)

    def compute_modality_loss(
        self,
        h_full: torch.Tensor,
        node_has_modality: torch.Tensor,
        decoder: nn.Module,
        relation_emb: nn.Embedding,
        triples: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Compute LP loss for a single modality on triples where both h and t have it.

        Negative sampling corrupts with modality-having entities only to give
        the decoder a clean gradient signal.

        Args:
            h_full: [N, D] modality embeddings (zeros where modality missing).
            node_has_modality: [N] boolean mask.
            decoder: Modality-specific decoder.
            relation_emb: Modality-specific relation embeddings.
            triples: [B, 3] sampled training triples.
            margin: Margin for ranking loss.

        Returns:
            Scalar loss (0.0 if no qualifying triples).
        """
        device = h_full.device
        heads = triples[:, 0]
        tails = triples[:, 2]

        # Filter to triples where both head and tail have this modality
        both_have = node_has_modality[heads] & node_has_modality[tails]
        if both_have.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mod_triples = triples[both_have]
        mh = mod_triples[:, 0]
        mr = mod_triples[:, 1]
        mt = mod_triples[:, 2]

        # Positive scores
        pos_scores = self.score_triples_modality(h_full, decoder, relation_emb, mh, mr, mt)

        # Negative sampling: corrupt with modality-having entities only
        modality_entity_ids = node_has_modality.nonzero(as_tuple=True)[0]
        n = mod_triples.shape[0]
        neg_triples = mod_triples.clone()
        mask = torch.rand(n, device=device) < 0.5
        rand_idx = torch.randint(0, modality_entity_ids.shape[0], (n,), device=device)
        random_mod_entities = modality_entity_ids[rand_idx]
        neg_triples[mask, 0] = random_mod_entities[mask]
        neg_triples[~mask, 2] = random_mod_entities[~mask]

        neg_scores = self.score_triples_modality(
            h_full, decoder, relation_emb,
            neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2],
        )

        loss = torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()
        return loss

    def compute_contrastive_alignment_loss(
        self,
        h_struct: torch.Tensor,
        h_text_full: torch.Tensor,
        h_mol_full: torch.Tensor,
        node_has_text: torch.Tensor,
        node_has_mol: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """InfoNCE loss aligning struct↔text and struct↔mol for same entities.

        Pushes same-entity representations from different modalities closer
        while pushing different-entity representations apart.

        Args:
            h_struct: [N, D] structural embeddings.
            h_text_full: [N, D] text embeddings (zeros where missing).
            h_mol_full: [N, D] molecular embeddings (zeros where missing).
            node_has_text: [N] boolean mask.
            node_has_mol: [N] boolean mask.
            temperature: Softmax temperature for InfoNCE.

        Returns:
            Scalar contrastive loss.
        """
        device = h_struct.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_pairs = 0

        # Struct ↔ Text alignment
        text_idx = node_has_text.nonzero(as_tuple=True)[0]
        if text_idx.numel() >= 2:
            # Subsample for memory efficiency
            if text_idx.numel() > 1024:
                perm = torch.randperm(text_idx.numel(), device=device)[:1024]
                text_idx = text_idx[perm]

            z_s = F.normalize(h_struct[text_idx], dim=-1)
            z_t = F.normalize(h_text_full[text_idx], dim=-1)
            sim = z_s @ z_t.T / temperature
            labels = torch.arange(sim.shape[0], device=device)
            loss_st = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
            total_loss = total_loss + loss_st
            n_pairs += 1

        # Struct ↔ Mol alignment
        mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
        if mol_idx.numel() >= 2:
            if mol_idx.numel() > 1024:
                perm = torch.randperm(mol_idx.numel(), device=device)[:1024]
                mol_idx = mol_idx[perm]

            z_s = F.normalize(h_struct[mol_idx], dim=-1)
            z_m = F.normalize(h_mol_full[mol_idx], dim=-1)
            sim = z_s @ z_m.T / temperature
            labels = torch.arange(sim.shape[0], device=device)
            loss_sm = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
            total_loss = total_loss + loss_sm
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss

    def _score_all_entities_modality(
        self,
        h_embs: torch.Tensor,
        rels: torch.Tensor,
        all_embeddings: torch.Tensor,
        decoder: nn.Module,
        relation_emb: nn.Embedding,
    ) -> torch.Tensor:
        """Score all entities as tails for given (h, r) using a modality-specific decoder.

        Args:
            h_embs: [B, D] head embeddings.
            rels: [B] relation indices.
            all_embeddings: [N, D] all entity embeddings for this modality.
            decoder: Modality-specific decoder.
            relation_emb: Modality-specific relation embeddings.

        Returns:
            Scores [B, N].
        """
        if self.config["decoder_type"] == "Bilinear":
            M = decoder.relation_matrices[rels]
            h_M = torch.einsum("bi,bij->bj", h_embs, M)
            return h_M @ all_embeddings.T
        elif self.config["decoder_type"] == "TransE":
            r_embs = relation_emb(rels)
            query = h_embs + r_embs
            return -torch.cdist(query, all_embeddings, p=decoder.p_norm)
        elif self.config["decoder_type"] == "RotatE":
            # RotatE (PyKEEN convention):
            #   entity embeddings are real [N, 2*D] = D complex pairs
            #   relation embeddings are real [N_r, D] = D phases
            #   r_complex = exp(i*phase)
            #   score = -||h ∘ r - t||_2
            r_phases = relation_emb(rels)  # [B, D]
            B = h_embs.shape[0]
            N = all_embeddings.shape[0]
            D = decoder.embedding_dim  # number of complex pairs
            h_c = torch.view_as_complex(h_embs.reshape(B, D, 2).contiguous())  # [B, D]
            t_c = torch.view_as_complex(all_embeddings.reshape(N, D, 2).contiguous())  # [N, D]
            r_c = torch.complex(torch.cos(r_phases), torch.sin(r_phases))  # [B, D]
            # query = h ∘ r, shape [B, D]
            query = h_c * r_c
            scores = torch.empty(B, N, device=h_embs.device)
            chunk = 8192
            for i in range(0, N, chunk):
                j = min(i + chunk, N)
                diff = query.unsqueeze(1) - t_c[i:j].unsqueeze(0)
                scores[:, i:j] = -diff.abs().pow(2).sum(dim=-1).sqrt().float()
            return scores
        else:  # DistMult
            r_embs = relation_emb(rels)
            query = h_embs * r_embs
            return query @ all_embeddings.T

    def train_continually(
        self,
        task_sequence: OrderedDict[str, dict[str, np.ndarray]],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        device: str = "auto",
        text_embeddings: torch.Tensor | None = None,
        mol_fingerprints: torch.Tensor | None = None,
        node_has_text: torch.Tensor | None = None,
        node_has_mol: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        seed: int = 42,
    ) -> dict:
        """Train CMKL on a sequence of tasks.

        For each task:
        1. Train with combined loss (task + EWC + replay)
        2. Compute per-modality Fisher information
        3. Add exemplars to multimodal memory buffer
        4. Evaluate on all tasks seen so far

        Args:
            task_sequence: OrderedDict of task_name -> {'train': array, 'val': array, 'test': array}.
                Arrays contain string triples (head, relation, tail).
            entity_to_id: Global entity-to-ID mapping.
            relation_to_id: Global relation-to-ID mapping.
            device: Device for training.
            text_embeddings: [N, 768] pre-computed text features.
            mol_fingerprints: [N, 1024] Morgan fingerprints.
            node_has_text: [N] boolean mask.
            node_has_mol: [N] boolean mask.
            edge_index: [2, E] edges for R-GCN.
            edge_type: [E] edge types.
            seed: Random seed.

        Returns:
            Dict with results_matrix, per-task metrics, training logs.
        """
        c = self.config
        device = get_device(device)
        torch.manual_seed(seed)

        # Initialize model for data dimensions
        if self.structural_encoder is None:
            self.init_for_data(len(entity_to_id), len(relation_to_id))
        self.to(device)

        # Initialize continual learning modules
        self.ewc = ModalityAwareEWC(
            self,
            lambda_struct=c["lambda_struct"],
            lambda_text=c["lambda_text"],
            lambda_mol=c["lambda_mol"],
            lambda_fusion=c.get("lambda_fusion", 5.0),
            lambda_relation=c.get("lambda_relation", 10.0),
        )
        self.replay_buffer = MultimodalMemoryBuffer(
            max_size=c["replay_buffer_size"],
            strategy=c["replay_strategy"],
        )

        # Initialize distillation if enabled
        if c.get("use_distillation", False):
            self.distillation = KnowledgeDistillation(
                temperature=c.get("distillation_temperature", 2.0),
                alpha=c.get("distillation_alpha", 0.5),
            )
            self._teacher_model = None
            logger.info("Knowledge distillation enabled (T=%.1f, alpha=%.2f)",
                        self.distillation.temperature, self.distillation.alpha)

        # Move multimodal features to device
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(device)
        if mol_fingerprints is not None:
            mol_fingerprints = mol_fingerprints.to(device)
        if node_has_text is not None:
            node_has_text = node_has_text.to(device)
        if node_has_mol is not None:
            node_has_mol = node_has_mol.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)

        task_names = list(task_sequence.keys())
        num_tasks = len(task_names)
        results_matrix = np.zeros((num_tasks, num_tasks))

        optimizer = torch.optim.Adam(self.parameters(), lr=c["lr"])

        for task_idx, task_name in enumerate(task_names):
            logger.info(f"=== Task {task_idx + 1}/{num_tasks}: {task_name} ===")
            task_data = task_sequence[task_name]

            # Reset OGM loss history for new task (loss profiles change per task)
            if self.ogm is not None:
                self.ogm.reset()

            # Map string triples to integer IDs
            train_triples = self._map_triples(
                task_data["train"], entity_to_id, relation_to_id
            )
            train_triples_t = torch.tensor(train_triples, dtype=torch.long, device=device)

            # Training loop
            self.train()
            for epoch in range(c["num_epochs"]):
                epoch_loss = self._train_epoch(
                    train_triples_t,
                    optimizer,
                    entity_to_id,
                    relation_to_id,
                    text_embeddings,
                    mol_fingerprints,
                    node_has_text,
                    node_has_mol,
                    edge_index,
                    edge_type,
                    device,
                )
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    msg = f"  Epoch {epoch + 1}/{c['num_epochs']}: loss={epoch_loss:.4f}"
                    if hasattr(self.fusion, '_last_router_weights') and self.fusion._last_router_weights is not None:
                        w = self.fusion._last_router_weights
                        msg += f" | Router(text nodes): s={w[0]:.3f} t={w[1]:.3f} m={w[2]:.3f}"
                    if c["fusion_type"] == "score_fusion" and hasattr(self, '_last_modality_losses'):
                        ml = self._last_modality_losses
                        msg += f" | losses: s={ml['struct']:.4f} t={ml['text']:.4f} m={ml['mol']:.4f}"
                        if hasattr(self, '_last_ogm_weights'):
                            ow = self._last_ogm_weights
                            msg += f" | OGM: s={ow['struct']:.2f} t={ow['text']:.2f} m={ow['mol']:.2f}"
                    logger.info(msg)

            # After training on this task:
            # 1. Compute per-modality Fisher
            self._compute_fisher_for_task(
                train_triples_t,
                entity_to_id,
                relation_to_id,
                text_embeddings,
                mol_fingerprints,
                node_has_text,
                node_has_mol,
                edge_index,
                edge_type,
                device,
            )

            # 2. Add exemplars to replay buffer
            self.eval()
            with torch.no_grad():
                if c["fusion_type"] == "score_fusion":
                    mod_data = self.forward_per_modality(
                        edge_index, edge_type,
                        text_embeddings, mol_fingerprints,
                        node_has_text, node_has_mol,
                    )
                    h_fused = mod_data["h_struct"]
                else:
                    h_fused = self.forward(
                        edge_index, edge_type,
                        text_embeddings, mol_fingerprints,
                        node_has_text, node_has_mol,
                    )
            # Select a subset of training triples for the buffer
            n_exemplars = min(len(train_triples), c["replay_buffer_size"] // num_tasks)
            indices = np.random.choice(len(train_triples), n_exemplars, replace=False)
            exemplar_triples = train_triples[indices]
            self.replay_buffer.add_exemplars(
                exemplar_triples,
                h_fused,
                text_embeddings,
                mol_fingerprints,
                node_has_text,
                node_has_mol,
                task_id=task_idx,
            )

            # 3. Create teacher copy for distillation on the next task
            if self.distillation is not None:
                self._teacher_model = KnowledgeDistillation.create_teacher_copy(self)
                self._teacher_model.to(device)

            # 4. Evaluate on all tasks seen so far
            self.eval()
            import time as _t
            _eval_start = _t.time()
            logger.info("  [eval] starting eval phase")

            # Build filter set: all known triples from tasks seen so far
            all_known_triples = np.concatenate([
                np.concatenate([
                    task_sequence[task_names[k]][split]
                    for split in ("train", "val", "test")
                    if split in task_sequence[task_names[k]]
                ])
                for k in range(task_idx + 1)
            ])
            logger.info(f"  [eval] all_known_triples built: {len(all_known_triples)} rows in {_t.time()-_eval_start:.1f}s")

            for eval_idx in range(task_idx + 1):
                eval_name = task_names[eval_idx]
                eval_data = task_sequence[eval_name]

                _t0 = _t.time()
                # Compute MRR with filtered ranking
                test_mrr = self._evaluate_mrr(
                    eval_data["test"],
                    entity_to_id,
                    relation_to_id,
                    text_embeddings,
                    mol_fingerprints,
                    node_has_text,
                    node_has_mol,
                    edge_index,
                    edge_type,
                    device,
                    known_triples=all_known_triples,
                )
                results_matrix[task_idx, eval_idx] = test_mrr
                logger.info(f"  Eval {eval_name}: MRR={test_mrr:.4f} (took {_t.time()-_t0:.1f}s)")

        return {
            "results_matrix": results_matrix.tolist(),
            "task_names": task_names,
            "config": c,
            "seed": seed,
        }

    def _map_triples(
        self,
        triples: np.ndarray,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Ensure triples are int64 format.

        Note: load_task_sequence() already returns pre-mapped int64 arrays,
        so this is effectively an identity operation. Previously this tried
        to map via entity_to_id.get() but the keys are strings while the
        values coming in are int64, causing all lookups to return 0.

        Args:
            triples: [N, 3] int64 triples (already mapped).
            entity_to_id: Entity mapping (unused, kept for API compat).
            relation_to_id: Relation mapping (unused, kept for API compat).

        Returns:
            [N, 3] integer triples.
        """
        return np.asarray(triples, dtype=np.int64)

    def _train_epoch(
        self,
        train_triples: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        entity_to_id: dict,
        relation_to_id: dict,
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
    ) -> float:
        """Run one training epoch: one R-GCN forward + sampled batch training.

        Standard R-GCN link prediction loop (Schlichtkrull et al., 2018):
        1. ONE full-graph R-GCN forward pass (with gradients)
        2. Sample a batch of triples (not iterate all)
        3. Compute combined loss (task + EWC + replay + distillation)
        4. ONE backward pass through decoder, fusion, AND R-GCN
        5. ONE optimizer.step()

        For score_fusion mode: separate per-modality losses with independent
        gradient paths, plus optional OGM-GE modulation and contrastive alignment.
        """
        c = self.config
        self.train()

        n = train_triples.shape[0]
        samples_per_epoch = min(c.get("samples_per_epoch", 50000), n)
        idx = torch.randint(0, n, (samples_per_epoch,), device=device)
        batch = train_triples[idx]

        if c["fusion_type"] == "score_fusion":
            return self._train_epoch_score_fusion(
                batch, optimizer, entity_to_id, relation_to_id,
                text_embeddings, mol_fingerprints,
                node_has_text, node_has_mol,
                edge_index, edge_type, device,
            )

        # --- Existing embedding-level fusion path (cross_attention, moe, concatenation) ---

        # Step 1: ONE full-graph R-GCN forward pass (with gradients)
        h_fused = self.forward(
            edge_index, edge_type,
            text_embeddings, mol_fingerprints,
            node_has_text, node_has_mol,
        )

        # Step 3: Task loss on sampled batch
        loss = self.compute_task_loss(
            h_fused, batch, entity_to_id, relation_to_id,
            margin=c["margin"],
        )

        # Step 3b: MoE auxiliary load-balancing loss
        if hasattr(self.fusion, 'aux_loss') and self.fusion.aux_loss is not None:
            loss = loss + self.fusion.aux_loss

        # Step 4: EWC penalty (operates on parameters, no forward needed)
        if self.ewc is not None:
            ewc_penalty = self.ewc.ewc_loss()
            loss = loss + ewc_penalty

        # Step 5: Distillation loss (subsample to avoid OOM on all-entity scoring)
        if (self.distillation is not None
                and self._teacher_model is not None):
            with torch.no_grad():
                teacher_h = self._teacher_model.forward(
                    edge_index, edge_type,
                    text_embeddings, mol_fingerprints,
                    node_has_text, node_has_mol,
                )
            distill_size = min(c["batch_size"], batch.shape[0])
            distill_idx = torch.randint(0, batch.shape[0], (distill_size,), device=device)
            distill_batch = batch[distill_idx]
            heads = distill_batch[:, 0]
            rels = distill_batch[:, 1]
            # Student all-entity scores
            if self.config["decoder_type"] == "Bilinear":
                s_h = h_fused[heads]
                s_M = self.decoder.relation_matrices[rels]
                s_hM = torch.einsum("bi,bij->bj", s_h, s_M)
                student_scores = s_hM @ h_fused.T
            elif self.config["decoder_type"] == "TransE":
                s_query = h_fused[heads] + self.relation_emb(rels)
                student_scores = -torch.cdist(
                    s_query, h_fused, p=self.decoder.p_norm)
            else:  # DistMult
                s_query = h_fused[heads] * self.relation_emb(rels)
                student_scores = s_query @ h_fused.T

            # Teacher all-entity scores (same formulation)
            with torch.no_grad():
                if self.config["decoder_type"] == "Bilinear":
                    t_h = teacher_h[heads]
                    t_M = self._teacher_model.decoder.relation_matrices[rels]
                    t_hM = torch.einsum("bi,bij->bj", t_h, t_M)
                    teacher_scores = t_hM @ teacher_h.T
                elif self.config["decoder_type"] == "TransE":
                    t_query = teacher_h[heads] + self._teacher_model.relation_emb(rels)
                    teacher_scores = -torch.cdist(
                        t_query, teacher_h, p=self._teacher_model.decoder.p_norm)
                else:  # DistMult
                    t_query = teacher_h[heads] * self._teacher_model.relation_emb(rels)
                    teacher_scores = t_query @ teacher_h.T

            loss = self.distillation.compute_combined_loss(
                loss, student_scores, teacher_scores)

        # Step 6: Replay loss
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_triples = self.replay_buffer.get_replay_triples(
                min(c["batch_size"], len(self.replay_buffer))
            )
            if replay_triples is not None:
                replay_t = torch.tensor(replay_triples, dtype=torch.long, device=device)
                replay_loss = self.compute_task_loss(
                    h_fused, replay_t, entity_to_id, relation_to_id,
                    margin=c["margin"],
                )
                loss = loss + c["replay_weight"] * replay_loss

        # Step 7: ONE backward pass + ONE optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def _train_epoch_score_fusion(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        entity_to_id: dict,
        relation_to_id: dict,
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
    ) -> float:
        """Score-level fusion training: per-modality decoders with independent losses.

        Each modality gets its own decoder + relation embeddings + loss.
        Losses are combined with optional OGM-GE gradient modulation.
        """
        c = self.config

        # Step 1: Encode per-modality (no fusion)
        mod = self.forward_per_modality(
            edge_index, edge_type,
            text_embeddings, mol_fingerprints,
            node_has_text, node_has_mol,
        )
        h_struct = mod["h_struct"]
        h_text_full = mod["h_text_full"]
        h_mol_full = mod["h_mol_full"]
        nht = mod["node_has_text"]
        nhm = mod["node_has_mol"]

        # Step 2: Per-modality losses
        loss_struct = self.compute_task_loss(
            h_struct, batch, entity_to_id, relation_to_id,
            margin=c["margin"],
        )
        loss_text = self.compute_modality_loss(
            h_text_full, nht, self.decoder_text, self.relation_emb_text,
            batch, margin=c["margin"],
        )
        loss_mol = self.compute_modality_loss(
            h_mol_full, nhm, self.decoder_mol, self.relation_emb_mol,
            batch, margin=c["margin"],
        )

        # Step 3: Combine with OGM-GE modulation or fixed weights
        if self.ogm is not None:
            ogm_weights = self.ogm.compute_modulation_weights({
                "struct": loss_struct.item(),
                "text": loss_text.item(),
                "mol": loss_mol.item(),
            })
            w_s = ogm_weights["struct"]
            w_t = ogm_weights["text"]
            w_m = ogm_weights["mol"]
        else:
            w_s, w_t, w_m = 1.0, 1.0, 1.0

        loss = (w_s * loss_struct
                + c.get("text_loss_weight", 1.0) * w_t * loss_text
                + c.get("mol_loss_weight", 1.0) * w_m * loss_mol)

        # Store per-modality losses for logging
        self._last_modality_losses = {
            "struct": loss_struct.item(),
            "text": loss_text.item(),
            "mol": loss_mol.item(),
        }
        if self.ogm is not None:
            self._last_ogm_weights = {"struct": w_s, "text": w_t, "mol": w_m}

        # Step 4: Contrastive alignment loss
        cw = c.get("contrastive_weight", 0.0)
        if cw > 0:
            contrastive_loss = self.compute_contrastive_alignment_loss(
                h_struct, h_text_full, h_mol_full,
                nht, nhm,
                temperature=c.get("contrastive_temp", 0.1),
            )
            loss = loss + cw * contrastive_loss

        # Step 5: EWC penalty
        if self.ewc is not None:
            ewc_penalty = self.ewc.ewc_loss()
            loss = loss + ewc_penalty

        # Step 6: Distillation (combined scores from teacher)
        if (self.distillation is not None
                and self._teacher_model is not None):
            alpha_t = c.get("score_fusion_alpha_text", 0.5)
            alpha_m = c.get("score_fusion_alpha_mol", 0.3)

            distill_size = min(c["batch_size"], batch.shape[0])
            distill_idx = torch.randint(0, batch.shape[0], (distill_size,), device=device)
            distill_batch = batch[distill_idx]
            heads = distill_batch[:, 0]
            rels = distill_batch[:, 1]

            # Student combined scores
            s_struct = self._score_all_entities_modality(
                h_struct[heads], rels, h_struct, self.decoder, self.relation_emb)
            s_text = self._score_all_entities_modality(
                h_text_full[heads], rels, h_text_full, self.decoder_text, self.relation_emb_text)
            s_mol = self._score_all_entities_modality(
                h_mol_full[heads], rels, h_mol_full, self.decoder_mol, self.relation_emb_mol)
            student_scores = s_struct + alpha_t * s_text + alpha_m * s_mol

            # Teacher combined scores
            with torch.no_grad():
                t_mod = self._teacher_model.forward_per_modality(
                    edge_index, edge_type,
                    text_embeddings, mol_fingerprints,
                    node_has_text, node_has_mol,
                )
                t_struct = self._teacher_model._score_all_entities_modality(
                    t_mod["h_struct"][heads], rels, t_mod["h_struct"],
                    self._teacher_model.decoder, self._teacher_model.relation_emb)
                t_text = self._teacher_model._score_all_entities_modality(
                    t_mod["h_text_full"][heads], rels, t_mod["h_text_full"],
                    self._teacher_model.decoder_text, self._teacher_model.relation_emb_text)
                t_mol = self._teacher_model._score_all_entities_modality(
                    t_mod["h_mol_full"][heads], rels, t_mod["h_mol_full"],
                    self._teacher_model.decoder_mol, self._teacher_model.relation_emb_mol)
                teacher_scores = t_struct + alpha_t * t_text + alpha_m * t_mol

            loss = self.distillation.compute_combined_loss(
                loss, student_scores, teacher_scores)

        # Step 7: Replay loss (all modality decoders to prevent forgetting)
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_triples = self.replay_buffer.get_replay_triples(
                min(c["batch_size"], len(self.replay_buffer))
            )
            if replay_triples is not None:
                replay_t = torch.tensor(replay_triples, dtype=torch.long, device=device)
                replay_struct = self.compute_task_loss(
                    h_struct, replay_t, entity_to_id, relation_to_id,
                    margin=c["margin"],
                )
                replay_text = self.compute_modality_loss(
                    h_text_full, nht, self.decoder_text, self.relation_emb_text,
                    replay_t, margin=c["margin"],
                )
                replay_mol = self.compute_modality_loss(
                    h_mol_full, nhm, self.decoder_mol, self.relation_emb_mol,
                    replay_t, margin=c["margin"],
                )
                replay_loss = replay_struct + replay_text + replay_mol
                loss = loss + c["replay_weight"] * replay_loss

        # Step 8: Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def _compute_fisher_for_task(
        self,
        train_triples: torch.Tensor,
        entity_to_id: dict,
        relation_to_id: dict,
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
    ) -> None:
        """Compute per-modality Fisher after finishing a task.

        Uses a single sampled batch (num_samples triples) with one full-graph
        R-GCN forward pass, consistent with the per-epoch training approach.
        """
        c = self.config
        # 1000 matches baseline EWC's default (src/baselines/ewc.py). The
        # previous 200 covered ~500-1K unique entities on a 123K-entity KG,
        # leaving most of the table un-penalized and therefore unprotected
        # under EWC. Raised to match the baseline and give entities fair
        # coverage during Fisher accumulation.
        num_samples = min(c.get("fisher_samples", 1000), train_triples.shape[0])

        # Sample a single batch of triples for Fisher estimation
        idx = torch.randint(0, train_triples.shape[0], (num_samples,),
                            device=device)
        fisher_batch = train_triples[idx]

        # Wrap in a DataLoader-like structure (single batch)
        dataset = torch.utils.data.TensorDataset(fisher_batch)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=num_samples, shuffle=False,
        )

        def compute_loss_fn(batch_tuple):
            triples_batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) else batch_tuple
            if self.config["fusion_type"] == "score_fusion":
                # All three modality losses for Fisher (captures gradients through all decoders)
                mod = self.forward_per_modality(
                    edge_index, edge_type,
                    text_embeddings, mol_fingerprints,
                    node_has_text, node_has_mol,
                )
                loss_s = self.compute_task_loss(
                    mod["h_struct"], triples_batch, entity_to_id, relation_to_id)
                loss_t = self.compute_modality_loss(
                    mod["h_text_full"], mod["node_has_text"],
                    self.decoder_text, self.relation_emb_text, triples_batch)
                loss_m = self.compute_modality_loss(
                    mod["h_mol_full"], mod["node_has_mol"],
                    self.decoder_mol, self.relation_emb_mol, triples_batch)
                return loss_s + loss_t + loss_m
            else:
                # One full-graph forward pass (with gradients for Fisher)
                h_fused = self.forward(
                    edge_index, edge_type,
                    text_embeddings, mol_fingerprints,
                    node_has_text, node_has_mol,
                )
                return self.compute_task_loss(
                    h_fused, triples_batch, entity_to_id, relation_to_id,
                )

        self.ewc.compute_modality_fisher(
            compute_loss_fn=compute_loss_fn,
            dataloader=dataloader,
            device=device,
            num_samples=num_samples,
        )

    @torch.no_grad()
    def _evaluate_mrr(
        self,
        test_triples: np.ndarray,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
        batch_size: int = 256,
        known_triples: np.ndarray | None = None,
    ) -> float:
        """Evaluate MRR on test triples with optional filtered ranking.

        For each test triple (h, r, t):
        - Score all entities as potential tails: score(h, r, e) for all e
        - Filter out known true tails (except the test triple itself)
        - Rank the true tail among remaining entities
        - MRR = mean(1/rank)

        Args:
            test_triples: [N, 3] int64 triples (pre-mapped by load_task_sequence).
            entity_to_id: Entity mapping (kept for API compat).
            relation_to_id: Relation mapping (kept for API compat).
            text_embeddings, mol_fingerprints, node_has_text, node_has_mol,
            edge_index, edge_type: Multimodal features.
            device: Device.
            batch_size: Evaluation batch size.
            known_triples: All known (h, r, t) triples for filtered ranking.
                If provided, known tails for each (h, r) are masked out.

        Returns:
            MRR score.
        """
        self.eval()
        import time as _et
        _stage = _et.time()
        mapped = self._map_triples(test_triples, entity_to_id, relation_to_id)
        if len(mapped) == 0:
            return 0.0
        # Sample-eval: cap at 50K test triples to match the baseline eval pipeline
        # (`src/baselines/_base.py::_score_triples`). Without this CMKL evaluates
        # the full 1.62M test set per call, taking ~10 min per call × 55 calls
        # per seed = ~9 h of eval per seed.
        eval_sample_cap = 50000
        if len(mapped) > eval_sample_cap:
            # Use the torch global RNG (already seeded by the training setup
            # via torch.manual_seed(seed)), matching the baseline pipeline in
            # src/baselines/_base.py::_score_triples. This makes different
            # experiment seeds sample different 50K subsets, which recovers
            # sample-selection variance and enables paired t-tests against
            # baselines on the same seed.
            sample_idx = torch.randperm(len(mapped))[:eval_sample_cap].cpu().numpy()
            mapped = mapped[sample_idx]
        logger.info(f"    [_evaluate_mrr] map+sample: {_et.time()-_stage:.2f}s ({len(mapped)} eval triples)")

        # Build filter: for each (h, r), collect known tail entities. Vectorized
        # numpy group-by — Python iteration over 8M+ rows was previously the
        # dominant cost on the first eval pass.
        _stage = _et.time()
        hr_to_tails: dict[tuple[int, int], set[int]] = {}
        if known_triples is not None and len(known_triples) > 0:
            kt = np.asarray(known_triples, dtype=np.int64)
            heads_np = kt[:, 0]
            rels_np = kt[:, 1]
            tails_np = kt[:, 2]
            for h, r, t in zip(heads_np.tolist(), rels_np.tolist(), tails_np.tolist()):
                key = (h, r)
                bucket = hr_to_tails.get(key)
                if bucket is None:
                    bucket = set()
                    hr_to_tails[key] = bucket
                bucket.add(t)
        logger.info(f"    [_evaluate_mrr] hr_to_tails build: {_et.time()-_stage:.2f}s ({len(hr_to_tails)} keys)")

        # Get embeddings
        _stage = _et.time()
        is_score_fusion = self.config["fusion_type"] == "score_fusion"
        if is_score_fusion:
            mod = self.forward_per_modality(
                edge_index, edge_type,
                text_embeddings, mol_fingerprints,
                node_has_text, node_has_mol,
            )
            h_struct = mod["h_struct"]
            h_text_full = mod["h_text_full"]
            h_mol_full = mod["h_mol_full"]
            alpha_text = self.config.get("score_fusion_alpha_text", 0.5)
            alpha_mol = self.config.get("score_fusion_alpha_mol", 0.3)
        else:
            h_fused = self.forward(
                edge_index, edge_type,
                text_embeddings, mol_fingerprints,
                node_has_text, node_has_mol,
            )
        logger.info(f"    [_evaluate_mrr] forward: {_et.time()-_stage:.2f}s")
        _stage = _et.time()

        ranks = []
        mapped_t = torch.tensor(mapped, dtype=torch.long, device=device)

        for start in range(0, len(mapped_t), batch_size):
            batch = mapped_t[start:start + batch_size]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            B = heads.shape[0]

            if is_score_fusion:
                # Score-level fusion: combine scores from all 3 modalities
                scores_struct = self._score_all_entities_modality(
                    h_struct[heads], rels, h_struct,
                    self.decoder, self.relation_emb,
                )
                scores_text = self._score_all_entities_modality(
                    h_text_full[heads], rels, h_text_full,
                    self.decoder_text, self.relation_emb_text,
                )
                scores_mol = self._score_all_entities_modality(
                    h_mol_full[heads], rels, h_mol_full,
                    self.decoder_mol, self.relation_emb_mol,
                )
                # Zero out text/mol scores for entities without that modality
                # Prevents spurious scores (especially TransE: -||h+r-0|| != 0)
                nht = mod["node_has_text"]
                nhm = mod["node_has_mol"]
                scores_text[:, ~nht] = 0.0
                scores_mol[:, ~nhm] = 0.0
                all_scores = scores_struct + alpha_text * scores_text + alpha_mol * scores_mol
            else:
                # Embedding-level fusion: single fused scoring
                h_embs = h_fused[heads]  # [B, D]
                if self.config["decoder_type"] == "Bilinear":
                    M = self.decoder.relation_matrices[rels]  # [B, D, D]
                    h_M = torch.einsum("bi,bij->bj", h_embs, M)  # [B, D]
                    all_scores = h_M @ h_fused.T  # [B, N]
                else:
                    r_embs = self.relation_emb(rels)  # [B, D]
                    if self.config["decoder_type"] == "TransE":
                        query = h_embs + r_embs  # [B, D]
                        all_scores = -torch.cdist(query, h_fused, p=self.decoder.p_norm)  # [B, N]
                    elif self.config["decoder_type"] == "RotatE":
                        # RotatE PyKEEN convention: entities are 2D real (= D complex
                        # pairs); relations are D phases mapped to unit-modulus complex.
                        # Score = -||h*r - t||_2. Closed-form via |h*r|^2 + |t|^2 -
                        # 2*Re((h*r) . conj(t)). With unit |r|, |h*r|^2 = |h|^2. The
                        # cross term reduces to two real matmuls (qr@tr.T + qi@ti.T),
                        # avoiding the multi-GB chunked subtraction.
                        D = self.decoder.embedding_dim
                        Bc = h_embs.shape[0]
                        N = h_fused.shape[0]
                        h_re = h_embs.reshape(Bc, D, 2)[:, :, 0]
                        h_im = h_embs.reshape(Bc, D, 2)[:, :, 1]
                        t_re = h_fused.reshape(N, D, 2)[:, :, 0]
                        t_im = h_fused.reshape(N, D, 2)[:, :, 1]
                        cos_r = torch.cos(r_embs)
                        sin_r = torch.sin(r_embs)
                        # query = h * r (complex multiply): qr = hr*cr - hi*sr; qi = hr*sr + hi*cr
                        qr = h_re * cos_r - h_im * sin_r
                        qi = h_re * sin_r + h_im * cos_r
                        norm_q_sq = (qr * qr + qi * qi).sum(dim=-1)  # [B]
                        norm_t_sq = (t_re * t_re + t_im * t_im).sum(dim=-1)  # [N]
                        inner_real = qr @ t_re.T + qi @ t_im.T  # [B, N]
                        dist_sq = norm_q_sq.unsqueeze(1) + norm_t_sq.unsqueeze(0) - 2.0 * inner_real
                        all_scores = -dist_sq.clamp_min(0).sqrt()
                    elif self.config["decoder_type"] == "ComplEx":
                        # ComplEx: score = Re(sum(h * r * conj(t))). Entities and
                        # relations are stored as 2D real (= D complex pairs).
                        # (h * r) @ conj(t).T works via real/imag expansion:
                        #   query = h * r (complex multiply)
                        #   <query, conj(t)>.real = qr @ tr.T + qi @ ti.T
                        D = self.decoder.embedding_dim
                        Bc = h_embs.shape[0]
                        N = h_fused.shape[0]
                        h_re = h_embs.reshape(Bc, D, 2)[:, :, 0]
                        h_im = h_embs.reshape(Bc, D, 2)[:, :, 1]
                        t_re = h_fused.reshape(N, D, 2)[:, :, 0]
                        t_im = h_fused.reshape(N, D, 2)[:, :, 1]
                        r_re = r_embs.reshape(Bc, D, 2)[:, :, 0]
                        r_im = r_embs.reshape(Bc, D, 2)[:, :, 1]
                        qr = h_re * r_re - h_im * r_im
                        qi = h_re * r_im + h_im * r_re
                        all_scores = qr @ t_re.T + qi @ t_im.T
                    else:
                        query = h_embs * r_embs  # [B, D]
                        all_scores = query @ h_fused.T  # [B, N]

            # Filtered ranking: mask out known tails except the true one
            if hr_to_tails:
                for b_idx in range(B):
                    h_val = heads[b_idx].item()
                    r_val = rels[b_idx].item()
                    t_val = tails[b_idx].item()
                    known = hr_to_tails.get((h_val, r_val), set())
                    if known:
                        mask_ids = [t for t in known if t != t_val]
                        if mask_ids:
                            all_scores[b_idx, mask_ids] = float("-inf")

            # Get rank of true tail
            true_scores = all_scores[torch.arange(B, device=device), tails]  # [B]
            # Count how many entities score >= true tail (1-based rank)
            batch_ranks = (all_scores >= true_scores.unsqueeze(1)).sum(dim=1).float()
            ranks.extend(batch_ranks.cpu().tolist())

        logger.info(f"    [_evaluate_mrr] eval loop ({len(mapped_t)//batch_size+1} batches): {_et.time()-_stage:.2f}s")

        if not ranks:
            return 0.0

        mrr = float(np.mean([1.0 / r for r in ranks]))
        return mrr

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint including CL state."""
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
        }
        if self.ewc is not None:
            state["ewc_state"] = self.ewc.state_dict()
        if self.replay_buffer is not None:
            state["replay_state"] = self.replay_buffer.state_dict()
        torch.save(state, str(path))
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> CMKL:
        """Load model from checkpoint."""
        state = torch.load(str(path), map_location=device)
        config = state["config"]
        config["num_entities"] = state["num_entities"]
        config["num_relations"] = state["num_relations"]
        model = cls(config)
        model.init_for_data(state["num_entities"], state["num_relations"])
        model.load_state_dict(state["model_state_dict"])
        if "ewc_state" in state:
            model.ewc = ModalityAwareEWC(model)
            model.ewc.load_state_dict(state["ewc_state"])
        if "replay_state" in state:
            model.replay_buffer = MultimodalMemoryBuffer()
            model.replay_buffer.load_state_dict(state["replay_state"])
        return model.to(device)
