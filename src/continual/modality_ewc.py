"""Modality-Aware Elastic Weight Consolidation for CMKL.

EWC with separate Fisher matrices per modality. Key insight: different
modalities experience different distribution shifts across tasks. Molecular
structures are stable; text descriptions evolve; graph structure changes
with new entities. Per-modality Fisher matrices allow modality-specific
regularization strength.

Default lambdas: lambda_struct=10.0, lambda_text=5.0, lambda_mol=1.0

Usage:
    from src.continual.modality_ewc import ModalityAwareEWC
    ewc = ModalityAwareEWC(model, lambda_struct=10.0, lambda_text=5.0, lambda_mol=1.0)
    ewc.compute_modality_fisher(dataloader)
    penalty = ewc.ewc_loss()
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Mapping from modality name to the expected encoder attribute on the CMKL model
MODALITY_ENCODER_MAP = {
    "structural": "structural_encoder",
    "textual": "textual_encoder",
    "molecular": "molecular_encoder",
    "fusion": "fusion",
    # Main (fused) decoder and relation embeddings. Critical for RotatE where
    # relation phases ARE rotation angles — unprotected drift between tasks
    # warps the entire embedding geometry and silently destroys continual
    # performance. Previously these were at the CMKL root and silently
    # excluded from any Fisher bucket.
    "relation": "relation_emb",
    # Score-level fusion per-modality decoders (None when not using score_fusion)
    "text_decoder": "decoder_text",
    "text_relation": "relation_emb_text",
    "mol_decoder": "decoder_mol",
    "mol_relation": "relation_emb_mol",
}


class ModalityAwareEWC:
    """EWC with separate Fisher matrices per modality.

    Computes and maintains separate Fisher information matrices for each
    modality's encoder parameters. During training on subsequent tasks,
    applies modality-specific regularization penalties to prevent forgetting.

    Args:
        model: The CMKL model with named encoders (structural_encoder,
            textual_encoder, molecular_encoder).
        lambda_struct: Regularization strength for structural encoder.
            Graph structure changes most with new entities.
        lambda_text: Regularization strength for textual encoder.
            Text descriptions may evolve but less dramatically.
        lambda_mol: Regularization strength for molecular encoder.
            Molecular structures are relatively stable.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_struct: float = 10.0,
        lambda_text: float = 5.0,
        lambda_mol: float = 1.0,
        lambda_fusion: float = 5.0,
        lambda_relation: float = 10.0,
    ) -> None:
        self.model = model
        self.lambdas = {
            "structural": lambda_struct,
            "textual": lambda_text,
            "molecular": lambda_mol,
            "fusion": lambda_fusion,
            # Main relation embeddings — high lambda since RotatE relations
            # are rotation angles and any drift warps the whole geometry.
            "relation": lambda_relation,
            # Score-level fusion decoders (reuse modality lambdas)
            "text_decoder": lambda_text,
            "text_relation": lambda_text,
            "mol_decoder": lambda_mol,
            "mol_relation": lambda_mol,
        }
        # Fisher diagonal per modality: {modality: {param_name: tensor}}
        self.fisher_per_modality: dict[str, dict[str, torch.Tensor]] = {}
        # Reference params (after training on task t): {modality: {param_name: tensor}}
        self.old_params_per_modality: dict[str, dict[str, torch.Tensor]] = {}

    def _get_encoder(self, modality: str) -> nn.Module | None:
        """Get the encoder module for a modality, or None if it doesn't exist."""
        attr_name = MODALITY_ENCODER_MAP.get(modality)
        if attr_name and hasattr(self.model, attr_name):
            return getattr(self.model, attr_name)
        return None

    def _get_trainable_params(self, encoder: nn.Module) -> dict[str, nn.Parameter]:
        """Get trainable parameters from an encoder."""
        return {
            name: param
            for name, param in encoder.named_parameters()
            if param.requires_grad
        }

    def compute_modality_fisher(
        self,
        compute_loss_fn: callable,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        num_samples: int = 200,
    ) -> None:
        """Compute separate Fisher diagonals for each encoder's parameters.

        Uses the empirical Fisher information matrix approximation:
        F_k = E[grad(log p(y|x, theta))^2] for each parameter k.

        In practice, we compute gradients of the task loss on a subset of
        training data and accumulate squared gradients as Fisher estimates.

        After computation, stores both Fisher matrices and current parameter
        values as reference for computing EWC penalties on future tasks.

        Args:
            compute_loss_fn: A callable that takes a batch dict and returns
                a scalar loss tensor. Should be the task loss (not total loss).
            dataloader: DataLoader for the current task's training data.
            device: Device for computation.
            num_samples: Max number of samples for Fisher estimation.
        """
        self.model.eval()

        # Initialize Fisher accumulators for each modality
        fisher_accum: dict[str, dict[str, torch.Tensor]] = {}
        for modality in MODALITY_ENCODER_MAP:
            encoder = self._get_encoder(modality)
            if encoder is None:
                continue
            params = self._get_trainable_params(encoder)
            if not params:
                continue
            fisher_accum[modality] = {
                name: torch.zeros_like(param)
                for name, param in params.items()
            }

        # Accumulate squared gradients over dataloader samples
        n_samples = 0
        for batch in dataloader:
            if n_samples >= num_samples:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            self.model.zero_grad()
            loss = compute_loss_fn(batch)
            loss.backward()

            # Accumulate squared gradients per modality
            for modality in fisher_accum:
                encoder = self._get_encoder(modality)
                for name, param in self._get_trainable_params(encoder).items():
                    if param.grad is not None:
                        fisher_accum[modality][name] += param.grad.detach() ** 2

            n_samples += 1

        if n_samples == 0:
            logger.warning("No samples processed for Fisher computation")
            return

        # Average and accumulate into running Fisher
        for modality in fisher_accum:
            for name in fisher_accum[modality]:
                fisher_accum[modality][name] /= n_samples

            # Accumulate across tasks (sum Fisher matrices from all previous tasks)
            if modality not in self.fisher_per_modality:
                self.fisher_per_modality[modality] = fisher_accum[modality]
            else:
                for name in fisher_accum[modality]:
                    if name in self.fisher_per_modality[modality]:
                        self.fisher_per_modality[modality][name] += fisher_accum[modality][name]
                    else:
                        self.fisher_per_modality[modality][name] = fisher_accum[modality][name]

        # Store current parameter values as reference
        for modality in MODALITY_ENCODER_MAP:
            encoder = self._get_encoder(modality)
            if encoder is None:
                continue
            params = self._get_trainable_params(encoder)
            self.old_params_per_modality[modality] = {
                name: param.detach().clone()
                for name, param in params.items()
            }

        logger.info(
            "Computed per-modality Fisher: %s",
            {m: len(p) for m, p in self.fisher_per_modality.items()},
        )

    def ewc_loss(self) -> torch.Tensor:
        """Compute modality-weighted EWC penalty.

        total = sum over modalities of:
            lambda_m * sum_k(F_k * (theta_k - theta*_k)^2) / 2

        Returns:
            Scalar tensor with total modality-aware EWC penalty.
            Returns 0 if no Fisher information has been computed yet.
        """
        if not self.fisher_per_modality:
            # No previous tasks - no penalty
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        total_penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for modality, fisher_dict in self.fisher_per_modality.items():
            lam = self.lambdas.get(modality, 1.0)
            encoder = self._get_encoder(modality)
            if encoder is None:
                continue

            old_params = self.old_params_per_modality.get(modality, {})
            current_params = self._get_trainable_params(encoder)

            modality_penalty = torch.tensor(0.0, device=total_penalty.device)
            for name, fisher in fisher_dict.items():
                if name in current_params and name in old_params:
                    diff = current_params[name] - old_params[name]
                    modality_penalty = modality_penalty + (fisher * diff ** 2).sum()

            total_penalty = total_penalty + lam * modality_penalty

        return total_penalty / 2.0

    def state_dict(self) -> dict:
        """Serialize EWC state for checkpointing."""
        return {
            "fisher_per_modality": {
                m: {n: f.cpu() for n, f in fd.items()}
                for m, fd in self.fisher_per_modality.items()
            },
            "old_params_per_modality": {
                m: {n: p.cpu() for n, p in pd.items()}
                for m, pd in self.old_params_per_modality.items()
            },
            "lambdas": self.lambdas,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EWC state from checkpoint."""
        device = next(self.model.parameters()).device
        self.lambdas = state["lambdas"]
        self.fisher_per_modality = {
            m: {n: f.to(device) for n, f in fd.items()}
            for m, fd in state["fisher_per_modality"].items()
        }
        self.old_params_per_modality = {
            m: {n: p.to(device) for n, p in pd.items()}
            for m, pd in state["old_params_per_modality"].items()
        }
