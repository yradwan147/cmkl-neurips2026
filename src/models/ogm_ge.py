"""On-the-fly Gradient Modulation with Generalization Enhancement (OGM-GE).

Balances multimodal learning by monitoring per-modality convergence speeds
and scaling gradient contributions accordingly. Prevents dominant modalities
(e.g., jointly-optimized structural encoder) from suppressing weaker ones
(e.g., frozen text projections).

Reference: Peng et al. (CVPR 2022) — "Balanced Multimodal Learning via
On-the-fly Gradient Modulation"

Usage:
    from src.models.ogm_ge import OGMGEModulator
    ogm = OGMGEModulator(modality_names=["struct", "text", "mol"], alpha=1.0)
    weights = ogm.compute_modulation_weights({"struct": 0.5, "text": 0.8, "mol": 0.9})
    # weights = {"struct": 1.0, "text": 1.3, "mol": 1.4}  (boost lagging modalities)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OGMGEModulator:
    """On-the-fly Gradient Modulation with Generalization Enhancement.

    Tracks per-modality loss values across epochs and computes gradient
    scaling factors to balance convergence speeds. Modalities converging
    faster get dampened; slower ones get boosted.

    Args:
        modality_names: Names of modalities to track (e.g., ["struct", "text", "mol"]).
        alpha: Modulation strength. Higher = more aggressive balancing.
            0.0 = no modulation (all weights = 1.0).
    """

    def __init__(
        self,
        modality_names: list[str] | None = None,
        alpha: float = 1.0,
    ) -> None:
        if modality_names is None:
            modality_names = ["struct", "text", "mol"]
        self.modality_names = modality_names
        self.alpha = alpha
        self.prev_losses: dict[str, float] = {}

    def compute_modulation_weights(
        self,
        current_losses: dict[str, float],
    ) -> dict[str, float]:
        """Compute per-modality gradient scaling factors.

        Compares each modality's loss ratio (current/previous) to the mean
        ratio. Modalities with faster convergence (lower ratio) get dampened;
        slower ones get boosted.

        Args:
            current_losses: Dict mapping modality name to current loss value.
                Modalities with loss <= 0 are skipped (weight = 1.0).

        Returns:
            Dict mapping modality name to gradient scaling factor (>= 0).
        """
        weights = {name: 1.0 for name in self.modality_names}

        if not self.prev_losses or self.alpha == 0.0:
            # First epoch or disabled: store losses and return uniform weights
            for name in self.modality_names:
                if name in current_losses and current_losses[name] > 0:
                    self.prev_losses[name] = current_losses[name]
            return weights

        # Compute convergence ratios: current_loss / prev_loss
        # Ratio < 1 means converging (loss decreasing), > 1 means diverging
        ratios: dict[str, float] = {}
        for name in self.modality_names:
            curr = current_losses.get(name, 0.0)
            prev = self.prev_losses.get(name, 0.0)
            if curr > 0 and prev > 0:
                ratios[name] = curr / prev
            # Skip modalities with zero loss (no qualifying triples)

        if len(ratios) < 2:
            # Need at least 2 active modalities to modulate
            for name in self.modality_names:
                if name in current_losses and current_losses[name] > 0:
                    self.prev_losses[name] = current_losses[name]
            return weights

        # Mean convergence ratio across active modalities
        mean_ratio = sum(ratios.values()) / len(ratios)

        # OGM-GE: scale based on relative convergence speed
        # Fast-converging (ratio < mean) → dampen (weight < 1)
        # Slow-converging (ratio > mean) → boost (weight > 1)
        for name, ratio in ratios.items():
            if mean_ratio > 0:
                relative = ratio / mean_ratio
                # Clamp to prevent extreme values
                modulation = 1.0 + self.alpha * (relative - 1.0)
                weights[name] = max(0.1, min(modulation, 5.0))

        # Update stored losses
        for name in self.modality_names:
            if name in current_losses and current_losses[name] > 0:
                self.prev_losses[name] = current_losses[name]

        return weights

    def reset(self) -> None:
        """Reset loss history (call between tasks in continual learning)."""
        self.prev_losses.clear()
