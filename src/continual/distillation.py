"""Knowledge distillation for continual learning (optional CMKL enhancement).

If EWC + replay isn't sufficient, knowledge distillation preserves the old
model's "dark knowledge" about entity relationships. The old model (teacher)
generates soft targets on new task data, and the new model (student) learns
from both hard targets and soft targets.

Formula:
    L_soft = T^2 * KL(softmax(student_scores / T) || softmax(teacher_scores / T))
    L_total = alpha * L_hard + (1 - alpha) * L_soft

Usage:
    from src.continual.distillation import KnowledgeDistillation
    kd = KnowledgeDistillation(temperature=2.0, alpha=0.5)
    teacher = KnowledgeDistillation.create_teacher_copy(model)
    distill_loss = kd.compute_distillation_loss(student_scores, teacher_scores)
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KnowledgeDistillation:
    """Knowledge distillation from old model (teacher) to new model (student).

    The teacher is a frozen deep copy of the model after training on
    the previous task. During training on the new task, the student
    receives a combined loss: hard targets (ground truth) + soft targets
    (teacher's score distribution).

    Args:
        temperature: Softmax temperature for distillation. Higher T produces
            softer probability distributions, transferring more "dark knowledge".
        alpha: Weight for hard loss. (1 - alpha) weights the soft loss.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        self.temperature = temperature
        self.alpha = alpha

    @staticmethod
    def create_teacher_copy(model: nn.Module) -> nn.Module:
        """Create a frozen deep copy of the model to serve as teacher.

        Args:
            model: The model to copy (after finishing training on current task).

        Returns:
            Frozen copy of the model (all params require_grad=False).
        """
        teacher = copy.deepcopy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        logger.info("Created frozen teacher copy (%d params)",
                    sum(p.numel() for p in teacher.parameters()))
        return teacher

    def compute_distillation_loss(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL-divergence distillation loss between teacher and student.

        Both inputs should be raw scores (logits) over entities for a batch
        of queries. The method applies temperature-scaled softmax and computes
        KL divergence.

        L_soft = T^2 * KL(softmax(student / T) || softmax(teacher / T))

        Args:
            student_scores: [B, N] raw scores from the student model.
            teacher_scores: [B, N] raw scores from the frozen teacher model.

        Returns:
            Scalar distillation loss (soft target component only).
        """
        T = self.temperature

        # Temperature-scaled log-softmax for student, softmax for teacher
        student_log_probs = F.log_softmax(student_scores / T, dim=-1)
        teacher_probs = F.softmax(teacher_scores / T, dim=-1)

        # KL(P_teacher || P_student) = sum P_teacher * (log P_teacher - log P_student)
        # PyTorch KL expects (log_input, target) where target is the true distribution
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        # Scale by T^2 (gradient magnitude correction from Hinton et al. 2015)
        return (T ** 2) * kl_loss

    def compute_combined_loss(
        self,
        hard_loss: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined hard + soft distillation loss.

        L_total = alpha * L_hard + (1 - alpha) * L_soft

        Args:
            hard_loss: Scalar loss from the ground truth (task loss).
            student_scores: [B, N] raw student scores.
            teacher_scores: [B, N] raw teacher scores.

        Returns:
            Combined scalar loss.
        """
        soft_loss = self.compute_distillation_loss(student_scores, teacher_scores)
        total = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        return total
