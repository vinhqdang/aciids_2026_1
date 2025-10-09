"""
Combined loss functions with IRM penalties and label-aware sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .focal_losses import AsymmetricFocalLoss
from .irm_loss import IRMLoss


class CombinedFocalLoss(nn.Module):
    """
    Combined Focal Loss with multiple components:
    - Asymmetric Focal Loss for imbalance
    - Optional IRM penalty for domain invariance
    - Optional auxiliary losses (contrastive, reconstruction)
    """

    def __init__(self,
                 focal_gamma_pos: float = 0.0,
                 focal_gamma_neg: float = 2.0,
                 focal_alpha: float = 0.25,
                 use_irm: bool = False,
                 irm_penalty_weight: float = 0.1,
                 irm_penalty_anneal_iters: int = 500):
        super().__init__()

        # Main focal loss
        self.focal_loss = AsymmetricFocalLoss(
            gamma_pos=focal_gamma_pos,
            gamma_neg=focal_gamma_neg,
            alpha=focal_alpha
        )

        # IRM loss for domain invariance
        self.use_irm = use_irm
        if use_irm:
            self.irm_loss = IRMLoss(
                penalty_weight=irm_penalty_weight,
                penalty_anneal_iters=irm_penalty_anneal_iters
            )

        self.iteration = 0

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                domains: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: (batch_size,) predicted logits
            targets: (batch_size,) binary labels
            domains: (batch_size,) domain IDs (optional, for IRM)

        Returns:
            Dictionary with 'total_loss' and component losses
        """
        # Main focal loss
        focal = self.focal_loss(logits, targets)

        losses = {
            'focal_loss': focal,
            'total_loss': focal
        }

        # Add IRM penalty if enabled
        if self.use_irm and domains is not None:
            irm_penalty = self.irm_loss(logits, targets, domains, self.iteration)
            losses['irm_penalty'] = irm_penalty
            losses['total_loss'] = losses['total_loss'] + irm_penalty

        self.iteration += 1

        return losses


class LabelAwareSampler:
    """
    Label-aware sampling for handling class imbalance.

    Implements weighted sampling to oversample minority class (frauds).
    """

    def __init__(self,
                 labels: torch.Tensor,
                 oversample_ratio: float = 1.0):
        """
        Args:
            labels: (num_samples,) binary labels
            oversample_ratio: How much to oversample minority class (1.0 = balanced)
        """
        self.labels = labels
        self.oversample_ratio = oversample_ratio

        # Compute class weights
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 0).sum().item()
        total = len(labels)

        # Weight inversely proportional to class frequency
        if pos_count > 0 and neg_count > 0:
            self.pos_weight = (total / (2 * pos_count)) * oversample_ratio
            self.neg_weight = total / (2 * neg_count)
        else:
            self.pos_weight = 1.0
            self.neg_weight = 1.0

    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for weighted sampling.

        Returns:
            weights: (num_samples,) sampling weights
        """
        weights = torch.ones_like(self.labels, dtype=torch.float)
        weights[self.labels == 1] = self.pos_weight
        weights[self.labels == 0] = self.neg_weight

        return weights

    def get_weighted_sampler(self):
        """Get PyTorch WeightedRandomSampler."""
        from torch.utils.data import WeightedRandomSampler

        weights = self.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        return sampler
