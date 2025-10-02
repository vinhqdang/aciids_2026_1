"""
Invariant Risk Minimization (IRM) Loss for drift-aware learning
Helps model learn stable representations across temporal distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class IRMLiteLoss(nn.Module):
    """
    Lightweight IRM penalty for temporal stability.

    Penalizes variance of class-conditional feature means across time slices.
    Encourages learning representations invariant to temporal shifts.

    Reference: "Invariant Risk Minimization" (Arjovsky et al.)
    """
    def __init__(self, penalty_weight: float = 0.1, penalty_anneal: str = 'constant'):
        """
        Args:
            penalty_weight: Weight for IRM penalty term
            penalty_anneal: Annealing strategy ('constant', 'linear', 'cosine')
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.penalty_anneal = penalty_anneal
        self.current_epoch = 0

    def forward(self,
                model_output: torch.Tensor,
                targets: torch.Tensor,
                time_slices: torch.Tensor) -> torch.Tensor:
        """
        Compute IRM penalty based on temporal slices.

        Args:
            model_output: (batch_size, feature_dim) model representations
            targets: (batch_size,) binary labels
            time_slices: (batch_size,) time bucket assignments

        Returns:
            penalty: IRM penalty scalar
        """
        device = model_output.device
        unique_slices = time_slices.unique()

        if len(unique_slices) < 2:
            # Need at least 2 time slices for IRM
            return torch.tensor(0.0, device=device)

        # Compute mean representations per class per time slice
        class_means_per_slice = []

        for time_id in unique_slices:
            time_mask = time_slices == time_id

            # Positive class mean
            pos_mask = (targets == 1) & time_mask
            if pos_mask.sum() > 0:
                pos_mean = model_output[pos_mask].mean(dim=0)
            else:
                pos_mean = torch.zeros(model_output.size(1), device=device)

            # Negative class mean
            neg_mask = (targets == 0) & time_mask
            if neg_mask.sum() > 0:
                neg_mean = model_output[neg_mask].mean(dim=0)
            else:
                neg_mean = torch.zeros(model_output.size(1), device=device)

            class_means_per_slice.append((pos_mean, neg_mean))

        # Compute variance across time slices for each class
        pos_means = torch.stack([m[0] for m in class_means_per_slice])
        neg_means = torch.stack([m[1] for m in class_means_per_slice])

        # Penalty: variance of means across time
        pos_variance = pos_means.var(dim=0).mean()
        neg_variance = neg_means.var(dim=0).mean()

        penalty = (pos_variance + neg_variance) / 2

        # Apply penalty weight with annealing
        weight = self._get_annealed_weight()
        return weight * penalty

    def _get_annealed_weight(self) -> float:
        """Get annealed penalty weight based on current epoch."""
        if self.penalty_anneal == 'constant':
            return self.penalty_weight
        elif self.penalty_anneal == 'linear':
            # Linearly increase from 0 to penalty_weight over first epochs
            return min(self.penalty_weight, self.penalty_weight * self.current_epoch / 10)
        elif self.penalty_anneal == 'cosine':
            # Cosine annealing
            import math
            return self.penalty_weight * (1 - math.cos(math.pi * min(self.current_epoch / 20, 1))) / 2
        else:
            return self.penalty_weight

    def step_epoch(self):
        """Increment epoch counter for annealing."""
        self.current_epoch += 1


class IRMv1Penalty(nn.Module):
    """
    Original IRM penalty based on gradient variance.

    More computationally expensive but theoretically grounded.
    """
    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                environment_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute IRM penalty across environments.

        Args:
            logits: (batch_size,) model predictions
            targets: (batch_size,) binary labels
            environment_ids: (batch_size,) environment/time slice IDs

        Returns:
            penalty: IRM penalty scalar
        """
        device = logits.device
        unique_envs = environment_ids.unique()

        if len(unique_envs) < 2:
            return torch.tensor(0.0, device=device)

        penalties = []
        dummy_w = torch.tensor(1.0, requires_grad=True, device=device)

        for env_id in unique_envs:
            env_mask = environment_ids == env_id
            env_logits = logits[env_mask]
            env_targets = targets[env_mask]

            if env_logits.numel() == 0:
                continue

            # Scale logits by dummy parameter
            scaled_logits = env_logits * dummy_w

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(
                scaled_logits, env_targets.float(), reduction='mean'
            )

            # Compute gradient
            grad = torch.autograd.grad(loss, dummy_w, create_graph=True)[0]
            penalties.append(grad ** 2)

        if len(penalties) == 0:
            return torch.tensor(0.0, device=device)

        # Mean of squared gradients
        penalty = torch.stack(penalties).mean()
        return self.penalty_weight * penalty


class TemporalStabilityLoss(nn.Module):
    """
    Temporal stability loss using contrastive learning.

    Encourages similar representations for temporally close samples.
    """
    def __init__(self, temperature: float = 0.5, time_window: float = 3600.0):
        """
        Args:
            temperature: Temperature for contrastive loss
            time_window: Time window (seconds) for considering samples as similar
        """
        super().__init__()
        self.temperature = temperature
        self.time_window = time_window

    def forward(self,
                representations: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal stability loss.

        Args:
            representations: (batch_size, dim) feature representations
            timestamps: (batch_size,) timestamps

        Returns:
            loss: temporal stability loss
        """
        batch_size = representations.size(0)
        device = representations.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize representations
        representations = F.normalize(representations, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(representations, representations.T) / self.temperature

        # Compute temporal distance matrix
        time_diff = (timestamps.unsqueeze(1) - timestamps.unsqueeze(0)).abs()
        temporal_mask = (time_diff < self.time_window).float()

        # Remove self-similarities
        mask = torch.eye(batch_size, device=device)
        temporal_mask = temporal_mask * (1 - mask)

        # Compute loss: maximize similarity for temporally close samples
        positive_pairs = temporal_mask.sum(dim=1)
        has_positives = positive_pairs > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device)

        # InfoNCE-style loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        loss = -(log_prob * temporal_mask).sum(dim=1) / (positive_pairs + 1e-8)
        loss = loss[has_positives].mean()

        return loss
