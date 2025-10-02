"""
Focal Loss implementations for STREAM-FraudX
Includes Asymmetric Focal Loss and Focal Tversky Loss for imbalanced fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for imbalanced classification.

    Different focusing parameters for positive and negative samples.
    Useful when false negatives are more costly than false positives.

    Reference: "Asymmetric Loss For Multi-Label Classification"
    """
    def __init__(self,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean'):
        """
        Args:
            gamma_pos: Focusing parameter for positive samples (frauds)
            gamma_neg: Focusing parameter for negative samples (normal)
            alpha: Weighting factor for positive class
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) predicted logits
            targets: (batch_size,) binary labels {0, 1}
        Returns:
            loss: scalar or (batch_size,) depending on reduction
        """
        probs = torch.sigmoid(logits)

        # Positive samples (frauds)
        pos_mask = targets == 1
        pos_probs = probs[pos_mask]
        if pos_probs.numel() > 0:
            pos_loss = -self.alpha * (1 - pos_probs) ** self.gamma_pos * torch.log(pos_probs + 1e-8)
        else:
            pos_loss = torch.tensor(0.0, device=logits.device)

        # Negative samples (normal)
        neg_mask = targets == 0
        neg_probs = probs[neg_mask]
        if neg_probs.numel() > 0:
            neg_loss = -(1 - self.alpha) * neg_probs ** self.gamma_neg * torch.log(1 - neg_probs + 1e-8)
        else:
            neg_loss = torch.tensor(0.0, device=logits.device)

        # Combine losses
        loss = torch.zeros_like(logits)
        loss[pos_mask] = pos_loss
        loss[neg_mask] = neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling imbalanced data.

    Combines Tversky Index (generalization of Dice) with focal mechanism.
    Controls trade-off between false positives and false negatives.

    Reference: "Focal Tversky loss function for imbalanced data"
    """
    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 gamma: float = 1.0,
                 smooth: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for false negatives (higher = penalize FN more)
            beta: Weight for false positives (higher = penalize FP more)
            gamma: Focal parameter (higher = focus on hard examples)
            smooth: Smoothing constant
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        assert abs(alpha + beta - 1.0) < 1e-5, "alpha + beta should equal 1"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) predicted logits
            targets: (batch_size,) binary labels {0, 1}
        Returns:
            loss: scalar or (batch_size,) depending on reduction
        """
        probs = torch.sigmoid(logits)

        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        # Focal Tversky loss
        loss = (1 - tversky_index) ** self.gamma

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss
        else:
            return loss.expand(logits.size(0))


class CombinedFocalLoss(nn.Module):
    """
    Combined loss: Asymmetric Focal Loss + Focal Tversky Loss.

    This is the main loss used in STREAM-FraudX for supervised training.
    """
    def __init__(self,
                 afl_weight: float = 1.0,
                 tversky_weight: float = 0.5,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 2.0,
                 alpha_afl: float = 0.25,
                 alpha_tversky: float = 0.7,
                 beta_tversky: float = 0.3,
                 gamma_tversky: float = 1.0):
        """
        Args:
            afl_weight: Weight for asymmetric focal loss
            tversky_weight: Weight for focal tversky loss
            Other args: See individual loss classes
        """
        super().__init__()

        self.afl = AsymmetricFocalLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            alpha=alpha_afl
        )

        self.tversky = FocalTverskyLoss(
            alpha=alpha_tversky,
            beta=beta_tversky,
            gamma=gamma_tversky
        )

        self.afl_weight = afl_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) predicted logits
            targets: (batch_size,) binary labels {0, 1}
        Returns:
            loss: scalar combined loss
        """
        loss_afl = self.afl(logits, targets)
        loss_tversky = self.tversky(logits, targets)

        return self.afl_weight * loss_afl + self.tversky_weight * loss_tversky


class BinaryFocalLoss(nn.Module):
    """
    Standard binary focal loss (for comparison).

    Reference: "Focal Loss for Dense Object Detection"
    """
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) predicted logits
            targets: (batch_size,) binary labels {0, 1}
        Returns:
            loss: scalar or (batch_size,) depending on reduction
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        probs = torch.sigmoid(logits)

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
