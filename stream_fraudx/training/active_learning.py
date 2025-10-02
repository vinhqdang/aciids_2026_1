"""
Active Learning with Conformal Uncertainty for STREAM-FraudX
Selects most informative samples for labeling under budget constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class ConformalPredictor:
    """
    Conformal prediction for uncertainty quantification and calibration.

    Provides prediction sets with coverage guarantees and non-conformity scores.
    """
    def __init__(self,
                 miscoverage_rate: float = 0.1,
                 calibration_size: int = 1000):
        """
        Args:
            miscoverage_rate: Target miscoverage rate (1 - coverage)
            calibration_size: Size of calibration set
        """
        self.miscoverage_rate = miscoverage_rate
        self.calibration_size = calibration_size

        # Calibration set: stores (score, label, non_conformity)
        self.calibration_buffer = deque(maxlen=calibration_size)
        self.quantile = None

    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor):
        """
        Update calibration set with new predictions.

        Args:
            scores: (batch_size,) predicted probabilities
            labels: (batch_size,) true labels
        """
        # Compute non-conformity scores
        non_conformity = self._compute_non_conformity(scores, labels)

        # Add to calibration buffer
        for score, label, nc in zip(scores.tolist(), labels.tolist(), non_conformity.tolist()):
            self.calibration_buffer.append((score, label, nc))

        # Update quantile
        if len(self.calibration_buffer) >= 100:  # Minimum calibration size
            nc_scores = [item[2] for item in self.calibration_buffer]
            self.quantile = np.quantile(nc_scores, 1 - self.miscoverage_rate)

    def _compute_non_conformity(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute non-conformity scores.

        Higher non-conformity = more uncertain/surprising prediction.
        """
        # For binary classification: use absolute calibration error
        # Non-conformity = |score - label|
        return torch.abs(scores - labels.float())

    def get_uncertainty(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Get uncertainty estimates for predictions.

        Args:
            scores: (batch_size,) predicted probabilities

        Returns:
            uncertainty: (batch_size,) uncertainty scores (higher = more uncertain)
        """
        if self.quantile is None:
            # Fallback: use entropy-based uncertainty
            return self._entropy_uncertainty(scores)

        # Conformal uncertainty: distance from decision boundaries
        # considering calibrated quantile
        pos_nc = torch.abs(scores - 1.0)  # Non-conformity if true positive
        neg_nc = torch.abs(scores - 0.0)  # Non-conformity if true negative

        # Uncertainty is minimum non-conformity (most likely wrong)
        uncertainty = torch.min(pos_nc, neg_nc)

        return uncertainty

    def _entropy_uncertainty(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute entropy-based uncertainty."""
        p = scores.clamp(1e-8, 1 - 1e-8)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return entropy

    def predict_with_confidence(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence guarantees.

        Args:
            scores: (batch_size,) predicted probabilities

        Returns:
            predictions: (batch_size,) binary predictions
            confident: (batch_size,) boolean indicating if prediction is confident
        """
        if self.quantile is None:
            # No calibration yet, use default threshold
            predictions = (scores > 0.5).long()
            confident = torch.ones_like(predictions).bool()
            return predictions, confident

        # Predictions are confident if within calibrated quantile
        pos_nc = torch.abs(scores - 1.0)
        neg_nc = torch.abs(scores - 0.0)

        # Confident positive if pos_nc < quantile
        # Confident negative if neg_nc < quantile
        confident_pos = (scores > 0.5) & (pos_nc < self.quantile)
        confident_neg = (scores <= 0.5) & (neg_nc < self.quantile)

        predictions = (scores > 0.5).long()
        confident = confident_pos | confident_neg

        return predictions, confident


class ActiveLearner:
    """
    Active learning module for selecting samples to label.

    Combines conformal uncertainty with business cost to prioritize labeling.
    """
    def __init__(self,
                 conformal_predictor: ConformalPredictor,
                 use_business_cost: bool = True,
                 diversity_weight: float = 0.1):
        """
        Args:
            conformal_predictor: Conformal predictor for uncertainty
            use_business_cost: Whether to incorporate transaction amounts
            diversity_weight: Weight for diversity in selection
        """
        self.conformal = conformal_predictor
        self.use_business_cost = use_business_cost
        self.diversity_weight = diversity_weight

        # Track recently selected samples for diversity
        self.recent_selections = deque(maxlen=1000)

    def select_for_labeling(self,
                           scores: torch.Tensor,
                           batch_data: Dict[str, torch.Tensor],
                           budget: int) -> List[int]:
        """
        Select samples to label from a batch.

        Args:
            scores: (batch_size,) predicted fraud probabilities
            batch_data: Dictionary containing batch features
            budget: Number of labels available

        Returns:
            selected_indices: List of indices to label
        """
        if budget <= 0:
            return []

        batch_size = scores.size(0)

        # Compute uncertainty scores
        uncertainty = self.conformal.get_uncertainty(scores)

        # Compute acquisition scores
        acquisition_scores = self._compute_acquisition_scores(
            scores, uncertainty, batch_data
        )

        # Select top-k samples
        k = min(budget, batch_size)
        _, top_indices = torch.topk(acquisition_scores, k)

        selected = top_indices.tolist()

        # Update recent selections
        if 'src_nodes' in batch_data:
            self.recent_selections.extend(batch_data['src_nodes'][selected].tolist())

        return selected

    def _compute_acquisition_scores(self,
                                    scores: torch.Tensor,
                                    uncertainty: torch.Tensor,
                                    batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute acquisition scores for active learning.

        Combines:
        1. Conformal uncertainty (informativeness)
        2. Business cost (high-value transactions)
        3. Diversity (avoid similar samples)
        """
        acquisition = uncertainty.clone()

        # Add business cost component
        if self.use_business_cost and 'continuous' in batch_data:
            # Assume first continuous feature is transaction amount
            amounts = batch_data['continuous'][:, 0]
            # Normalize amounts
            amounts_norm = (amounts - amounts.mean()) / (amounts.std() + 1e-8)
            # High-value transactions are prioritized
            business_cost = torch.sigmoid(amounts_norm)

            acquisition = acquisition * (1 + business_cost)

        # Add diversity component
        if self.diversity_weight > 0 and 'src_nodes' in batch_data:
            diversity = self._compute_diversity_scores(batch_data['src_nodes'])
            acquisition = acquisition + self.diversity_weight * diversity

        return acquisition

    def _compute_diversity_scores(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity scores - higher for nodes not recently selected.
        """
        diversity_scores = torch.ones_like(node_ids, dtype=torch.float)

        if len(self.recent_selections) == 0:
            return diversity_scores

        # Penalize recently selected nodes
        recent_set = set(self.recent_selections)
        for i, node_id in enumerate(node_ids.tolist()):
            if node_id in recent_set:
                diversity_scores[i] *= 0.1  # Strong penalty

        return diversity_scores


class UncertaintySampler:
    """
    Simple uncertainty sampling baseline (for comparison).
    """
    def __init__(self, method: str = 'entropy'):
        """
        Args:
            method: 'entropy', 'margin', or 'least_confident'
        """
        self.method = method

    def select(self, scores: torch.Tensor, budget: int) -> List[int]:
        """
        Select samples based on uncertainty.

        Args:
            scores: (batch_size,) predicted probabilities
            budget: Number of samples to select

        Returns:
            selected_indices: List of indices
        """
        if self.method == 'entropy':
            uncertainty = self._entropy(scores)
        elif self.method == 'margin':
            uncertainty = self._margin(scores)
        else:  # least_confident
            uncertainty = self._least_confident(scores)

        k = min(budget, scores.size(0))
        _, top_indices = torch.topk(uncertainty, k)

        return top_indices.tolist()

    def _entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """Entropy-based uncertainty."""
        p = scores.clamp(1e-8, 1 - 1e-8)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    def _margin(self, scores: torch.Tensor) -> torch.Tensor:
        """Margin-based uncertainty (distance from decision boundary)."""
        return -torch.abs(scores - 0.5)  # Negative because we want min distance

    def _least_confident(self, scores: torch.Tensor) -> torch.Tensor:
        """Least confident sampling."""
        confidence = torch.max(scores, 1 - scores)
        return -confidence  # Negative for consistency with topk


class RandomSampler:
    """
    Random sampling baseline.
    """
    def select(self, scores: torch.Tensor, budget: int) -> List[int]:
        """
        Randomly select samples.

        Args:
            scores: (batch_size,) predicted probabilities (unused)
            budget: Number of samples to select

        Returns:
            selected_indices: List of indices
        """
        batch_size = scores.size(0)
        k = min(budget, batch_size)
        indices = torch.randperm(batch_size)[:k]
        return indices.tolist()
