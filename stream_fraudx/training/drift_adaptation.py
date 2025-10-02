"""
Drift-Aware Online Adaptation for STREAM-FraudX
Implements Meta-Adapter with Reptile-style updates and streaming adaptation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from collections import deque
import copy


class MetaAdapterOptimizer:
    """
    Meta-learning optimizer for adapters using Reptile algorithm.

    Reptile performs fast adaptation by interpolating between
    meta-initialization and task-specific weights.
    """
    def __init__(self,
                 adapter_params: List[nn.Parameter],
                 meta_lr: float = 0.01,
                 inner_lr: float = 0.001,
                 inner_steps: int = 5):
        """
        Args:
            adapter_params: List of adapter parameters to meta-learn
            meta_lr: Meta-learning rate for Reptile updates
            inner_lr: Learning rate for inner loop optimization
            inner_steps: Number of gradient steps per adaptation
        """
        self.adapter_params = list(adapter_params)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # Store meta-initialization
        self.meta_params = [p.clone().detach() for p in self.adapter_params]

        # Inner optimizer
        self.inner_optimizer = optim.Adam(self.adapter_params, lr=inner_lr)

    def adapt(self, loss_fn, data_batch, model):
        """
        Perform fast adaptation on a data batch.

        Args:
            loss_fn: Loss function
            data_batch: Batch of data for adaptation
            model: The model containing adapters
        """
        # Save current state
        initial_params = [p.clone() for p in self.adapter_params]

        # Inner loop: gradient descent on task
        for _ in range(self.inner_steps):
            self.inner_optimizer.zero_grad()
            loss = loss_fn(model, data_batch)
            loss.backward()
            self.inner_optimizer.step()

        # Meta-update: move meta-params toward adapted params (Reptile)
        with torch.no_grad():
            for meta_p, adapted_p in zip(self.meta_params, self.adapter_params):
                meta_p.add_((adapted_p - meta_p) * self.meta_lr)

    def reset_to_meta(self):
        """Reset adapter parameters to meta-initialization."""
        with torch.no_grad():
            for p, meta_p in zip(self.adapter_params, self.meta_params):
                p.copy_(meta_p)

    def save_meta_state(self, path: str):
        """Save meta-initialization to disk."""
        torch.save([p.cpu() for p in self.meta_params], path)

    def load_meta_state(self, path: str):
        """Load meta-initialization from disk."""
        loaded = torch.load(path)
        for meta_p, loaded_p in zip(self.meta_params, loaded):
            meta_p.copy_(loaded_p.to(meta_p.device))


class StreamingAdaptation:
    """
    Online adaptation manager for streaming fraud detection.

    Handles micro-batch processing, label collection, and adapter updates.
    """
    def __init__(self,
                 model: nn.Module,
                 meta_optimizer: MetaAdapterOptimizer,
                 loss_fn: nn.Module,
                 daily_label_budget: int = 100,
                 microbatch_seconds: int = 30,
                 adaptation_frequency: int = 10):
        """
        Args:
            model: The fraud detection model
            meta_optimizer: Meta-adapter optimizer
            loss_fn: Loss function for adaptation
            daily_label_budget: Maximum labels to query per day
            microbatch_seconds: Size of processing micro-batches (seconds)
            adaptation_frequency: Adapt every N micro-batches
        """
        self.model = model
        self.meta_optimizer = meta_optimizer
        self.loss_fn = loss_fn
        self.daily_label_budget = daily_label_budget
        self.microbatch_seconds = microbatch_seconds
        self.adaptation_frequency = adaptation_frequency

        # Tracking
        self.labels_used_today = 0
        self.microbatch_count = 0
        self.labeled_buffer = deque(maxlen=1000)  # Store recent labeled samples

    def process_microbatch(self,
                          microbatch: Dict[str, torch.Tensor],
                          active_learner = None) -> torch.Tensor:
        """
        Process a micro-batch of events.

        Args:
            microbatch: Batch of events
            active_learner: Active learning module for label selection

        Returns:
            scores: Fraud scores for the micro-batch
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(microbatch)

        # Select samples for labeling if budget allows
        if active_learner and self.labels_used_today < self.daily_label_budget:
            selected_indices = active_learner.select_for_labeling(
                scores, microbatch,
                budget=self.daily_label_budget - self.labels_used_today
            )

            if selected_indices:
                # In practice, this would trigger a labeling request
                # For now, simulate with existing labels if available
                if 'labels' in microbatch:
                    labeled_samples = {
                        k: v[selected_indices] if torch.is_tensor(v) else v
                        for k, v in microbatch.items()
                    }
                    self.labeled_buffer.append(labeled_samples)
                    self.labels_used_today += len(selected_indices)

        # Periodically adapt using labeled buffer
        self.microbatch_count += 1
        if self.microbatch_count % self.adaptation_frequency == 0:
            self._perform_adaptation()

        return scores

    def _perform_adaptation(self):
        """Perform adapter update using recent labeled samples."""
        if len(self.labeled_buffer) == 0:
            return

        # Combine recent labeled samples
        adaptation_batch = self._combine_buffer_samples()

        # Adapt model
        self.model.train()
        self.meta_optimizer.adapt(self.loss_fn, adaptation_batch, self.model)
        self.model.eval()

    def _combine_buffer_samples(self) -> Dict[str, torch.Tensor]:
        """Combine samples from labeled buffer into a single batch."""
        # Simple concatenation (in practice, may want sampling strategy)
        all_keys = self.labeled_buffer[0].keys()
        combined = {}

        for key in all_keys:
            if torch.is_tensor(self.labeled_buffer[0][key]):
                combined[key] = torch.cat([batch[key] for batch in self.labeled_buffer])

        return combined

    def reset_daily_budget(self):
        """Reset daily label budget (call at day boundary)."""
        self.labels_used_today = 0

    def save_state(self, path: str):
        """Save adaptation state."""
        state = {
            'meta_params': self.meta_optimizer.meta_params,
            'labels_used': self.labels_used_today,
            'microbatch_count': self.microbatch_count
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load adaptation state."""
        state = torch.load(path)
        self.meta_optimizer.meta_params = state['meta_params']
        self.labels_used_today = state['labels_used']
        self.microbatch_count = state['microbatch_count']


class DriftDetector:
    """
    Simple drift detector based on performance degradation.

    Monitors key metrics and signals when significant drift is detected.
    """
    def __init__(self,
                 window_size: int = 100,
                 threshold: float = 0.1):
        """
        Args:
            window_size: Number of recent predictions to track
            threshold: Relative performance drop to trigger drift alert
        """
        self.window_size = window_size
        self.threshold = threshold

        self.scores_window = deque(maxlen=window_size)
        self.labels_window = deque(maxlen=window_size)
        self.baseline_performance = None

    def update(self, scores: torch.Tensor, labels: torch.Tensor):
        """Update drift detector with new predictions."""
        self.scores_window.extend(scores.tolist())
        self.labels_window.extend(labels.tolist())

        # Update baseline if not set
        if self.baseline_performance is None and len(self.scores_window) >= self.window_size:
            self.baseline_performance = self._compute_performance()

    def check_drift(self) -> bool:
        """
        Check if drift has been detected.

        Returns:
            True if drift detected, False otherwise
        """
        if len(self.scores_window) < self.window_size or self.baseline_performance is None:
            return False

        current_performance = self._compute_performance()
        relative_drop = (self.baseline_performance - current_performance) / (self.baseline_performance + 1e-8)

        return relative_drop > self.threshold

    def _compute_performance(self) -> float:
        """Compute performance metric (e.g., AUC) on current window."""
        if len(self.scores_window) == 0:
            return 0.0

        # Simple metric: mean absolute calibration error
        scores = torch.tensor(list(self.scores_window))
        labels = torch.tensor(list(self.labels_window))

        # Bin scores and compute calibration
        num_bins = 10
        scores_binned = (scores * num_bins).long().clamp(0, num_bins - 1)

        calibration_error = 0.0
        for bin_idx in range(num_bins):
            mask = scores_binned == bin_idx
            if mask.sum() > 0:
                bin_scores = scores[mask].mean()
                bin_labels = labels[mask].float().mean()
                calibration_error += (bin_scores - bin_labels).abs()

        return 1.0 - (calibration_error / num_bins)  # Higher is better

    def reset(self):
        """Reset drift detector."""
        self.scores_window.clear()
        self.labels_window.clear()
        self.baseline_performance = None
