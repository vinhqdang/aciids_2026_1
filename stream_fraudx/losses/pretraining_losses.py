"""
Self-Supervised Pretraining Losses for STREAM-FraudX
Implements Masked Edge Modeling (MEM) and Subgraph Contrastive Learning (InfoNCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import random


class MaskedEdgeModelingLoss(nn.Module):
    """
    Masked Edge Modeling (MEM) loss for graph pretraining.

    Masks edge attributes and predicts them from context.
    Similar to masked language modeling but for graph edges.
    """
    def __init__(self,
                 mask_ratio: float = 0.15,
                 attribute_dims: Dict[str, int] = None):
        """
        Args:
            mask_ratio: Fraction of edges to mask
            attribute_dims: Dict mapping attribute names to output dimensions
                           e.g., {'amount_bin': 50, 'mcc_bin': 20, 'device_type': 10}
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.attribute_dims = attribute_dims or {}

        # Prediction heads for each attribute
        self.prediction_heads = nn.ModuleDict()

    def build_heads(self, input_dim: int):
        """Build prediction heads after knowing input dimension."""
        for attr_name, output_dim in self.attribute_dims.items():
            self.prediction_heads[attr_name] = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, output_dim)
            )

    def mask_edges(self, batch_events: Dict[str, torch.Tensor]) -> Tuple[Dict, torch.Tensor, Dict]:
        """
        Randomly mask edge attributes.

        Args:
            batch_events: Batch of events with edge attributes

        Returns:
            masked_events: Events with some attributes masked
            mask: Boolean tensor indicating masked positions
            targets: Original attribute values for masked edges
        """
        batch_size = batch_events['edge_attrs'].size(0)
        device = batch_events['edge_attrs'].device

        # Random mask
        mask = torch.rand(batch_size, device=device) < self.mask_ratio

        # Store targets for masked attributes
        targets = {}
        if 'edge_attr_discrete' in batch_events:
            # Discrete attributes (categorical)
            targets = {
                attr: batch_events['edge_attr_discrete'][attr][mask]
                for attr in batch_events['edge_attr_discrete']
            }

        # Create masked version
        masked_events = batch_events.copy()
        if mask.any():
            # Replace masked attributes with zeros or special mask token
            masked_edge_attrs = batch_events['edge_attrs'].clone()
            masked_edge_attrs[mask] = 0  # Mask with zeros
            masked_events['edge_attrs'] = masked_edge_attrs

        return masked_events, mask, targets

    def forward(self,
                edge_representations: torch.Tensor,
                mask: torch.Tensor,
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute MEM loss.

        Args:
            edge_representations: (batch_size, dim) edge representations from model
            mask: (batch_size,) boolean mask indicating which edges were masked
            targets: Dict of target attribute values for masked edges

        Returns:
            loss: MEM reconstruction loss
        """
        if not mask.any():
            return torch.tensor(0.0, device=edge_representations.device)

        # Get representations for masked edges
        masked_reps = edge_representations[mask]

        # Compute prediction loss for each attribute
        total_loss = 0.0
        num_attributes = 0

        for attr_name, attr_targets in targets.items():
            if attr_name not in self.prediction_heads:
                continue

            # Predict attribute
            logits = self.prediction_heads[attr_name](masked_reps)

            # Cross-entropy loss
            loss = F.cross_entropy(logits, attr_targets.long())
            total_loss += loss
            num_attributes += 1

        if num_attributes == 0:
            return torch.tensor(0.0, device=edge_representations.device)

        return total_loss / num_attributes


class SubgraphContrastiveLoss(nn.Module):
    """
    Subgraph contrastive learning with InfoNCE loss.

    Creates positive pairs from temporally coherent subgraphs.
    Pushes apart negative samples across time/structure.
    """
    def __init__(self,
                 temperature: float = 0.2,
                 queue_size: int = 4096,
                 use_queue: bool = True):
        """
        Args:
            temperature: Temperature parameter for InfoNCE
            queue_size: Size of negative sample queue
            use_queue: Whether to use memory queue for negatives
        """
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.use_queue = use_queue

        # Queue for negative samples
        if use_queue:
            self.register_buffer("queue", torch.randn(queue_size, 128))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue = F.normalize(self.queue, dim=1)

    def create_augmented_views(self,
                               batch_events: Dict[str, torch.Tensor],
                               model) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of the same subgraph.

        Augmentations:
        1. Temporal random walk sampling
        2. Edge dropout + time jitter

        Args:
            batch_events: Batch of events
            model: The model to encode events

        Returns:
            view1: (batch_size, dim) first view representations
            view2: (batch_size, dim) second view representations
        """
        # View 1: Original events
        view1 = model(batch_events, update_memory=False)

        # View 2: Augmented events
        augmented = self._augment_events(batch_events)
        view2 = model(augmented, update_memory=False)

        return view1, view2

    def _augment_events(self, batch_events: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to create different view.
        """
        augmented = {}

        # Edge dropout (randomly drop some edge features)
        if 'edge_attrs' in batch_events:
            edge_attrs = batch_events['edge_attrs'].clone()
            dropout_mask = torch.rand(edge_attrs.size(0), 1, device=edge_attrs.device) > 0.1
            augmented['edge_attrs'] = edge_attrs * dropout_mask
        else:
            augmented['edge_attrs'] = batch_events['edge_attrs']

        # Time jitter (add small noise to timestamps)
        if 'timestamps' in batch_events:
            timestamps = batch_events['timestamps'].clone()
            jitter = torch.randn_like(timestamps) * 0.01  # Small jitter
            augmented['timestamps'] = timestamps + jitter
        else:
            augmented['timestamps'] = batch_events['timestamps']

        # Keep other fields unchanged
        augmented['src_nodes'] = batch_events['src_nodes']
        augmented['dst_nodes'] = batch_events['dst_nodes']

        return augmented

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            view1: (batch_size, dim) first view representations
            view2: (batch_size, dim) second view representations

        Returns:
            loss: InfoNCE loss
        """
        batch_size = view1.size(0)
        device = view1.device

        # Normalize representations
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        # Positive pairs: view1[i] with view2[i]
        pos_sim = torch.sum(view1 * view2, dim=1) / self.temperature  # (B,)

        # Negative pairs
        if self.use_queue and hasattr(self, 'queue'):
            # Use queue for more negative samples
            neg_sim_1 = torch.matmul(view1, self.queue.T) / self.temperature  # (B, queue_size)
            neg_sim_2 = torch.matmul(view2, self.queue.T) / self.temperature  # (B, queue_size)

            # Update queue
            self._dequeue_and_enqueue(view2)

            # Combine positive and negative similarities
            logits_1 = torch.cat([pos_sim.unsqueeze(1), neg_sim_1], dim=1)  # (B, 1+queue_size)
            logits_2 = torch.cat([pos_sim.unsqueeze(1), neg_sim_2], dim=1)

        else:
            # Use in-batch negatives only
            neg_sim_1 = torch.matmul(view1, view2.T) / self.temperature  # (B, B)
            neg_sim_2 = neg_sim_1.T

            # Mask out self-similarities
            mask = torch.eye(batch_size, device=device).bool()
            neg_sim_1.masked_fill_(mask, float('-inf'))
            neg_sim_2.masked_fill_(mask, float('-inf'))

            logits_1 = torch.cat([pos_sim.unsqueeze(1), neg_sim_1], dim=1)
            logits_2 = torch.cat([pos_sim.unsqueeze(1), neg_sim_2], dim=1)

        # Targets: positive pairs are at index 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Cross-entropy loss for both views
        loss_1 = F.cross_entropy(logits_1, targets)
        loss_2 = F.cross_entropy(logits_2, targets)

        return (loss_1 + loss_2) / 2

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new negative samples."""
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)

        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]

        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


class PretrainingLoss(nn.Module):
    """
    Combined pretraining loss: MEM + InfoNCE.

    This is the main loss for self-supervised pretraining stage.
    """
    def __init__(self,
                 mem_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 **kwargs):
        """
        Args:
            mem_weight: Weight for MEM loss
            contrastive_weight: Weight for contrastive loss
        """
        super().__init__()

        self.mem_loss = MaskedEdgeModelingLoss(**kwargs.get('mem_kwargs', {}))
        self.contrastive_loss = SubgraphContrastiveLoss(**kwargs.get('contrastive_kwargs', {}))

        self.mem_weight = mem_weight
        self.contrastive_weight = contrastive_weight

    def forward(self,
                model,
                batch_events: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined pretraining loss.

        Args:
            model: The model being trained
            batch_events: Batch of events

        Returns:
            loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # MEM loss
        masked_events, mask, targets = self.mem_loss.mask_edges(batch_events)
        edge_reps = model(masked_events, update_memory=False)
        loss_mem = self.mem_loss(edge_reps, mask, targets)

        # Contrastive loss
        view1, view2 = self.contrastive_loss.create_augmented_views(batch_events, model)
        loss_contrastive = self.contrastive_loss(view1, view2)

        # Combine losses
        total_loss = self.mem_weight * loss_mem + self.contrastive_weight * loss_contrastive

        loss_dict = {
            'mem': loss_mem.item(),
            'contrastive': loss_contrastive.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict
