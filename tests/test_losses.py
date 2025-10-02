"""
Tests for loss functions.
"""

import pytest
import torch
from stream_fraudx.losses.focal_losses import CombinedFocalLoss, TverskyLoss
from stream_fraudx.losses.irm_loss import IRMLiteLoss
from stream_fraudx.losses.pretraining_losses import MaskedEdgeModelingLoss


def test_combined_focal_loss():
    """Test CombinedFocalLoss computation."""
    loss_fn = CombinedFocalLoss()

    logits = torch.randn(32, requires_grad=True)
    labels = torch.randint(0, 2, (32,))

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert loss.requires_grad, "Loss should require grad"

    # Backward should work
    loss.backward()
    assert logits.grad is not None


def test_tversky_loss():
    """Test TverskyLoss computation."""
    loss_fn = TverskyLoss(alpha=0.7, beta=0.3)

    logits = torch.randn(32, requires_grad=True)
    labels = torch.randint(0, 2, (32,))

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert loss.requires_grad, "Loss should require grad"


def test_irm_lite_loss():
    """Test IRMLiteLoss computation."""
    loss_fn = IRMLiteLoss(penalty_weight=0.1)

    embeddings = torch.randn(32, 128, requires_grad=True)
    labels = torch.randint(0, 2, (32,))
    time_slices = torch.randint(0, 4, (32,))

    penalty = loss_fn(embeddings, labels, time_slices)

    assert penalty.item() >= 0, "Penalty should be non-negative"

    # Should work with single time slice (returns 0)
    time_slices_single = torch.zeros(32, dtype=torch.long)
    penalty_single = loss_fn(embeddings, labels, time_slices_single)
    assert penalty_single.item() == 0.0


def test_masked_edge_modeling_loss():
    """Test MaskedEdgeModelingLoss initialization."""
    loss_fn = MaskedEdgeModelingLoss(
        mask_ratio=0.15,
        attribute_dims={'amount_bin': 50, 'mcc_bin': 20}
    )

    assert loss_fn.mask_ratio == 0.15
    assert 'amount_bin' in loss_fn.attribute_dims

    # Build heads
    loss_fn.build_heads(input_dim=256)

    assert hasattr(loss_fn, 'prediction_heads')
    assert 'amount_bin' in loss_fn.prediction_heads
