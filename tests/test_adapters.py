"""
Tests for adapter modules.
"""

import pytest
import torch
from stream_fraudx.models.adapters import LoRAAdapter, PrefixAdapter, AdapterLayer


def test_lora_adapter():
    """Test LoRA adapter forward pass."""
    adapter = LoRAAdapter(input_dim=128, rank=8)

    x = torch.randn(16, 128)
    output = adapter(x)

    assert output.shape == x.shape
    assert not torch.allclose(output, x), "Adapter should modify input"


def test_prefix_adapter():
    """Test Prefix adapter forward pass."""
    adapter = PrefixAdapter(input_dim=128, prefix_length=10)

    x = torch.randn(16, 32, 128)  # (batch, seq_len, dim)
    output = adapter(x)

    assert output.shape == x.shape


def test_adapter_layer():
    """Test AdapterLayer with different types."""
    # Test LoRA
    adapter = AdapterLayer(input_dim=128, adapter_type='lora', rank=8)
    x = torch.randn(16, 128)
    output = adapter(x)
    assert output.shape == x.shape

    # Test Prefix
    adapter_prefix = AdapterLayer(input_dim=128, adapter_type='prefix', prefix_length=10)
    x_seq = torch.randn(16, 32, 128)
    output_seq = adapter_prefix(x_seq)
    assert output_seq.shape == x_seq.shape


def test_adapter_gradients():
    """Test that adapters have trainable parameters."""
    adapter = LoRAAdapter(input_dim=128, rank=8)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert trainable_params > 0, "Adapter should have trainable parameters"

    # Test backward pass
    x = torch.randn(16, 128, requires_grad=True)
    output = adapter(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
