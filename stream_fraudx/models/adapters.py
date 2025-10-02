"""
Parameter-Efficient Adapters for STREAM-FraudX
Implements bottleneck adapters for fast drift adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module for parameter-efficient fine-tuning.
    Uses low-rank bottleneck structure: d -> d/r -> d
    """
    def __init__(self,
                 input_dim: int,
                 bottleneck_dim: Optional[int] = None,
                 reduction_factor: int = 8,
                 activation: str = 'gelu',
                 dropout: float = 0.1):
        super().__init__()

        if bottleneck_dim is None:
            bottleneck_dim = max(input_dim // reduction_factor, 16)

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights for stable training
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, ..., input_dim)
        Returns:
            adapted: (batch_size, ..., input_dim) with residual connection
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x  # Residual connection


class AdapterLayer(nn.Module):
    """
    Full adapter layer that wraps a module with adapter.
    Can be inserted into transformer blocks or message functions.
    """
    def __init__(self,
                 module: nn.Module,
                 adapter_dim: int,
                 reduction_factor: int = 8):
        super().__init__()

        self.module = module
        self.adapter = BottleneckAdapter(adapter_dim, reduction_factor=reduction_factor)

    def forward(self, *args, **kwargs):
        """
        Forward through module then adapter.
        """
        output = self.module(*args, **kwargs)
        return self.adapter(output)


class GraphTowerAdapters(nn.Module):
    """
    Adapter modules for Temporal Graph Tower.
    Adds adapters to message functions for drift adaptation.
    """
    def __init__(self,
                 node_dim: int,
                 num_layers: int,
                 reduction_factor: int = 8):
        super().__init__()

        self.adapters = nn.ModuleList([
            BottleneckAdapter(node_dim, reduction_factor=reduction_factor)
            for _ in range(num_layers)
        ])

    def forward(self, layer_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply adapters to each layer output.
        Args:
            layer_outputs: List of (batch_size, node_dim) tensors
        Returns:
            adapted_outputs: List of adapted tensors
        """
        return [adapter(output) for adapter, output in zip(self.adapters, layer_outputs)]


class TabularTowerAdapters(nn.Module):
    """
    Adapter modules for Tabular Transformer Tower.
    Adds adapters to transformer blocks for drift adaptation.
    """
    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 reduction_factor: int = 8):
        super().__init__()

        self.adapters = nn.ModuleList([
            BottleneckAdapter(hidden_dim, reduction_factor=reduction_factor)
            for _ in range(num_layers)
        ])

    def forward(self, layer_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply adapters to each transformer layer output.
        Args:
            layer_outputs: List of (batch_size, seq_len, hidden_dim) tensors
        Returns:
            adapted_outputs: List of adapted tensors
        """
        return [adapter(output) for adapter, output in zip(self.adapters, layer_outputs)]


class FusionAdapter(nn.Module):
    """
    Adapter for fusion layer to handle cross-tower drift.
    """
    def __init__(self, fusion_dim: int, reduction_factor: int = 8):
        super().__init__()

        self.adapter = BottleneckAdapter(fusion_dim, reduction_factor=reduction_factor)

    def forward(self, fusion_output: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter to fusion output.
        """
        return self.adapter(fusion_output)


class MetaAdapter(nn.Module):
    """
    Meta-learning wrapper for adapters.
    Supports Reptile-style meta-gradient updates.
    """
    def __init__(self, adapters: nn.ModuleList):
        super().__init__()

        self.adapters = adapters
        self.meta_params = list(adapters.parameters())

        # Store initialization for meta-learning
        self.init_params = [p.clone().detach() for p in self.meta_params]

    def save_meta_state(self):
        """Save current adapter state as meta initialization."""
        self.init_params = [p.clone().detach() for p in self.meta_params]

    def meta_update(self, lr: float = 0.01):
        """
        Perform Reptile-style meta update.
        Move initialization toward current adapted parameters.
        """
        with torch.no_grad():
            for init_p, curr_p in zip(self.init_params, self.meta_params):
                init_p.copy_(init_p + lr * (curr_p - init_p))

    def reset_to_meta(self):
        """Reset adapters to meta initialization."""
        with torch.no_grad():
            for p, init_p in zip(self.meta_params, self.init_params):
                p.copy_(init_p)

    def get_adapter_params(self) -> List[torch.nn.Parameter]:
        """Get all adapter parameters for optimization."""
        return self.meta_params


class AdapterConfig:
    """Configuration for adapter setup."""
    def __init__(self,
                 use_graph_adapters: bool = True,
                 use_tabular_adapters: bool = True,
                 use_fusion_adapters: bool = True,
                 reduction_factor: int = 8,
                 dropout: float = 0.1):
        self.use_graph_adapters = use_graph_adapters
        self.use_tabular_adapters = use_tabular_adapters
        self.use_fusion_adapters = use_fusion_adapters
        self.reduction_factor = reduction_factor
        self.dropout = dropout
