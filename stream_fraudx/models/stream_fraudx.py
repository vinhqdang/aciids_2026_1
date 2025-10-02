"""
Main STREAM-FraudX Model
Dual-Tower architecture with Temporal Graph + Tabular Transformer and Gated Fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .temporal_graph_tower import TemporalGraphTower
from .tabular_transformer_tower import TabularTransformerTower
from .fusion import GatedCrossAttentionFusion, ConcatFusion, FraudDetectionHead
from .adapters import (
    GraphTowerAdapters, TabularTowerAdapters, FusionAdapter,
    MetaAdapter, AdapterConfig
)


class STREAMFraudX(nn.Module):
    """
    STREAM-FraudX: Label-Efficient Streaming Fraud Detection

    Combines:
    - Temporal Graph Tower (TGT) for relational patterns
    - Tabular Transformer Tower (TTT) for transaction features
    - Gated Cross-Attention Fusion
    - Parameter-Efficient Adapters for drift adaptation
    """
    def __init__(self,
                 # Graph tower config
                 graph_node_dim: int = 128,
                 graph_edge_dim: int = 64,
                 graph_time_dim: int = 32,
                 graph_hidden_dim: int = 256,
                 graph_num_layers: int = 2,
                 max_neighbors: int = 20,
                 memory_size: int = 10000,
                 # Tabular tower config
                 continuous_dims: List[int] = None,
                 categorical_vocab_sizes: List[int] = None,
                 tabular_embedding_dim: int = 128,
                 tabular_num_layers: int = 3,
                 tabular_num_heads: int = 8,
                 # Fusion config
                 fusion_hidden_dim: int = 256,
                 fusion_num_heads: int = 4,
                 use_cross_attention: bool = True,
                 # Head config
                 head_hidden_dim: int = 128,
                 # Adapter config
                 use_adapters: bool = True,
                 adapter_reduction_factor: int = 8):
        super().__init__()

        # Default configurations
        if continuous_dims is None:
            continuous_dims = list(range(10))  # 10 continuous features
        if categorical_vocab_sizes is None:
            categorical_vocab_sizes = [100] * 5  # 5 categorical features

        self.use_adapters = use_adapters
        self.use_cross_attention = use_cross_attention

        # ===== Temporal Graph Tower =====
        self.tgt = TemporalGraphTower(
            node_dim=graph_node_dim,
            edge_dim=graph_edge_dim,
            time_dim=graph_time_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers,
            max_neighbors=max_neighbors,
            memory_size=memory_size
        )

        # ===== Tabular Transformer Tower =====
        self.ttt = TabularTransformerTower(
            continuous_dims=continuous_dims,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=tabular_embedding_dim,
            num_layers=tabular_num_layers,
            num_heads=tabular_num_heads
        )

        # ===== Fusion =====
        graph_output_dim = graph_node_dim * 2  # Concatenated src + dst
        tabular_output_dim = tabular_embedding_dim

        if use_cross_attention:
            self.fusion = GatedCrossAttentionFusion(
                graph_dim=graph_output_dim,
                tabular_dim=tabular_output_dim,
                hidden_dim=fusion_hidden_dim,
                num_heads=fusion_num_heads
            )
        else:
            self.fusion = ConcatFusion(
                graph_dim=graph_output_dim,
                tabular_dim=tabular_output_dim,
                hidden_dim=fusion_hidden_dim
            )

        # ===== Detection Head =====
        self.head = FraudDetectionHead(
            input_dim=self.fusion.output_dim,
            hidden_dim=head_hidden_dim
        )

        # ===== Adapters (optional) =====
        if use_adapters:
            # Graph adapter needs to match output dimension (2*node_dim)
            self.graph_adapters = GraphTowerAdapters(
                node_dim=graph_node_dim * 2,  # Graph tower outputs concatenated src+dst
                num_layers=graph_num_layers,
                reduction_factor=adapter_reduction_factor
            )

            self.tabular_adapters = TabularTowerAdapters(
                hidden_dim=tabular_embedding_dim,
                num_layers=tabular_num_layers,
                reduction_factor=adapter_reduction_factor
            )

            self.fusion_adapter = FusionAdapter(
                fusion_dim=fusion_hidden_dim,
                reduction_factor=adapter_reduction_factor
            )

            # Meta-adapter wrapper
            all_adapters = nn.ModuleList([
                self.graph_adapters,
                self.tabular_adapters,
                self.fusion_adapter
            ])
            self.meta_adapter = MetaAdapter(all_adapters)
        else:
            self.graph_adapters = None
            self.tabular_adapters = None
            self.fusion_adapter = None
            self.meta_adapter = None

    def forward(self,
                batch: Dict[str, torch.Tensor],
                update_memory: bool = True,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass for fraud detection.

        Args:
            batch: Dictionary containing:
                - Graph data: src_nodes, dst_nodes, edge_attrs, timestamps
                - Tabular data: continuous, categorical
            update_memory: Whether to update graph memory (True during training/inference)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            logits: (batch_size,) fraud detection logits
            OR (logits, embeddings) if return_embeddings=True
        """
        # ===== Graph Tower =====
        graph_events = {
            'src_nodes': batch['src_nodes'],
            'dst_nodes': batch['dst_nodes'],
            'edge_attrs': batch['edge_attrs'],
            'timestamps': batch['timestamps']
        }
        graph_emb = self.tgt(graph_events, update_memory=update_memory)

        # Apply graph adapters if available
        if self.use_adapters and self.graph_adapters is not None:
            # For simplicity, apply adapter to final output
            # In full implementation, would apply to each layer
            graph_emb = self.graph_adapters.adapters[-1](graph_emb)

        # ===== Tabular Tower =====
        tabular_features = {
            'continuous': batch['continuous'],
            'categorical': batch['categorical'],
            'timestamps': batch['timestamps']
        }
        tabular_emb = self.ttt(tabular_features)

        # Apply tabular adapters if available
        if self.use_adapters and self.tabular_adapters is not None:
            tabular_emb = self.tabular_adapters.adapters[-1](tabular_emb)

        # ===== Fusion =====
        fused_emb = self.fusion(graph_emb, tabular_emb)

        # Apply fusion adapter if available
        if self.use_adapters and self.fusion_adapter is not None:
            fused_emb = self.fusion_adapter(fused_emb)

        # ===== Detection Head =====
        logits = self.head(fused_emb)

        if return_embeddings:
            embeddings = {
                'graph': graph_emb,
                'tabular': tabular_emb,
                'fused': fused_emb
            }
            return logits, embeddings

        return logits

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            batch: Batch dictionary

        Returns:
            probs: (batch_size,) fraud probabilities
        """
        logits = self.forward(batch, update_memory=False)
        return torch.sigmoid(logits)

    def reset_memory(self):
        """Reset graph tower memory (useful between episodes)."""
        self.tgt.reset()

    def get_adapter_params(self) -> List[torch.nn.Parameter]:
        """Get all adapter parameters for meta-learning."""
        if not self.use_adapters or self.meta_adapter is None:
            return []
        return self.meta_adapter.get_adapter_params()

    def freeze_backbone(self):
        """Freeze backbone parameters (TGT, TTT, Fusion, Head)."""
        for param in self.tgt.parameters():
            param.requires_grad = False
        for param in self.ttt.parameters():
            param.requires_grad = False
        for param in self.fusion.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.tgt.parameters():
            param.requires_grad = True
        for param in self.ttt.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def get_model_size(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'tgt': count_params(self.tgt),
            'ttt': count_params(self.ttt),
            'fusion': count_params(self.fusion),
            'head': count_params(self.head),
            'adapters': count_params(self.graph_adapters) +
                       count_params(self.tabular_adapters) +
                       count_params(self.fusion_adapter) if self.use_adapters else 0,
            'total': count_params(self)
        }


class STREAMFraudXConfig:
    """Configuration class for STREAM-FraudX model."""

    def __init__(self):
        # Graph tower
        self.graph_node_dim = 128
        self.graph_edge_dim = 64
        self.graph_time_dim = 32
        self.graph_hidden_dim = 256
        self.graph_num_layers = 2
        self.max_neighbors = 20
        self.memory_size = 10000

        # Tabular tower
        self.continuous_dims = list(range(10))
        self.categorical_vocab_sizes = [100] * 5
        self.tabular_embedding_dim = 128
        self.tabular_num_layers = 3
        self.tabular_num_heads = 8

        # Fusion
        self.fusion_hidden_dim = 256
        self.fusion_num_heads = 4
        self.use_cross_attention = True

        # Head
        self.head_hidden_dim = 128

        # Adapters
        self.use_adapters = True
        self.adapter_reduction_factor = 8

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
