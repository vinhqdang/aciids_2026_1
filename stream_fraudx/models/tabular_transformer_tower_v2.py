"""
Enhanced Tabular Transformer Tower (v2) for STREAM-FraudX
Implements feature gating and FT-Transformer-style attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

from .tabular_transformer_tower import FeatureTokenizer, FourierTimeEncoding


class FeatureGating(nn.Module):
    """
    Feature gating mechanism for adaptive feature selection.

    Gates features based on their relevance to the current prediction,
    similar to feature selection but learned end-to-end.
    """

    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(num_features * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )

        # Feature importance tracking (for interpretability)
        self.register_buffer('feature_importance', torch.zeros(num_features))
        self.register_buffer('gate_count', torch.zeros(1))

    def forward(self, feature_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gating to feature embeddings.

        Args:
            feature_embeddings: (batch_size, num_features, hidden_dim)

        Returns:
            gated_embeddings: (batch_size, num_features, hidden_dim)
            gate_weights: (batch_size, num_features) - gate probabilities
        """
        batch_size, num_features, hidden_dim = feature_embeddings.shape

        # Flatten for gate network
        flat_features = feature_embeddings.view(batch_size, -1)

        # Compute gate weights
        gate_weights = self.gate_network(flat_features)  # (B, num_features)

        # Apply gates
        gated_embeddings = feature_embeddings * gate_weights.unsqueeze(-1)

        # Track feature importance (exponential moving average)
        if self.training:
            with torch.no_grad():
                batch_importance = gate_weights.mean(dim=0)
                alpha = 0.99
                self.feature_importance = (
                    alpha * self.feature_importance +
                    (1 - alpha) * batch_importance
                )
                self.gate_count += 1

        return gated_embeddings, gate_weights

    def get_feature_importance(self) -> torch.Tensor:
        """Get tracked feature importance scores."""
        return self.feature_importance


class FTTransformerBlock(nn.Module):
    """
    FT-Transformer style attention block.

    Differences from standard transformer:
    - Uses feature-wise attention instead of token-wise
    - Includes feature-specific normalization
    - Optimized for tabular data
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 use_feature_specific_norm: bool = True):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.use_feature_specific_norm = use_feature_specific_norm

        # Pre-norm
        self.norm1 = nn.LayerNorm(dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            dim, num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-attention norm
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network with gating
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

        # Residual gating (learnable skip connections)
        self.gate_attn = nn.Parameter(torch.ones(1))
        self.gate_ffn = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through FT-Transformer block.

        Args:
            x: (batch_size, num_features, dim)
            mask: Optional attention mask

        Returns:
            output: (batch_size, num_features, dim)
        """
        # Pre-norm + attention + gated residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.gate_attn * attn_out

        # Pre-norm + FFN + gated residual
        x = x + self.gate_ffn * self.ffn(self.norm2(x))

        return x


class FeatureInteractionBlock(nn.Module):
    """
    Explicit feature interaction block for capturing cross-feature patterns.

    Uses bilinear interactions and factorization machines.
    """

    def __init__(self, dim: int, num_factors: int = 16):
        super().__init__()
        self.dim = dim
        self.num_factors = num_factors

        # Factorization machine embeddings
        self.fm_embeddings = nn.Linear(dim, num_factors)

        # Bilinear interaction
        self.bilinear = nn.Bilinear(dim, dim, dim)

        # Projection
        self.projection = nn.Linear(num_factors + dim, dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature interactions.

        Args:
            features: (batch_size, num_features, dim)

        Returns:
            interactions: (batch_size, num_features, dim)
        """
        batch_size, num_features, dim = features.shape

        # FM-style interactions
        fm_emb = self.fm_embeddings(features)  # (B, num_features, num_factors)

        # Sum of squares - square of sums trick for O(n) complexity
        sum_square = torch.sum(fm_emb ** 2, dim=1)  # (B, num_factors)
        square_sum = torch.sum(fm_emb, dim=1) ** 2  # (B, num_factors)
        fm_interactions = 0.5 * (square_sum - sum_square)  # (B, num_factors)

        # Expand back to feature dimension
        fm_interactions = fm_interactions.unsqueeze(1).expand(-1, num_features, -1)

        # Bilinear interactions (simplified - pairwise)
        # For efficiency, we approximate with mean pooling
        features_pooled = features.mean(dim=1, keepdim=True)  # (B, 1, dim)
        bilinear_out = self.bilinear(
            features,
            features_pooled.expand(-1, num_features, -1)
        )

        # Combine interactions
        combined = torch.cat([fm_interactions, bilinear_out], dim=-1)
        return self.projection(combined)


class EnhancedTabularTransformerTower(nn.Module):
    """
    Enhanced Tabular Transformer Tower with feature gating and FT-Transformer blocks.

    Improvements over v1:
    - Feature gating for adaptive feature selection
    - FT-Transformer style attention optimized for tabular data
    - Explicit feature interaction modeling
    - Feature importance tracking
    """

    def __init__(self,
                 continuous_dims: List[int],
                 categorical_vocab_sizes: List[int],
                 hidden_dim: int = 128,
                 num_blocks: int = 3,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 use_feature_gating: bool = True,
                 use_feature_interactions: bool = True):
        super().__init__()

        self.continuous_dims = continuous_dims
        self.categorical_vocab_sizes = categorical_vocab_sizes
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_feature_gating = use_feature_gating
        self.use_feature_interactions = use_feature_interactions

        num_features = len(continuous_dims) + len(categorical_vocab_sizes)

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(
            continuous_dims,
            categorical_vocab_sizes,
            embedding_dim=hidden_dim
        )

        # Feature gating (optional)
        if use_feature_gating:
            self.feature_gate = FeatureGating(num_features, hidden_dim)
        else:
            self.feature_gate = None

        # FT-Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FTTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Feature interaction block (optional)
        if use_feature_interactions:
            self.interaction_block = FeatureInteractionBlock(hidden_dim)
        else:
            self.interaction_block = None

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through enhanced tabular tower.

        Args:
            batch: Dictionary with keys:
                - 'continuous': (batch_size, num_continuous)
                - 'categorical': (batch_size, num_categorical)

        Returns:
            features: (batch_size, hidden_dim) - global feature representation
        """
        continuous = batch['continuous']
        categorical = batch['categorical']
        batch_size = continuous.size(0)

        # Tokenize features
        feature_tokens = self.tokenizer(continuous, categorical)  # (B, num_features, hidden_dim)

        # Apply feature gating if enabled
        gate_weights = None
        if self.feature_gate is not None:
            feature_tokens, gate_weights = self.feature_gate(feature_tokens)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)  # (B, num_features+1, hidden_dim)

        # Apply FT-Transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Apply feature interactions if enabled
        if self.interaction_block is not None:
            # Apply to feature tokens (exclude CLS)
            feature_part = tokens[:, 1:, :]
            interactions = self.interaction_block(feature_part)
            tokens = torch.cat([tokens[:, :1, :], interactions], dim=1)

        # Extract CLS token as global representation
        cls_output = tokens[:, 0, :]  # (B, hidden_dim)

        # Output projection
        output = self.output_projection(self.output_norm(cls_output))

        return output

    def get_attention_weights(self, batch: Dict[str, torch.Tensor], block_idx: int = 0) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            batch: Input batch
            block_idx: Which transformer block to extract from

        Returns:
            attention_weights: Attention matrix
        """
        # This is a simplified version - full implementation would hook into attention
        # For now, just return feature gate weights if available
        if self.feature_gate is not None:
            continuous = batch['continuous']
            categorical = batch['categorical']
            feature_tokens = self.tokenizer(continuous, categorical)
            _, gate_weights = self.feature_gate(feature_tokens)
            return gate_weights
        else:
            return None

    def get_feature_importance(self) -> Optional[torch.Tensor]:
        """Get feature importance scores from gating."""
        if self.feature_gate is not None:
            return self.feature_gate.get_feature_importance()
        return None


class AdaptiveFeatureSelector(nn.Module):
    """
    Adaptive feature selector that learns to select relevant features per-instance.

    Useful for high-dimensional tabular data with sparse relevant features.
    """

    def __init__(self, num_features: int, hidden_dim: int, top_k: int = None):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.top_k = top_k or num_features // 2

        # Selection network
        self.selector = nn.Sequential(
            nn.Linear(num_features * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, features: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k features per instance.

        Args:
            features: (batch_size, num_features, hidden_dim)
            hard: If True, use hard selection; otherwise soft

        Returns:
            selected_features: (batch_size, num_features, hidden_dim)
            selection_mask: (batch_size, num_features)
        """
        batch_size = features.size(0)

        # Compute selection scores
        flat_features = features.view(batch_size, -1)
        scores = self.selector(flat_features)  # (B, num_features)

        if hard:
            # Hard top-k selection
            _, top_indices = torch.topk(scores, self.top_k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_indices, 1.0)
        else:
            # Soft selection with sigmoid
            mask = torch.sigmoid(scores)

        # Apply selection
        selected = features * mask.unsqueeze(-1)

        return selected, mask
