"""
Enhanced Fusion Modules (v2) for STREAM-FraudX
Implements FiLM-style residual modulation for bidirectional conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation γ * x + β where γ and β are
    conditioned on another modality.
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # Generate scale (gamma) and shift (beta) from condition
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            features: (batch_size, feature_dim) - features to modulate
            condition: (batch_size, condition_dim) - conditioning signal

        Returns:
            modulated: (batch_size, feature_dim) - modulated features
        """
        # Generate gamma and beta
        film_params = self.film_generator(condition)  # (B, 2*feature_dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each (B, feature_dim)

        # Apply affine transformation
        modulated = gamma * features + beta

        return modulated


class ResidualFiLMBlock(nn.Module):
    """
    Residual block with FiLM conditioning.

    Combines residual connections with feature-wise modulation.
    """

    def __init__(self,
                 feature_dim: int,
                 condition_dim: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        hidden_dim = hidden_dim or feature_dim * 2

        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )

        # FiLM modulation
        self.film = FiLMLayer(feature_dim, condition_dim)

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.

        Args:
            features: (batch_size, feature_dim)
            condition: (batch_size, condition_dim)

        Returns:
            output: (batch_size, feature_dim)
        """
        # Transform features
        transformed = self.transform(features)

        # Apply FiLM modulation
        modulated = self.film(transformed, condition)

        # Residual connection with learnable weight
        output = features + self.residual_weight * modulated

        # Layer norm
        output = self.norm(output)

        return output


class BidirectionalFiLMFusion(nn.Module):
    """
    Bidirectional FiLM fusion module.

    Enables graph and tabular modalities to condition each other
    via FiLM modulation in both directions.
    """

    def __init__(self,
                 graph_dim: int,
                 tabular_dim: int,
                 hidden_dim: int = 256,
                 num_film_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.graph_dim = graph_dim
        self.tabular_dim = tabular_dim
        self.hidden_dim = hidden_dim

        # Project to common dimension
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        # FiLM blocks: graph conditioned on tabular
        self.graph_film_blocks = nn.ModuleList([
            ResidualFiLMBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_film_blocks)
        ])

        # FiLM blocks: tabular conditioned on graph
        self.tabular_film_blocks = nn.ModuleList([
            ResidualFiLMBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_film_blocks)
        ])

        # Cross-attention for additional interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Output fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, graph_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional FiLM fusion.

        Args:
            graph_emb: (batch_size, graph_dim)
            tabular_emb: (batch_size, tabular_dim)

        Returns:
            fused: (batch_size, hidden_dim)
        """
        # Project to common dimension
        graph_feat = self.graph_proj(graph_emb)  # (B, hidden_dim)
        tabular_feat = self.tabular_proj(tabular_emb)  # (B, hidden_dim)

        # Apply alternating FiLM conditioning
        for graph_film, tabular_film in zip(self.graph_film_blocks, self.tabular_film_blocks):
            # Graph conditioned on tabular
            graph_feat_new = graph_film(graph_feat, tabular_feat)

            # Tabular conditioned on graph
            tabular_feat_new = tabular_film(tabular_feat, graph_feat)

            # Update
            graph_feat = graph_feat_new
            tabular_feat = tabular_feat_new

        # Additional cross-attention
        # Stack for attention
        graph_seq = graph_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        tabular_seq = tabular_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        combined = torch.cat([graph_seq, tabular_seq], dim=1)  # (B, 2, hidden_dim)

        # Self-attention over both modalities
        attended, _ = self.cross_attention(combined, combined, combined)  # (B, 2, hidden_dim)

        # Extract and fuse
        graph_attended = attended[:, 0, :]  # (B, hidden_dim)
        tabular_attended = attended[:, 1, :]  # (B, hidden_dim)

        # Residual connection
        graph_final = graph_feat + graph_attended
        tabular_final = tabular_feat + tabular_attended

        # Final fusion
        fused = torch.cat([graph_final, tabular_final], dim=-1)  # (B, 2*hidden_dim)
        output = self.output_fusion(fused)  # (B, hidden_dim)

        return output


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion with learned modality weights.

    Dynamically weights graph vs tabular information per-instance.
    """

    def __init__(self,
                 graph_dim: int,
                 tabular_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # Project modalities
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        # Modality weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(graph_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 weights for graph and tabular
            nn.Softmax(dim=-1)
        )

        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, graph_emb: torch.Tensor, tabular_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive fusion with learned weights.

        Args:
            graph_emb: (batch_size, graph_dim)
            tabular_emb: (batch_size, tabular_dim)

        Returns:
            fused: (batch_size, hidden_dim)
            weights: (batch_size, 2) - modality weights [graph_weight, tabular_weight]
        """
        # Compute modality weights
        concat = torch.cat([graph_emb, tabular_emb], dim=-1)
        weights = self.weight_predictor(concat)  # (B, 2)

        # Project modalities
        graph_feat = self.graph_proj(graph_emb)  # (B, hidden_dim)
        tabular_feat = self.tabular_proj(tabular_emb)  # (B, hidden_dim)

        # Weighted combination
        graph_weight = weights[:, 0:1]  # (B, 1)
        tabular_weight = weights[:, 1:2]  # (B, 1)

        fused = graph_weight * graph_feat + tabular_weight * tabular_feat

        # Transform
        output = self.transform(fused)

        return output, weights


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion with multiple levels of interaction.

    Combines FiLM modulation, cross-attention, and adaptive weighting.
    """

    def __init__(self,
                 graph_dim: int,
                 tabular_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # Level 1: FiLM modulation
        self.film_fusion = BidirectionalFiLMFusion(
            graph_dim, tabular_dim, hidden_dim, num_film_blocks=1, dropout=dropout
        )

        # Level 2: Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Level 3: Adaptive weighting
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, graph_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical fusion.

        Args:
            graph_emb: (batch_size, graph_dim)
            tabular_emb: (batch_size, tabular_dim)

        Returns:
            fused: (batch_size, hidden_dim)
        """
        # Level 1: FiLM-based fusion
        film_fused = self.film_fusion(graph_emb, tabular_emb)  # (B, hidden_dim)

        # Level 2: Self-attention refinement
        film_seq = film_fused.unsqueeze(1)  # (B, 1, hidden_dim)
        attended, _ = self.cross_attention(film_seq, film_seq, film_seq)
        attended = attended.squeeze(1)  # (B, hidden_dim)

        # Residual
        level2_out = film_fused + attended

        # Level 3: Adaptive gating
        gate = self.adaptive_gate(level2_out)
        gated = gate * level2_out

        # Output
        output = self.output_norm(gated)

        return output


class EnhancedFraudDetectionHead(nn.Module):
    """
    Enhanced detection head with uncertainty estimation.

    Provides fraud probability + uncertainty score.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 use_uncertainty: bool = True):
        super().__init__()

        self.use_uncertainty = use_uncertainty

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Uncertainty head (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )

    def forward(self, fused_emb: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty.

        Args:
            fused_emb: (batch_size, input_dim)

        Returns:
            logits: (batch_size,) - fraud scores
            uncertainty: (batch_size,) - uncertainty scores (if enabled)
        """
        logits = self.classifier(fused_emb).squeeze(-1)  # (B,)

        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(fused_emb).squeeze(-1)  # (B,)

        return logits, uncertainty

    def predict(self, fused_emb: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get probability predictions.

        Returns:
            probs: (batch_size,) - fraud probabilities
            uncertainty: (batch_size,) - uncertainty scores (if enabled)
        """
        logits, uncertainty = self.forward(fused_emb)
        probs = torch.sigmoid(logits)

        return probs, uncertainty
