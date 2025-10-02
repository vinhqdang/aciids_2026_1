"""
Tabular Transformer Tower (TTT) for STREAM-FraudX
Processes raw transaction features with transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class FeatureTokenizer(nn.Module):
    """
    Tokenizes heterogeneous tabular features for transformer input.
    Handles continuous, categorical, and temporal features.
    """
    def __init__(self,
                 continuous_dims: List[int],
                 categorical_vocab_sizes: List[int],
                 embedding_dim: int = 64,
                 num_bins: int = 50):
        super().__init__()

        self.continuous_dims = continuous_dims
        self.categorical_vocab_sizes = categorical_vocab_sizes
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins

        # Learned binning for continuous features
        self.bin_embeddings = nn.ModuleList([
            nn.Embedding(num_bins, embedding_dim) for _ in continuous_dims
        ])

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in categorical_vocab_sizes
        ])

        # Projection to unify dimensions
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, continuous_features: torch.Tensor,
                categorical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_features: (batch_size, num_continuous)
            categorical_features: (batch_size, num_categorical)

        Returns:
            token_embeddings: (batch_size, num_features, embedding_dim)
        """
        batch_size = continuous_features.size(0)
        tokens = []

        # Process continuous features with learned binning
        for i, (feat, bin_emb) in enumerate(zip(
            continuous_features.T, self.bin_embeddings
        )):
            # Discretize into bins
            bin_indices = torch.clamp(
                (feat * self.num_bins).long(),
                0, self.num_bins - 1
            )
            tokens.append(bin_emb(bin_indices))

        # Process categorical features
        for i, (feat, cat_emb) in enumerate(zip(
            categorical_features.T, self.cat_embeddings
        )):
            tokens.append(cat_emb(feat.long()))

        # Stack and project
        token_embeddings = torch.stack(tokens, dim=1)  # (B, num_features, emb_dim)
        return self.projection(token_embeddings)


class FourierTimeEncoding(nn.Module):
    """
    Fourier-based time encoding for cyclical temporal patterns.
    """
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        # Learnable frequencies
        self.freq = nn.Parameter(torch.randn(dim // 2))
        self.phase = nn.Parameter(torch.randn(dim // 2))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch_size,)
        Returns:
            time_encoding: (batch_size, dim)
        """
        t = timestamps.unsqueeze(-1)  # (B, 1)
        # Compute sin and cos components
        angle = 2 * math.pi * self.freq * t + self.phase
        sin_comp = torch.sin(angle)
        cos_comp = torch.cos(angle)
        return torch.cat([sin_comp, cos_comp], dim=-1)  # (B, dim)


class TransformerBlock(nn.Module):
    """
    Standard transformer block with multi-head attention and FFN.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            output: (batch_size, seq_len, dim)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TabularTransformerTower(nn.Module):
    """
    Tabular Transformer Tower for processing transaction features.
    Uses tokenization and transformer layers for rich feature representation.
    """
    def __init__(self,
                 continuous_dims: List[int],
                 categorical_vocab_sizes: List[int],
                 embedding_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 time_encoding_dim: int = 32):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_features = len(continuous_dims) + len(categorical_vocab_sizes)

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(
            continuous_dims, categorical_vocab_sizes,
            embedding_dim, num_bins=50
        )

        # Time encoding
        self.time_encoder = FourierTimeEncoding(time_encoding_dim)
        self.time_projection = nn.Linear(time_encoding_dim, embedding_dim)

        # Positional encoding for features
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_features + 1, embedding_dim)  # +1 for time token
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, batch_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for tabular features.

        Args:
            batch_features: Dictionary containing:
                - continuous: (batch_size, num_continuous)
                - categorical: (batch_size, num_categorical)
                - timestamps: (batch_size,)

        Returns:
            representations: (batch_size, embedding_dim) global feature representation
        """
        batch_size = batch_features['continuous'].size(0)
        device = batch_features['continuous'].device

        # Tokenize features
        feature_tokens = self.tokenizer(
            batch_features['continuous'],
            batch_features['categorical']
        )  # (B, num_features, emb_dim)

        # Encode time
        time_encoding = self.time_encoder(batch_features['timestamps'])
        time_token = self.time_projection(time_encoding).unsqueeze(1)  # (B, 1, emb_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, emb_dim)

        # Concatenate all tokens
        tokens = torch.cat([cls_tokens, feature_tokens, time_token], dim=1)  # (B, num_features+2, emb_dim)

        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :tokens.size(1), :]
        tokens = tokens + pos_emb

        # Pass through transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Normalize
        tokens = self.norm(tokens)

        # Return CLS token representation
        return tokens[:, 0, :]  # (B, emb_dim)


class SimpleTabularEncoder(nn.Module):
    """
    Simplified MLP-based tabular encoder for baseline comparisons.
    """
    def __init__(self,
                 continuous_dim: int,
                 categorical_vocab_sizes: List[int],
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = [256, 128]):
        super().__init__()

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in categorical_vocab_sizes
        ])

        # Calculate total input dimension
        total_cat_dim = len(categorical_vocab_sizes) * embedding_dim
        input_dim = continuous_dim + total_cat_dim

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: (batch_size, continuous_dim)
            categorical: (batch_size, num_categorical)
        Returns:
            encoding: (batch_size, output_dim)
        """
        # Embed categorical features
        cat_embeddings = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_embeddings.append(emb_layer(categorical[:, i].long()))

        # Concatenate all features
        if cat_embeddings:
            cat_concat = torch.cat(cat_embeddings, dim=-1)
            features = torch.cat([continuous, cat_concat], dim=-1)
        else:
            features = continuous

        return self.mlp(features)
