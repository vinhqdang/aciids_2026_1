"""
STREAM-FraudX Research Model v3

Novel deep learning architecture for fraud detection combining:
1. Multi-scale feature extraction (micro/meso/macro patterns)
2. Temporal-aware attention with time decay
3. Hierarchical representation learning

Research contributions:
- Time-decay attention that emphasizes recent transactions
- Multi-granularity pattern extraction via parallel CNNs
- End-to-end trainable architecture for imbalanced fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_extractor import MultiScaleExtractor
from .temporal_attention import TemporalTransformer


class STREAMFraudXResearch(nn.Module):
    """
    Novel fraud detection architecture combining multi-scale extraction
    and temporal-aware attention.

    Architecture:
    1. Input Embedding: Project raw features to embedding space
    2. Multi-Scale Extraction: Extract patterns at multiple temporal scales
    3. Temporal Transformer: Apply time-aware attention
    4. Classification Head: Predict fraud probability
    """

    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        cnn_channels=64,
        lstm_hidden=128,
        num_transformer_layers=3,
        num_heads=8,
        d_ff=512,
        dropout=0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-scale feature extraction
        self.multiscale_extractor = MultiScaleExtractor(
            input_dim=embedding_dim,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            dropout=dropout
        )

        # Temporal transformer
        multiscale_output_dim = lstm_hidden * 2  # MultiScaleExtractor output dim
        self.temporal_transformer = TemporalTransformer(
            d_model=multiscale_output_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(multiscale_output_dim, multiscale_output_dim // 2),
            nn.LayerNorm(multiscale_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(multiscale_output_dim // 2, multiscale_output_dim // 4),
            nn.LayerNorm(multiscale_output_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(multiscale_output_dim // 4, 1)
        )

    def forward(self, x, timestamps=None, mask=None):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            timestamps: Transaction timestamps [batch_size, seq_len] (optional)
            mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            logits: Fraud prediction logits [batch_size]
            attention_weights: List of attention weights from each layer
        """
        # Embed input features
        x = self.input_embedding(x)  # [batch, seq, embedding_dim]

        # Extract multi-scale features
        x = self.multiscale_extractor(x)  # [batch, seq, lstm_hidden*2]

        # Apply temporal transformer
        x, attention_weights = self.temporal_transformer(x, timestamps, mask)  # [batch, seq, lstm_hidden*2]

        # Global pooling: combine temporal information
        # Use both max and mean pooling for richer representation
        x_max = torch.max(x, dim=1)[0]  # [batch, lstm_hidden*2]
        x_mean = torch.mean(x, dim=1)   # [batch, lstm_hidden*2]
        x_global = x_max + x_mean       # [batch, lstm_hidden*2]

        # Classify
        logits = self.classifier(x_global).squeeze(-1)  # [batch]

        return logits, attention_weights


class STREAMFraudXResearchSimple(nn.Module):
    """
    Simplified version for single-transaction prediction (no temporal sequence).
    Used when sequence length = 1 or for baseline comparison.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            logits: Fraud prediction logits [batch_size]
        """
        logits = self.network(x).squeeze(-1)
        return logits, None


if __name__ == '__main__':
    # Test research model
    print("Testing STREAM-FraudX Research Model...")

    batch_size, seq_len, input_dim = 32, 10, 51

    model = STREAMFraudXResearch(
        input_dim=input_dim,
        embedding_dim=128,
        cnn_channels=64,
        lstm_hidden=128,
        num_transformer_layers=3,
        num_heads=8,
        d_ff=512,
        dropout=0.2
    )

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float() * 3600

    logits, attention_weights = model(x, timestamps)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    print(f"Attention weights shape: {attention_weights[0].shape}")

    # Test simple model
    print("\nTesting STREAM-FraudX Simple Model...")
    simple_model = STREAMFraudXResearchSimple(input_dim=input_dim)
    x_simple = torch.randn(batch_size, input_dim)
    logits_simple, _ = simple_model(x_simple)
    print(f"Simple input shape: {x_simple.shape}")
    print(f"Simple output logits shape: {logits_simple.shape}")

    print("\n✓ All tests passed!")
