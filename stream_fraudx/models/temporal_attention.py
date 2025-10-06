"""
Temporal-Aware Attention Mechanism for Fraud Detection

Novel contribution: Time-decay attention that emphasizes recent transactions
while maintaining long-range dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalAttention(nn.Module):
    """
    Temporal-aware multi-head attention with time decay.

    Key innovation: Attention scores decay exponentially with time difference,
    making recent transactions more influential while preserving long-term patterns.
    """

    def __init__(self, d_model, num_heads=8, dropout=0.1, max_time_delta=86400):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_time_delta = max_time_delta  # 1 day in seconds

        # Standard attention projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Learnable time decay parameter
        self.time_decay_alpha = nn.Parameter(torch.tensor(1.0))

        # Time embedding for relative positions
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_heads)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, timestamps=None, mask=None):
        """
        Args:
            x: Input features [batch_size, seq_len, d_model]
            timestamps: Transaction timestamps [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            output: Attended features [batch_size, seq_len, d_model]
            attention_weights: For visualization [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Standard attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add temporal decay if timestamps provided
        if timestamps is not None:
            time_decay = self._compute_time_decay(timestamps)
            attention_scores = attention_scores - self.time_decay_alpha * time_decay

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        # Softmax attention
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(attended)

        return output, attention_weights

    def _compute_time_decay(self, timestamps):
        """
        Compute time decay matrix from timestamps.

        Args:
            timestamps: [batch_size, seq_len]

        Returns:
            time_decay: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = timestamps.size()

        # Compute pairwise time differences
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [batch, seq, seq]
        time_diff = torch.abs(time_diff) / self.max_time_delta  # Normalize

        # Embed time differences
        time_features = time_diff.unsqueeze(-1)  # [batch, seq, seq, 1]
        time_decay = self.time_embedding(time_features)  # [batch, seq, seq, num_heads]

        # Reshape to [batch, num_heads, seq, seq]
        time_decay = time_decay.permute(0, 3, 1, 2)

        return time_decay


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block with temporal attention.
    """

    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.temporal_attention = TemporalAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timestamps=None, mask=None):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            timestamps: Timestamps [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            output: Transformed features [batch_size, seq_len, d_model]
        """
        # Temporal attention with residual
        attended, attention_weights = self.temporal_attention(self.norm1(x), timestamps, mask)
        x = x + self.dropout(attended)

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x, attention_weights


class TemporalTransformer(nn.Module):
    """
    Stack of temporal transformer blocks.
    """

    def __init__(self, d_model, num_layers=4, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, timestamps=None, mask=None):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            timestamps: Timestamps [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            output: Transformed features [batch_size, seq_len, d_model]
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []

        for layer in self.layers:
            x, attention_weights = layer(x, timestamps, mask)
            all_attention_weights.append(attention_weights)

        return self.norm(x), all_attention_weights


if __name__ == '__main__':
    # Test temporal attention
    batch_size, seq_len, d_model = 32, 10, 128

    model = TemporalTransformer(d_model, num_layers=2, num_heads=8)

    x = torch.randn(batch_size, seq_len, d_model)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float() * 3600

    output, attention_weights = model(x, timestamps)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
    print(f"Time decay alpha: {model.layers[0].temporal_attention.time_decay_alpha.item():.4f}")
