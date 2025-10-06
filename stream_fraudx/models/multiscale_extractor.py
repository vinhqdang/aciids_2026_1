"""
Multi-Scale Feature Extraction for Fraud Detection

Novel contribution: Extract fraud patterns at multiple temporal scales
(micro, meso, macro) using parallel CNNs with different receptive fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for extracting fraud patterns at different granularities.

    - Micro scale (kernel=3): Immediate transaction patterns
    - Meso scale (kernel=7): Short-term behavioral patterns
    - Macro scale (kernel=15): Long-term trends
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        # Micro-scale: Local transaction patterns (3-transaction window)
        self.micro_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Meso-scale: Short-term behavioral patterns (7-transaction window)
        self.meso_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Macro-scale: Long-term trends (15-transaction window)
        self.macro_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: Input [batch_size, in_channels, seq_len]

        Returns:
            output: Multi-scale features [batch_size, out_channels, seq_len]
        """
        # Extract features at multiple scales
        micro_features = self.micro_conv(x)
        meso_features = self.meso_conv(x)
        macro_features = self.macro_conv(x)

        # Concatenate and fuse
        multi_scale = torch.cat([micro_features, meso_features, macro_features], dim=1)
        fused = self.fusion(multi_scale)

        return fused


class MultiScaleLSTM(nn.Module):
    """
    Multi-scale LSTM for capturing temporal dependencies at different timescales.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()

        # Short-term LSTM (processes every transaction)
        self.short_lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Long-term LSTM (processes downsampled sequence)
        self.long_lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 4, hidden_size * 2)

    def forward(self, x):
        """
        Args:
            x: Input [batch_size, seq_len, input_size]

        Returns:
            output: Multi-scale LSTM features [batch_size, seq_len, hidden_size * 2]
        """
        batch_size, seq_len, _ = x.size()

        # Short-term: Process full sequence
        short_output, _ = self.short_lstm(x)  # [batch, seq, hidden*2]

        # Long-term: Downsample by 2x and process
        if seq_len >= 4:
            x_downsampled = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=2,
                stride=2
            ).transpose(1, 2)
            long_output, _ = self.long_lstm(x_downsampled)

            # Upsample back to original length
            long_output = F.interpolate(
                long_output.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            long_output, _ = self.long_lstm(x)

        # Fuse short and long term features
        combined = torch.cat([short_output, long_output], dim=-1)
        fused = self.fusion(combined)

        return fused


class MultiScaleExtractor(nn.Module):
    """
    Complete multi-scale feature extractor combining CNN and LSTM.
    """

    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=128, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, cnn_channels)

        # Multi-scale CNN
        self.cnn = MultiScaleCNN(cnn_channels, cnn_channels, dropout)

        # Multi-scale LSTM
        self.lstm = MultiScaleLSTM(cnn_channels, lstm_hidden, num_layers=2, dropout=dropout)

        # Output projection
        self.output_dim = lstm_hidden * 2

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]

        Returns:
            output: Multi-scale features [batch_size, seq_len, output_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq, cnn_channels]

        # CNN processing (needs channel-first)
        x_cnn = x.transpose(1, 2)  # [batch, cnn_channels, seq]
        cnn_features = self.cnn(x_cnn)  # [batch, cnn_channels, seq]
        cnn_features = cnn_features.transpose(1, 2)  # [batch, seq, cnn_channels]

        # LSTM processing
        lstm_features = self.lstm(cnn_features)  # [batch, seq, lstm_hidden*2]

        return lstm_features


if __name__ == '__main__':
    # Test multi-scale extractor
    batch_size, seq_len, input_dim = 32, 20, 15

    model = MultiScaleExtractor(input_dim, cnn_channels=64, lstm_hidden=128)

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {model.output_dim}")

    # Test individual components
    cnn = MultiScaleCNN(64, 64)
    x_cnn = torch.randn(batch_size, 64, seq_len)
    cnn_out = cnn(x_cnn)
    print(f"\nCNN output shape: {cnn_out.shape}")

    lstm = MultiScaleLSTM(64, 128, num_layers=2)
    x_lstm = torch.randn(batch_size, seq_len, 64)
    lstm_out = lstm(x_lstm)
    print(f"LSTM output shape: {lstm_out.shape}")
