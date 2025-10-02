"""
Fusion modules for STREAM-FraudX
Implements gated cross-attention fusion between TGT and TTT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionFusion(nn.Module):
    """
    Gated cross-attention fusion module.
    Fuses relational (graph) and tabular signals with attention mechanism.
    """
    def __init__(self,
                 graph_dim: int,
                 tabular_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.graph_dim = graph_dim
        self.tabular_dim = tabular_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project to common dimension
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        # Cross-attention: graph -> tabular
        self.cross_attn_g2t = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: tabular -> graph
        self.cross_attn_t2g = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Gating mechanisms
        self.gate_g2t = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_t2g = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Layer norms
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_tabular = nn.LayerNorm(hidden_dim)

        # Output fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, graph_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_emb: (batch_size, graph_dim) from TGT
            tabular_emb: (batch_size, tabular_dim) from TTT

        Returns:
            fused_emb: (batch_size, hidden_dim) fused representation
        """
        batch_size = graph_emb.size(0)

        # Project to common dimension
        g = self.graph_proj(graph_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        t = self.tabular_proj(tabular_emb).unsqueeze(1)  # (B, 1, hidden_dim)

        # Cross-attention: query from tabular, key/value from graph
        t2g_attn, _ = self.cross_attn_t2g(t, g, g)  # (B, 1, hidden_dim)
        t2g_attn = t2g_attn.squeeze(1)  # (B, hidden_dim)

        # Cross-attention: query from graph, key/value from tabular
        g2t_attn, _ = self.cross_attn_g2t(g, t, t)  # (B, 1, hidden_dim)
        g2t_attn = g2t_attn.squeeze(1)  # (B, hidden_dim)

        # Gating for tabular enhancement
        g_squeeze = g.squeeze(1)
        t_squeeze = t.squeeze(1)

        gate_t = self.gate_g2t(torch.cat([t_squeeze, g2t_attn], dim=-1))
        enhanced_t = self.norm_tabular(t_squeeze + gate_t * g2t_attn)

        # Gating for graph enhancement
        gate_g = self.gate_t2g(torch.cat([g_squeeze, t2g_attn], dim=-1))
        enhanced_g = self.norm_graph(g_squeeze + gate_g * t2g_attn)

        # Final fusion
        fused = torch.cat([enhanced_g, enhanced_t], dim=-1)  # (B, 2*hidden_dim)
        output = self.fusion_mlp(fused)  # (B, hidden_dim)

        return output


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion (for ablation studies).
    """
    def __init__(self, graph_dim: int, tabular_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, graph_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_emb: (batch_size, graph_dim)
            tabular_emb: (batch_size, tabular_dim)

        Returns:
            fused_emb: (batch_size, hidden_dim)
        """
        concat = torch.cat([graph_emb, tabular_emb], dim=-1)
        return self.mlp(concat)


class FraudDetectionHead(nn.Module):
    """
    Detection head for fraud classification.
    Two-layer MLP with sigmoid output.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_emb: (batch_size, input_dim)

        Returns:
            scores: (batch_size,) fraud probability scores (before sigmoid)
        """
        logits = self.head(fused_emb).squeeze(-1)  # (B,)
        return logits

    def predict(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        """
        logits = self.forward(fused_emb)
        return torch.sigmoid(logits)
