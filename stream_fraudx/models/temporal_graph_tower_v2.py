"""
Enhanced Temporal Graph Tower (v2) for STREAM-FraudX
Implements recency-weighted attention and hot-node caching for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

from .temporal_graph_tower import Time2Vec, TemporalMemory, ReservoirNeighborSampler


class RecencyWeightedAttention(nn.Module):
    """
    Recency-weighted attention mechanism for temporal neighbors.

    Combines learned attention weights with exponential temporal decay.
    """

    def __init__(self, node_dim: int, time_dim: int, num_heads: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"

        # Multi-head attention components
        self.query = nn.Linear(node_dim, node_dim)
        self.key = nn.Linear(node_dim + time_dim, node_dim)
        self.value = nn.Linear(node_dim, node_dim)
        self.output = nn.Linear(node_dim, node_dim)

        # Learnable temporal decay rate
        self.temporal_decay = nn.Parameter(torch.tensor(0.1))

        # Layer norm
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self,
                query_emb: torch.Tensor,
                neighbor_embs: torch.Tensor,
                time_deltas: torch.Tensor,
                time_embs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute recency-weighted attention over neighbors.

        Args:
            query_emb: (batch_size, node_dim) - center node embeddings
            neighbor_embs: (batch_size, num_neighbors, node_dim) - neighbor embeddings
            time_deltas: (batch_size, num_neighbors) - time differences (current - neighbor time)
            time_embs: (batch_size, num_neighbors, time_dim) - temporal embeddings
            mask: (batch_size, num_neighbors) - attention mask (1 = valid, 0 = padding)

        Returns:
            attended: (batch_size, node_dim) - attended neighbor representation
        """
        batch_size, num_neighbors, _ = neighbor_embs.shape

        # Reshape for multi-head attention
        Q = self.query(query_emb).view(batch_size, 1, self.num_heads, self.head_dim)

        # Concatenate neighbor embeddings with time embeddings
        neighbor_time_concat = torch.cat([neighbor_embs, time_embs], dim=-1)
        K = self.key(neighbor_time_concat).view(batch_size, num_neighbors, self.num_heads, self.head_dim)
        V = self.value(neighbor_embs).view(batch_size, num_neighbors, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (B, num_heads, 1, head_dim)
        K = K.transpose(1, 2)  # (B, num_heads, num_neighbors, head_dim)
        V = V.transpose(1, 2)  # (B, num_heads, num_neighbors, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.squeeze(2)  # (B, num_heads, num_neighbors)

        # Apply temporal decay: weight = exp(-lambda * time_delta)
        # Clamp decay rate to prevent numerical instability
        decay_rate = torch.sigmoid(self.temporal_decay) * 0.5  # Limit to [0, 0.5]
        temporal_weights = torch.exp(-decay_rate * time_deltas.unsqueeze(1))  # (B, 1, num_neighbors)

        # Combine learned attention with temporal decay
        scores = scores + temporal_weights.expand_as(scores)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(scores)  # (B, num_heads, num_neighbors)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, num_neighbors)

        # Apply attention to values
        attn_weights = attn_weights.unsqueeze(2)  # (B, num_heads, 1, num_neighbors)
        V = V.transpose(2, 3)  # (B, num_heads, head_dim, num_neighbors)
        attended = torch.matmul(attn_weights, V.transpose(-2, -1))  # (B, num_heads, 1, head_dim)

        # Reshape and project
        attended = attended.squeeze(2).transpose(1, 2).contiguous()  # (B, 1, num_heads, head_dim)
        attended = attended.view(batch_size, -1)  # (B, node_dim)
        attended = self.output(attended)

        # Residual connection and layer norm
        attended = self.layer_norm(attended + query_emb)

        return attended


class HotNodeCache(nn.Module):
    """
    GPU-friendly cache for frequently accessed nodes.

    Maintains embeddings for hot nodes (high-degree, frequently accessed)
    to reduce memory lookups.
    """

    def __init__(self, node_dim: int, cache_size: int = 1000):
        super().__init__()
        self.node_dim = node_dim
        self.cache_size = cache_size

        # Cache storage (on GPU)
        self.register_buffer('cache_embeddings', torch.zeros(cache_size, node_dim))
        self.register_buffer('cache_node_ids', torch.full((cache_size,), -1, dtype=torch.long))
        self.register_buffer('cache_valid', torch.zeros(cache_size, dtype=torch.bool))

        # CPU-side mapping for fast lookup
        self.node_to_cache_idx = {}  # node_id -> cache_index
        self.next_eviction_idx = 0

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, node_ids: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached embeddings for nodes.

        Returns:
            embeddings: (batch_size, node_dim) - embeddings (zeros for misses)
            hit_mask: (batch_size,) - boolean mask indicating cache hits
        """
        batch_size = len(node_ids)
        embeddings = torch.zeros(batch_size, self.node_dim, device=device)
        hit_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i, node_id in enumerate(node_ids):
            if node_id in self.node_to_cache_idx:
                cache_idx = self.node_to_cache_idx[node_id]
                if self.cache_valid[cache_idx]:
                    embeddings[i] = self.cache_embeddings[cache_idx]
                    hit_mask[i] = True
                    self.hits += 1
                else:
                    self.misses += 1
            else:
                self.misses += 1

        return embeddings, hit_mask

    def put(self, node_ids: List[int], embeddings: torch.Tensor):
        """
        Update cache with new embeddings.

        Uses FIFO eviction policy for simplicity.
        """
        for node_id, emb in zip(node_ids, embeddings):
            if node_id in self.node_to_cache_idx:
                # Update existing entry
                cache_idx = self.node_to_cache_idx[node_id]
            else:
                # Evict oldest entry if cache is full
                if len(self.node_to_cache_idx) >= self.cache_size:
                    # Find entry to evict
                    cache_idx = self.next_eviction_idx
                    old_node_id = self.cache_node_ids[cache_idx].item()
                    if old_node_id in self.node_to_cache_idx:
                        del self.node_to_cache_idx[old_node_id]

                    self.next_eviction_idx = (self.next_eviction_idx + 1) % self.cache_size
                else:
                    cache_idx = len(self.node_to_cache_idx)

                self.node_to_cache_idx[node_id] = cache_idx

            # Update cache
            self.cache_embeddings[cache_idx] = emb.detach()
            self.cache_node_ids[cache_idx] = node_id
            self.cache_valid[cache_idx] = True

    def clear(self):
        """Clear the cache."""
        self.cache_valid.fill_(False)
        self.cache_node_ids.fill_(-1)
        self.node_to_cache_idx.clear()
        self.next_eviction_idx = 0
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'cache_size': len(self.node_to_cache_idx),
            'max_size': self.cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class EnhancedTemporalGraphTower(nn.Module):
    """
    Enhanced Temporal Graph Tower with recency-weighted attention and hot-node caching.

    Improvements over v1:
    - Recency-weighted multi-head attention instead of mean pooling
    - GPU-friendly hot-node cache for frequently accessed nodes
    - Learnable temporal decay rates
    - Better gradient flow with residual connections
    """

    def __init__(self,
                 node_dim: int = 128,
                 edge_dim: int = 64,
                 time_dim: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 max_neighbors: int = 20,
                 memory_size: int = 10000,
                 cache_size: int = 1000,
                 use_hot_cache: bool = True):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.use_hot_cache = use_hot_cache

        # Components
        self.time2vec = Time2Vec(time_dim)
        self.memory = TemporalMemory(node_dim, memory_size)
        self.sampler = ReservoirNeighborSampler(max_neighbors)

        # Hot node cache
        if use_hot_cache:
            self.hot_cache = HotNodeCache(node_dim, cache_size)
        else:
            self.hot_cache = None

        # Recency-weighted attention for each layer
        self.attentions = nn.ModuleList([
            RecencyWeightedAttention(node_dim, time_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Message functions for edge updates
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim + time_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, node_dim),
                nn.LayerNorm(node_dim)
            )
            for _ in range(num_layers)
        ])

        # GRU for temporal updates
        self.gru = nn.GRUCell(node_dim, node_dim)

        # Edge attribute encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self,
                batch_events: Dict[str, torch.Tensor],
                update_memory: bool = True) -> torch.Tensor:
        """
        Forward pass with enhanced attention mechanism.

        Args:
            batch_events: Dictionary containing:
                - src_nodes: (batch_size,) source node IDs
                - dst_nodes: (batch_size,) destination node IDs
                - edge_attrs: (batch_size, edge_dim) edge attributes
                - timestamps: (batch_size,) timestamps
            update_memory: Whether to update node memory

        Returns:
            node_embeddings: (batch_size, 2*node_dim) concatenated src and dst embeddings
        """
        src_nodes = batch_events['src_nodes']
        dst_nodes = batch_events['dst_nodes']
        edge_attrs = batch_events['edge_attrs']
        timestamps = batch_events['timestamps']

        batch_size = src_nodes.size(0)
        device = edge_attrs.device

        # Encode timestamps
        time_emb = self.time2vec(timestamps)  # (B, time_dim)

        # Encode edge attributes
        edge_encoded = self.edge_encoder(edge_attrs)  # (B, edge_dim)

        # Get current embeddings (try cache first, then memory)
        src_emb = self._get_embeddings(src_nodes.tolist(), device)
        dst_emb = self._get_embeddings(dst_nodes.tolist(), device)

        # Multi-layer message passing with attention
        for layer_idx in range(self.num_layers):
            # Aggregate messages from neighbors with recency-weighted attention
            src_attended = self._attend_neighbors(
                src_nodes.tolist(), timestamps.tolist(), src_emb, layer_idx, device
            )
            dst_attended = self._attend_neighbors(
                dst_nodes.tolist(), timestamps.tolist(), dst_emb, layer_idx, device
            )

            # Compute edge messages
            edge_message_src = self._compute_edge_message(
                src_emb, dst_emb, edge_encoded, time_emb, layer_idx
            )
            edge_message_dst = self._compute_edge_message(
                dst_emb, src_emb, edge_encoded, time_emb, layer_idx
            )

            # Combine messages
            total_message_src = edge_message_src + src_attended
            total_message_dst = edge_message_dst + dst_attended

            # Update with GRU
            src_emb = self.gru(total_message_src, src_emb)
            dst_emb = self.gru(total_message_dst, dst_emb)

        # Update memory and cache
        if update_memory:
            self.memory.update(src_nodes.tolist(), src_emb, timestamps.tolist())
            self.memory.update(dst_nodes.tolist(), dst_emb, timestamps.tolist())

            if self.hot_cache is not None:
                self.hot_cache.put(src_nodes.tolist(), src_emb)
                self.hot_cache.put(dst_nodes.tolist(), dst_emb)

            # Update neighbor sampler
            for i in range(batch_size):
                self.sampler.add_edge(
                    src_nodes[i].item(),
                    dst_nodes[i].item(),
                    timestamps[i].item(),
                    edge_encoded[i]
                )

        # Concatenate source and destination embeddings
        return torch.cat([src_emb, dst_emb], dim=-1)  # (B, 2*node_dim)

    def _get_embeddings(self, node_ids: List[int], device: torch.device) -> torch.Tensor:
        """Get embeddings from cache or memory."""
        if self.hot_cache is not None:
            # Try cache first
            cached_embs, hit_mask = self.hot_cache.get(node_ids, device)

            # Fill misses from memory
            miss_indices = (~hit_mask).nonzero(as_tuple=True)[0]
            if len(miss_indices) > 0:
                miss_node_ids = [node_ids[i] for i in miss_indices.tolist()]
                memory_embs = self.memory.get(miss_node_ids).to(device)
                cached_embs[miss_indices] = memory_embs

            return cached_embs
        else:
            # Just use memory
            return self.memory.get(node_ids).to(device)

    def _attend_neighbors(self,
                         node_ids: List[int],
                         timestamps: List[float],
                         query_embs: torch.Tensor,
                         layer_idx: int,
                         device: torch.device) -> torch.Tensor:
        """Attend to temporal neighbors with recency weighting."""
        batch_size = len(node_ids)
        attended = []

        for i, (node_id, curr_time, query_emb) in enumerate(zip(node_ids, timestamps, query_embs)):
            neighbors = self.sampler.get_neighbors(node_id)

            if not neighbors:
                # No neighbors, return zeros
                attended.append(torch.zeros_like(query_emb))
                continue

            # Extract neighbor info
            neighbor_ids = [n[0] for n in neighbors]
            neighbor_times = torch.tensor([n[1] for n in neighbors], device=device)
            neighbor_edge_attrs = torch.stack([n[2] for n in neighbors]).to(device)

            # Get neighbor embeddings
            neighbor_embs = self._get_embeddings(neighbor_ids, device)

            # Compute time deltas (current_time - neighbor_time)
            time_deltas = curr_time - neighbor_times

            # Compute time embeddings
            time_embs = self.time2vec(neighbor_times)

            # Apply recency-weighted attention
            neighbor_embs_batch = neighbor_embs.unsqueeze(0)  # (1, num_neighbors, node_dim)
            time_deltas_batch = time_deltas.unsqueeze(0)  # (1, num_neighbors)
            time_embs_batch = time_embs.unsqueeze(0)  # (1, num_neighbors, time_dim)
            query_emb_batch = query_emb.unsqueeze(0)  # (1, node_dim)

            attended_emb = self.attentions[layer_idx](
                query_emb_batch,
                neighbor_embs_batch,
                time_deltas_batch,
                time_embs_batch
            ).squeeze(0)

            attended.append(attended_emb)

        return torch.stack(attended)

    def _compute_edge_message(self,
                             src_emb: torch.Tensor,
                             dst_emb: torch.Tensor,
                             edge_encoded: torch.Tensor,
                             time_emb: torch.Tensor,
                             layer_idx: int) -> torch.Tensor:
        """Compute message for current edge."""
        concat = torch.cat([src_emb, dst_emb, edge_encoded, time_emb], dim=-1)
        return self.message_mlps[layer_idx](concat)

    def reset(self):
        """Reset memory, sampler, and cache."""
        self.memory.clear()
        self.sampler.clear()
        if self.hot_cache is not None:
            self.hot_cache.clear()

    def get_stats(self) -> Dict:
        """Get tower statistics."""
        stats = {
            'memory_size': len(self.memory.memory),
            'num_edges': sum(len(neighbors) for neighbors in self.sampler.neighbors.values())
        }

        if self.hot_cache is not None:
            stats['cache'] = self.hot_cache.get_stats()

        return stats
