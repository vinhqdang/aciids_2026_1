"""
Temporal Graph Tower (TGT) for STREAM-FraudX
Implements a temporal graph network with memory and reservoir sampling for scalable fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict, deque


class Time2Vec(nn.Module):
    """
    Time2Vec time encoding layer.
    Transforms scalar timestamps into rich temporal embeddings.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.randn(dim - 1))
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) timestamps
        Returns:
            (batch_size, dim) temporal embeddings
        """
        t = t.unsqueeze(-1)  # (B, 1)
        # Linear component
        v0 = self.w0 * t + self.b0  # (B, 1)
        # Periodic components
        v_periodic = torch.sin(self.w * t + self.b)  # (B, dim-1)
        return torch.cat([v0, v_periodic], dim=-1)  # (B, dim)


class TemporalMemory(nn.Module):
    """
    Memory module for temporal graph nodes.
    Maintains node states with LRU eviction policy.
    """
    def __init__(self, memory_dim: int, max_size: int = 10000):
        super().__init__()
        self.memory_dim = memory_dim
        self.max_size = max_size
        self.memory = {}  # node_id -> embedding
        self.last_update = {}  # node_id -> timestamp
        self.access_queue = deque(maxlen=max_size)

    def get(self, node_ids: List[int], default_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve node embeddings from memory.
        """
        device = next(iter(self.memory.values())).device if self.memory else \
                 (default_value.device if default_value is not None else torch.device('cpu'))

        embeddings = []
        for node_id in node_ids:
            if node_id in self.memory:
                embeddings.append(self.memory[node_id])
            else:
                if default_value is not None:
                    embeddings.append(default_value)
                else:
                    embeddings.append(torch.zeros(self.memory_dim, device=device))

        return torch.stack(embeddings)

    def update(self, node_ids: List[int], embeddings: torch.Tensor, timestamps: List[float]):
        """
        Update node embeddings in memory with LRU eviction.
        """
        for node_id, emb, ts in zip(node_ids, embeddings, timestamps):
            # Evict LRU if at capacity
            if len(self.memory) >= self.max_size and node_id not in self.memory:
                evict_id = self.access_queue.popleft()
                del self.memory[evict_id]
                del self.last_update[evict_id]

            self.memory[node_id] = emb.detach()
            self.last_update[node_id] = ts
            if node_id in self.access_queue:
                self.access_queue.remove(node_id)
            self.access_queue.append(node_id)

    def clear(self):
        """Clear all memory."""
        self.memory.clear()
        self.last_update.clear()
        self.access_queue.clear()


class ReservoirNeighborSampler:
    """
    Reservoir sampling for temporal neighbors.
    Maintains bounded number of recent neighbors per node.
    """
    def __init__(self, max_neighbors: int = 20):
        self.max_neighbors = max_neighbors
        self.neighbors = defaultdict(list)  # node_id -> [(neighbor_id, timestamp, edge_attrs)]

    def add_edge(self, src: int, dst: int, timestamp: float, edge_attrs: torch.Tensor):
        """Add edge to both directions."""
        self._add_directed_edge(src, dst, timestamp, edge_attrs)
        self._add_directed_edge(dst, src, timestamp, edge_attrs)

    def _add_directed_edge(self, src: int, dst: int, timestamp: float, edge_attrs: torch.Tensor):
        """Add directed edge with reservoir sampling."""
        neighbors = self.neighbors[src]

        if len(neighbors) < self.max_neighbors:
            neighbors.append((dst, timestamp, edge_attrs))
        else:
            # Reservoir sampling: replace random element
            idx = np.random.randint(0, len(neighbors))
            neighbors[idx] = (dst, timestamp, edge_attrs)

        # Keep sorted by timestamp (most recent first)
        neighbors.sort(key=lambda x: x[1], reverse=True)

    def get_neighbors(self, node_id: int, k: Optional[int] = None) -> List[Tuple[int, float, torch.Tensor]]:
        """Get k most recent neighbors."""
        neighbors = self.neighbors.get(node_id, [])
        if k is None:
            k = self.max_neighbors
        return neighbors[:k]

    def clear(self):
        """Clear all neighbor data."""
        self.neighbors.clear()


class TemporalMessageFunction(nn.Module):
    """
    Temporal message function for aggregating neighbor information.
    """
    def __init__(self, node_dim: int, edge_dim: int, time_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self,
                src_emb: torch.Tensor,
                dst_emb: torch.Tensor,
                edge_attrs: torch.Tensor,
                time_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute message for edge.
        Args:
            src_emb: (batch_size, node_dim)
            dst_emb: (batch_size, node_dim)
            edge_attrs: (batch_size, edge_dim)
            time_emb: (batch_size, time_dim)
        Returns:
            message: (batch_size, node_dim)
        """
        concat = torch.cat([src_emb, dst_emb, edge_attrs, time_emb], dim=-1)
        return self.mlp(concat)


class TemporalGraphTower(nn.Module):
    """
    Temporal Graph Tower for fraud detection.
    Processes evolving transaction graphs with temporal patterns.
    """
    def __init__(self,
                 node_dim: int = 128,
                 edge_dim: int = 64,
                 time_dim: int = 32,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 max_neighbors: int = 20,
                 memory_size: int = 10000):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.num_layers = num_layers

        # Components
        self.time2vec = Time2Vec(time_dim)
        self.memory = TemporalMemory(node_dim, memory_size)
        self.sampler = ReservoirNeighborSampler(max_neighbors)

        # Message functions for each layer
        self.message_functions = nn.ModuleList([
            TemporalMessageFunction(node_dim, edge_dim, time_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # GRU for temporal aggregation
        self.gru = nn.GRUCell(node_dim, node_dim)

        # Edge attribute encoder
        self.edge_encoder = nn.Linear(edge_dim, edge_dim)

    def forward(self,
                batch_events: Dict[str, torch.Tensor],
                update_memory: bool = True) -> torch.Tensor:
        """
        Forward pass for batch of events.

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

        # Get current embeddings from memory
        src_emb = self.memory.get(src_nodes.tolist())  # (B, node_dim)
        dst_emb = self.memory.get(dst_nodes.tolist())  # (B, node_dim)
        src_emb = src_emb.to(device)
        dst_emb = dst_emb.to(device)

        # Multi-layer message passing
        for layer_idx in range(self.num_layers):
            # Aggregate messages from neighbors
            src_messages = self._aggregate_neighbor_messages(
                src_nodes.tolist(), timestamps.tolist(), layer_idx, device
            )
            dst_messages = self._aggregate_neighbor_messages(
                dst_nodes.tolist(), timestamps.tolist(), layer_idx, device
            )

            # Compute edge messages
            edge_message_src = self.message_functions[layer_idx](
                src_emb, dst_emb, edge_encoded, time_emb
            )
            edge_message_dst = self.message_functions[layer_idx](
                dst_emb, src_emb, edge_encoded, time_emb
            )

            # Combine messages
            total_message_src = edge_message_src + src_messages
            total_message_dst = edge_message_dst + dst_messages

            # Update with GRU
            src_emb = self.gru(total_message_src, src_emb)
            dst_emb = self.gru(total_message_dst, dst_emb)

        # Update memory if requested
        if update_memory:
            self.memory.update(src_nodes.tolist(), src_emb, timestamps.tolist())
            self.memory.update(dst_nodes.tolist(), dst_emb, timestamps.tolist())

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

    def _aggregate_neighbor_messages(self,
                                     node_ids: List[int],
                                     timestamps: List[float],
                                     layer_idx: int,
                                     device: torch.device) -> torch.Tensor:
        """
        Aggregate messages from temporal neighbors.
        """
        batch_messages = []

        for node_id, curr_time in zip(node_ids, timestamps):
            neighbors = self.sampler.get_neighbors(node_id)

            if not neighbors:
                # No neighbors, return zero message
                batch_messages.append(torch.zeros(self.node_dim, device=device))
                continue

            # Get neighbor embeddings and compute messages
            neighbor_ids = [n[0] for n in neighbors]
            neighbor_times = torch.tensor([n[1] for n in neighbors], device=device)
            neighbor_edge_attrs = torch.stack([n[2] for n in neighbors]).to(device)

            neighbor_embs = self.memory.get(neighbor_ids).to(device)
            node_emb = self.memory.get([node_id]).to(device).repeat(len(neighbors), 1)

            # Compute time embeddings
            time_embs = self.time2vec(neighbor_times)

            # Compute messages
            messages = self.message_functions[layer_idx](
                neighbor_embs, node_emb, neighbor_edge_attrs, time_embs
            )

            # Aggregate with attention (simple mean for now)
            aggregated = messages.mean(dim=0)
            batch_messages.append(aggregated)

        return torch.stack(batch_messages)

    def reset(self):
        """Reset memory and neighbor sampler."""
        self.memory.clear()
        self.sampler.clear()
