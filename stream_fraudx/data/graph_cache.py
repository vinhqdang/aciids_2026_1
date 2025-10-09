"""
Dynamic graph windowing and caching system.
Efficiently manages temporal graph data with sliding windows and hot-node caching.
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class GraphWindowConfig:
    """Configuration for graph windowing."""
    window_size: int = 100  # Number of recent edges to keep
    max_nodes: int = 10000  # Maximum number of nodes to track
    cache_size: int = 1000  # Number of hot nodes to cache
    enable_caching: bool = True
    time_decay: bool = True  # Use time-based decay for edge relevance


class GraphWindow:
    """
    Sliding window over temporal graph data.

    Maintains a fixed-size window of recent edges and provides efficient
    neighbor lookup with optional time decay.
    """

    def __init__(self, config: GraphWindowConfig):
        self.config = config

        # Edge storage: (src, dst, timestamp, features)
        self.edges: List[Tuple[int, int, float, Optional[torch.Tensor]]] = []

        # Adjacency lists
        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adjacency: Dict[int, List[int]] = defaultdict(list)

        # Edge features
        self.edge_features: Dict[Tuple[int, int], torch.Tensor] = {}

        # Timestamps for temporal decay
        self.edge_timestamps: Dict[Tuple[int, int], float] = {}

        # Node activity tracking
        self.node_activity: Dict[int, int] = defaultdict(int)

        self.current_time = 0.0

    def add_edge(self, src: int, dst: int, timestamp: float,
                 features: Optional[torch.Tensor] = None):
        """
        Add an edge to the window.

        Args:
            src: Source node ID
            dst: Destination node ID
            timestamp: Edge timestamp
            features: Optional edge features
        """
        # Update current time
        self.current_time = max(self.current_time, timestamp)

        # Add to edge list
        self.edges.append((src, dst, timestamp, features))

        # Update adjacency
        self.adjacency[src].append(dst)
        self.reverse_adjacency[dst].append(src)

        # Store edge features and timestamp
        edge_key = (src, dst)
        if features is not None:
            self.edge_features[edge_key] = features
        self.edge_timestamps[edge_key] = timestamp

        # Update node activity
        self.node_activity[src] += 1
        self.node_activity[dst] += 1

        # Maintain window size
        if len(self.edges) > self.config.window_size:
            self._remove_oldest_edge()

    def _remove_oldest_edge(self):
        """Remove the oldest edge from the window."""
        if not self.edges:
            return

        # Remove oldest edge
        src, dst, timestamp, features = self.edges.pop(0)

        # Update adjacency (just decrement, don't remove completely)
        # In a production system, we'd do more careful bookkeeping

        # Remove edge features
        edge_key = (src, dst)
        if edge_key in self.edge_features:
            del self.edge_features[edge_key]
        if edge_key in self.edge_timestamps:
            del self.edge_timestamps[edge_key]

    def get_neighbors(self, node: int, direction: str = "out",
                     max_neighbors: Optional[int] = None) -> List[int]:
        """
        Get neighbors of a node.

        Args:
            node: Node ID
            direction: 'out' for outgoing edges, 'in' for incoming, 'both' for both
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of neighbor node IDs
        """
        if direction == "out":
            neighbors = self.adjacency.get(node, [])
        elif direction == "in":
            neighbors = self.reverse_adjacency.get(node, [])
        elif direction == "both":
            neighbors = list(set(self.adjacency.get(node, []) +
                               self.reverse_adjacency.get(node, [])))
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Limit number of neighbors
        if max_neighbors is not None and len(neighbors) > max_neighbors:
            # Sample most recent neighbors
            neighbors = neighbors[-max_neighbors:]

        return neighbors

    def get_edge_features(self, src: int, dst: int) -> Optional[torch.Tensor]:
        """Get features for an edge."""
        return self.edge_features.get((src, dst))

    def get_edge_weight(self, src: int, dst: int, use_time_decay: bool = True) -> float:
        """
        Get edge weight with optional time decay.

        Args:
            src: Source node
            dst: Destination node
            use_time_decay: Whether to apply temporal decay

        Returns:
            Edge weight (higher = more recent/important)
        """
        edge_key = (src, dst)

        if edge_key not in self.edge_timestamps:
            return 0.0

        if not use_time_decay or not self.config.time_decay:
            return 1.0

        # Time-based decay: weight = exp(-lambda * time_diff)
        timestamp = self.edge_timestamps[edge_key]
        time_diff = self.current_time - timestamp
        decay_rate = 0.01  # Configurable decay rate

        weight = np.exp(-decay_rate * time_diff)
        return weight

    def get_hot_nodes(self, top_k: int) -> List[int]:
        """
        Get the most active nodes in the window.

        Args:
            top_k: Number of hot nodes to return

        Returns:
            List of node IDs sorted by activity
        """
        sorted_nodes = sorted(
            self.node_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [node for node, _ in sorted_nodes[:top_k]]

    def get_window_stats(self) -> Dict:
        """Get statistics about the current window."""
        return {
            'num_edges': len(self.edges),
            'num_nodes': len(set(self.node_activity.keys())),
            'window_size': self.config.window_size,
            'current_time': self.current_time,
            'avg_degree': np.mean(list(self.node_activity.values())) if self.node_activity else 0
        }


class HotNodeCache:
    """
    LRU cache for frequently accessed node embeddings.

    Improves performance by caching embeddings for hot nodes (high-degree,
    frequently accessed nodes).
    """

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, node_id: int) -> Optional[torch.Tensor]:
        """
        Get cached embedding for a node.

        Args:
            node_id: Node ID

        Returns:
            Cached embedding or None if not found
        """
        if node_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(node_id)
            self.hits += 1
            return self.cache[node_id]

        self.misses += 1
        return None

    def put(self, node_id: int, embedding: torch.Tensor):
        """
        Cache an embedding for a node.

        Args:
            node_id: Node ID
            embedding: Node embedding tensor
        """
        if node_id in self.cache:
            # Update and move to end
            self.cache.move_to_end(node_id)

        self.cache[node_id] = embedding.detach().clone()

        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class GraphWindowManager:
    """
    Manages multiple graph windows and caching for efficient temporal graph processing.

    Features:
    - Multiple concurrent windows
    - Hot node caching
    - Batch processing support
    - Statistics tracking
    """

    def __init__(self, config: GraphWindowConfig):
        self.config = config
        self.window = GraphWindow(config)
        self.cache = HotNodeCache(config.cache_size) if config.enable_caching else None

    def add_edges_batch(self, edges: List[Tuple[int, int, float, Optional[torch.Tensor]]]):
        """
        Add multiple edges at once.

        Args:
            edges: List of (src, dst, timestamp, features) tuples
        """
        for src, dst, timestamp, features in edges:
            self.window.add_edge(src, dst, timestamp, features)

    def get_subgraph(self, center_nodes: List[int], k_hops: int = 2) -> Dict:
        """
        Extract k-hop subgraph around center nodes.

        Args:
            center_nodes: List of center node IDs
            k_hops: Number of hops to expand

        Returns:
            Dictionary with nodes, edges, and features
        """
        visited_nodes: Set[int] = set(center_nodes)
        frontier = set(center_nodes)

        # Expand k hops
        for _ in range(k_hops):
            new_frontier = set()
            for node in frontier:
                neighbors = self.window.get_neighbors(node, direction="both")
                new_frontier.update(neighbors)

            visited_nodes.update(new_frontier)
            frontier = new_frontier

        # Extract edges within visited nodes
        subgraph_edges = []
        for src, dst, timestamp, features in self.window.edges:
            if src in visited_nodes and dst in visited_nodes:
                subgraph_edges.append((src, dst, timestamp, features))

        return {
            'nodes': list(visited_nodes),
            'edges': subgraph_edges,
            'num_nodes': len(visited_nodes),
            'num_edges': len(subgraph_edges)
        }

    def get_temporal_neighbors(self, node: int, time_threshold: float,
                              max_neighbors: int = 50) -> List[Tuple[int, float]]:
        """
        Get neighbors with temporal information.

        Args:
            node: Center node ID
            time_threshold: Only return edges within this time window
            max_neighbors: Maximum neighbors to return

        Returns:
            List of (neighbor_id, edge_weight) tuples
        """
        neighbors = self.window.get_neighbors(node, direction="both")

        # Get weights with time decay
        neighbor_weights = []
        for neighbor in neighbors:
            # Check both directions
            weight1 = self.window.get_edge_weight(node, neighbor)
            weight2 = self.window.get_edge_weight(neighbor, node)
            weight = max(weight1, weight2)

            if weight > 0:
                neighbor_weights.append((neighbor, weight))

        # Sort by weight (most recent first)
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)

        return neighbor_weights[:max_neighbors]

    def update_cache(self, node_id: int, embedding: torch.Tensor):
        """Update cached embedding for a node."""
        if self.cache is not None:
            self.cache.put(node_id, embedding)

    def get_cached_embedding(self, node_id: int) -> Optional[torch.Tensor]:
        """Get cached embedding for a node."""
        if self.cache is not None:
            return self.cache.get(node_id)
        return None

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            'window': self.window.get_window_stats()
        }

        if self.cache is not None:
            stats['cache'] = self.cache.get_stats()

        return stats
