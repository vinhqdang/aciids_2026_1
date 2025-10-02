"""
Elliptic Bitcoin Transaction Dataset Loader
Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Download: kaggle datasets download -d ellipticco/elliptic-data-set

Graph structure with temporal labels (illicit vs licit transactions).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import networkx as nx


class EllipticDataset(Dataset):
    """
    Elliptic Bitcoin transaction graph dataset.

    Files:
    - elliptic_txs_features.csv: Node features (166 features per transaction)
    - elliptic_txs_classes.csv: Labels (1=illicit, 2=licit, unknown=unlabeled)
    - elliptic_txs_edgelist.csv: Graph edges (transaction flow)

    Features: 166 dimensional
    - 1 feature: Time step
    - 93 features: Local information (aggregated features)
    - 72 features: Aggregated features

    Ideal for testing temporal graph tower.
    """

    def __init__(self,
                 data_dir: str = 'data/elliptic',
                 use_labeled_only: bool = False,
                 time_window: Optional[int] = None):
        """
        Args:
            data_dir: Directory containing elliptic CSV files
            use_labeled_only: If True, only use labeled transactions
            time_window: If set, only use transactions in specific time window
        """
        self.data_dir = Path(data_dir)
        self.use_labeled_only = use_labeled_only

        print(f"Loading Elliptic Bitcoin dataset...")

        # Load node features
        features_file = self.data_dir / 'elliptic_txs_features.csv'
        if not features_file.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_file}\n"
                f"Download from: kaggle datasets download -d ellipticco/elliptic-data-set"
            )

        self.features_df = pd.read_csv(features_file, header=None)
        self.features_df.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(165)]

        # Load labels
        classes_file = self.data_dir / 'elliptic_txs_classes.csv'
        self.classes_df = pd.read_csv(classes_file)
        self.classes_df.columns = ['txId', 'class']

        # Load edges
        edges_file = self.data_dir / 'elliptic_txs_edgelist.csv'
        self.edges_df = pd.read_csv(edges_file)
        self.edges_df.columns = ['txId1', 'txId2']

        # Merge features and labels
        self.df = self.features_df.merge(self.classes_df, on='txId', how='left')

        # Filter by time window if specified
        if time_window is not None:
            self.df = self.df[self.df['time_step'] == time_window]

        # Filter labeled only if requested
        if use_labeled_only:
            self.df = self.df[self.df['class'].notna()].reset_index(drop=True)

        # Convert labels: 1 (illicit) -> 1, 2 (licit) -> 0, unknown -> -1
        self.df['label'] = self.df['class'].map({1: 1.0, 2: 0.0}).fillna(-1.0)

        # Create node mapping
        self.tx_to_node = {tx: idx for idx, tx in enumerate(self.df['txId'].unique())}

        # Build graph structure
        self._build_graph()

        print(f"Loaded {len(self.df)} transactions")
        labeled_count = (self.df['label'] >= 0).sum()
        print(f"Labeled: {labeled_count} ({labeled_count/len(self.df)*100:.1f}%)")
        if labeled_count > 0:
            illicit_rate = (self.df['label'] == 1).sum() / labeled_count
            print(f"Illicit rate: {illicit_rate:.2%}")

    def _build_graph(self):
        """Build graph structure from edge list."""
        # Filter edges to only include transactions in our dataset
        valid_txs = set(self.df['txId'].unique())

        self.edges = []
        for _, row in self.edges_df.iterrows():
            if row['txId1'] in valid_txs and row['txId2'] in valid_txs:
                src = self.tx_to_node.get(row['txId1'], 0)
                dst = self.tx_to_node.get(row['txId2'], 0)
                self.edges.append((src, dst))

        print(f"Graph: {len(self.tx_to_node)} nodes, {len(self.edges)} edges")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Node features (165 features after txId and time_step)
        feature_cols = [f'feat_{i}' for i in range(165)]
        node_features = torch.tensor(row[feature_cols].values, dtype=torch.float32)

        # Split into continuous (first 10) and rest for edge attrs
        continuous = node_features[:10]
        edge_attrs = torch.cat([node_features[10:74], torch.zeros(64 - 64)])  # Pad to 64

        # Categorical features (derived)
        time_step = int(row['time_step'])
        categorical = torch.tensor([
            time_step % 100,
            idx % 100,
            0, 0, 0
        ], dtype=torch.long)

        # Graph structure
        tx_id = row['txId']
        src_node = self.tx_to_node.get(tx_id, idx)

        # Find a connected node (if exists)
        connected_edges = [(s, d) for s, d in self.edges if s == src_node]
        if connected_edges:
            _, dst_node = connected_edges[0]
        else:
            dst_node = (src_node + 1) % len(self.tx_to_node)

        # Discrete edge attributes
        amount_estimate = int(abs(node_features[0].item()) * 10)  # Derived from first feature
        edge_attr_discrete = {
            'amount_bin': torch.tensor(min(amount_estimate, 49), dtype=torch.long),
            'mcc_bin': torch.tensor(time_step % 20, dtype=torch.long),
            'device_type': torch.tensor(idx % 10, dtype=torch.long)
        }

        # Timestamp (time_step as proxy, convert to seconds)
        timestamp = torch.tensor(float(time_step) * 3600 * 24, dtype=torch.float32)  # Days to seconds

        # Label
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Time slice
        time_slice = torch.tensor(time_step, dtype=torch.long)

        return {
            'src_nodes': torch.tensor(src_node, dtype=torch.long),
            'dst_nodes': torch.tensor(dst_node, dtype=torch.long),
            'edge_attrs': edge_attrs,
            'edge_attr_discrete': edge_attr_discrete,
            'continuous': continuous,
            'categorical': categorical,
            'timestamps': timestamp,
            'labels': label,
            'time_slices': time_slice
        }


def download_elliptic(data_dir: str = 'data/elliptic'):
    """Download Elliptic dataset using Kaggle API."""
    import os
    import zipfile

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Elliptic Bitcoin dataset...")
    os.system(f'kaggle datasets download -d ellipticco/elliptic-data-set -p {data_dir}')

    # Unzip
    for zip_file in data_path.glob('*.zip'):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        zip_file.unlink()

    print(f"✓ Dataset downloaded to {data_dir}")


if __name__ == '__main__':
    # Test loader
    try:
        dataset = EllipticDataset(data_dir='data/elliptic', use_labeled_only=True)
        print(f"\nDataset size: {len(dataset)}")

        sample = dataset[0]
        print("\nSample data:")
        for key, val in sample.items():
            if isinstance(val, dict):
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTo download the dataset, run:")
        print("  python -c 'from stream_fraudx.data.elliptic_loader import download_elliptic; download_elliptic()'")
