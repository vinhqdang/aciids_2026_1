"""
IEEE-CIS Fraud Detection Dataset Loader
Kaggle competition: https://www.kaggle.com/c/ieee-fraud-detection

Download instructions:
1. Install Kaggle API: pip install kaggle
2. Setup API key: https://www.kaggle.com/docs/api
3. Download: kaggle competitions download -c ieee-fraud-detection
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class IEEECISDataset(Dataset):
    """
    IEEE-CIS Fraud Detection Dataset.

    Features:
    - Transaction data: TransactionDT, TransactionAmt, ProductCD, card1-6, addr1-2, dist1-2
    - Identity data: id_01-38, DeviceType, DeviceInfo
    - Rich categorical features: card4, card6, P_emaildomain, R_emaildomain, M1-M9

    Task: Binary classification (isFraud)
    """

    def __init__(self,
                 data_dir: str = 'data/ieee-cis',
                 split: str = 'train',
                 create_graph: bool = True,
                 max_nodes: int = 50000):
        """
        Args:
            data_dir: Directory containing train_transaction.csv, train_identity.csv
            split: 'train' or 'test'
            create_graph: Whether to build co-occurrence graph
            max_nodes: Maximum number of nodes in graph
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.create_graph = create_graph
        self.max_nodes = max_nodes

        # Load data
        print(f"Loading IEEE-CIS {split} data...")
        self.df = self._load_data()

        # Build graph if requested
        if create_graph:
            print("Building co-occurrence graph...")
            self.graph_data = self._build_graph()

        print(f"Loaded {len(self.df)} transactions")
        if 'isFraud' in self.df.columns:
            fraud_rate = self.df['isFraud'].mean()
            print(f"Fraud rate: {fraud_rate:.2%}")

    def _load_data(self) -> pd.DataFrame:
        """Load and merge transaction + identity data."""
        # Load transaction data
        txn_file = self.data_dir / f'{self.split}_transaction.csv'
        if not txn_file.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {txn_file}\n"
                f"Download from: kaggle competitions download -c ieee-fraud-detection"
            )

        df_txn = pd.read_csv(txn_file)

        # Load identity data (if exists)
        identity_file = self.data_dir / f'{self.split}_identity.csv'
        if identity_file.exists():
            df_identity = pd.read_csv(identity_file)
            df = df_txn.merge(df_identity, on='TransactionID', how='left')
        else:
            df = df_txn

        return df

    def _build_graph(self) -> Dict:
        """
        Build co-occurrence graph from transaction data.
        Nodes: card1, card2, addr1, P_emaildomain, etc.
        Edges: transactions connect related entities
        """
        # Entity columns for graph construction
        entity_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                      'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']

        # Available entity columns
        available_cols = [col for col in entity_cols if col in self.df.columns]

        # Create node mapping
        node_to_id = {}
        node_id = 0

        for col in available_cols:
            unique_vals = self.df[col].dropna().unique()
            for val in unique_vals[:self.max_nodes // len(available_cols)]:
                node_key = f"{col}_{val}"
                if node_key not in node_to_id:
                    node_to_id[node_key] = node_id
                    node_id += 1

        # Build edges (transaction connects entities)
        edges = []
        for idx, row in self.df.iterrows():
            # Get all entity nodes for this transaction
            tx_nodes = []
            for col in available_cols:
                if pd.notna(row[col]):
                    node_key = f"{col}_{row[col]}"
                    if node_key in node_to_id:
                        tx_nodes.append(node_to_id[node_key])

            # Connect all entity pairs in this transaction
            for i in range(len(tx_nodes)):
                for j in range(i+1, len(tx_nodes)):
                    edges.append((tx_nodes[i], tx_nodes[j], idx))

        return {
            'node_to_id': node_to_id,
            'edges': edges,
            'num_nodes': len(node_to_id)
        }

    def _preprocess_features(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract continuous and categorical features."""
        # Continuous features
        continuous_cols = ['TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4',
                          'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                          'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15']

        continuous = []
        for col in continuous_cols:
            if col in row.index:
                val = row[col]
                continuous.append(val if pd.notna(val) else 0.0)

        # Pad to 10 features
        while len(continuous) < 10:
            continuous.append(0.0)
        continuous = continuous[:10]

        # Categorical features (use label encoding)
        categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        categorical = []

        for col in categorical_cols:
            if col in row.index and pd.notna(row[col]):
                # Simple hash-based encoding
                val = hash(str(row[col])) % 100
            else:
                val = 0
            categorical.append(val)

        return torch.tensor(continuous, dtype=torch.float32), torch.tensor(categorical, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Extract features
        continuous, categorical = self._preprocess_features(row)

        # Timestamp (TransactionDT is seconds from reference point)
        timestamp = torch.tensor(row['TransactionDT'] if 'TransactionDT' in row.index else 0.0, dtype=torch.float32)

        # Graph nodes (if graph enabled)
        if self.create_graph and hasattr(self, 'graph_data'):
            # Find nodes associated with this transaction
            src_node = idx % self.graph_data['num_nodes']  # Simplified
            dst_node = (idx + 1) % self.graph_data['num_nodes']
        else:
            src_node = idx % 1000
            dst_node = (idx + 1) % 1000

        # Edge attributes
        edge_attrs = torch.randn(64)  # Placeholder, should be derived from features

        # Discrete edge attributes for pretraining
        amount_bin = min(int(row['TransactionAmt'] / 10), 49) if 'TransactionAmt' in row.index else 0
        edge_attr_discrete = {
            'amount_bin': torch.tensor(amount_bin, dtype=torch.long),
            'mcc_bin': torch.tensor(0, dtype=torch.long),  # Not available in IEEE-CIS
            'device_type': torch.tensor(hash(str(row.get('DeviceType', ''))) % 10, dtype=torch.long)
        }

        # Label
        label = torch.tensor(row['isFraud'], dtype=torch.float32) if 'isFraud' in row.index else torch.tensor(0.0)

        # Time slice for IRM
        time_slice = torch.tensor(int(timestamp / 86400), dtype=torch.long)  # Day buckets

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


def download_ieee_cis(data_dir: str = 'data/ieee-cis'):
    """
    Download IEEE-CIS dataset using Kaggle API.

    Prerequisites:
    1. pip install kaggle
    2. Setup ~/.kaggle/kaggle.json with API credentials
    """
    import os
    import zipfile

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading IEEE-CIS Fraud Detection dataset...")
    print("Make sure you have accepted the competition rules on Kaggle!")

    os.system(f'kaggle competitions download -c ieee-fraud-detection -p {data_dir}')

    # Unzip files
    for zip_file in data_path.glob('*.zip'):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        zip_file.unlink()  # Remove zip after extraction

    print(f"✓ Dataset downloaded to {data_dir}")


if __name__ == '__main__':
    # Test loader
    try:
        dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train')
        print(f"\nDataset size: {len(dataset)}")

        # Test sample
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
        print("  python -c 'from stream_fraudx.data.ieee_cis_loader import download_ieee_cis; download_ieee_cis()'")
