"""
PaySim Mobile Money Simulator Dataset Loader
Source: https://www.kaggle.com/datasets/ealaxi/paysim1

Download: kaggle datasets download -d ealaxi/paysim1
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict


class PaySimDataset(Dataset):
    """
    PaySim mobile money fraud detection dataset.

    Features:
    - step: Time step (hour)
    - type: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
    - amount: Transaction amount
    - nameOrig: Customer ID initiating transaction
    - oldbalanceOrg: Initial balance before transaction
    - newbalanceOrig: Balance after transaction
    - nameDest: Recipient ID
    - oldbalanceDest: Initial balance recipient before transaction
    - newbalanceDest: Balance recipient after transaction
    - isFraud: Binary label

    Ideal for temporal drift experiments (transactions over time).
    """

    def __init__(self,
                 data_dir: str = 'data/paysim',
                 filename: str = 'PS_20174392719_1491204439457_log.csv',
                 fraction: float = 1.0,
                 create_graph: bool = True):
        """
        Args:
            data_dir: Directory containing PaySim CSV
            filename: CSV filename
            fraction: Fraction of data to use (for faster experimentation)
            create_graph: Build user-merchant graph
        """
        self.data_dir = Path(data_dir)
        self.create_graph = create_graph

        print(f"Loading PaySim dataset...")
        csv_path = self.data_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"PaySim file not found: {csv_path}\n"
                f"Download from: kaggle datasets download -d ealaxi/paysim1"
            )

        # Load data
        self.df = pd.read_csv(csv_path)

        # Sample if fraction < 1.0
        if fraction < 1.0:
            self.df = self.df.sample(frac=fraction, random_state=42).reset_index(drop=True)

        # Sort by time
        self.df = self.df.sort_values('step').reset_index(drop=True)

        # Create node mappings
        self._create_node_mapping()

        print(f"Loaded {len(self.df)} transactions")
        fraud_rate = self.df['isFraud'].mean()
        print(f"Fraud rate: {fraud_rate:.2%}")

    def _create_node_mapping(self):
        """Create mapping from customer IDs to node indices."""
        # Get unique originator and destination IDs
        all_ids = pd.concat([self.df['nameOrig'], self.df['nameDest']]).unique()

        self.id_to_node = {id_: idx for idx, id_ in enumerate(all_ids)}
        self.num_nodes = len(all_ids)

        print(f"Graph: {self.num_nodes} nodes (users/merchants)")

    def _encode_transaction_type(self, tx_type: str) -> int:
        """Encode transaction type to integer."""
        type_mapping = {
            'CASH_IN': 0,
            'CASH_OUT': 1,
            'DEBIT': 2,
            'PAYMENT': 3,
            'TRANSFER': 4
        }
        return type_mapping.get(tx_type, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Graph nodes
        src_node = self.id_to_node.get(row['nameOrig'], 0)
        dst_node = self.id_to_node.get(row['nameDest'], 0)

        # Transaction features
        amount = row['amount']
        old_balance_orig = row['oldbalanceOrg']
        new_balance_orig = row['newbalanceOrig']
        old_balance_dest = row['oldbalanceDest']
        new_balance_dest = row['newbalanceDest']

        # Derived features
        balance_diff_orig = new_balance_orig - old_balance_orig
        balance_diff_dest = new_balance_dest - old_balance_dest

        # Continuous features
        continuous = torch.tensor([
            amount,
            old_balance_orig,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest,
            balance_diff_orig,
            balance_diff_dest,
            amount / (old_balance_orig + 1),  # Amount ratio
            abs(balance_diff_orig + amount) / (amount + 1),  # Error in balance
            abs(balance_diff_dest - amount) / (amount + 1)   # Error in dest balance
        ], dtype=torch.float32)

        # Categorical features
        tx_type = self._encode_transaction_type(row['type'])
        categorical = torch.tensor([
            tx_type,
            src_node % 100,  # Hash for embedding
            dst_node % 100,
            int(row['isFlaggedFraud']) if 'isFlaggedFraud' in row.index else 0,
            0  # Placeholder
        ], dtype=torch.long)

        # Edge attributes (transaction-specific)
        edge_attrs = torch.cat([
            continuous[:5],  # First 5 continuous features
            torch.randn(59)  # Padding to 64
        ])

        # Discrete edge attributes
        amount_bin = min(int(np.log10(amount + 1) * 10), 49)  # Log-scale binning
        edge_attr_discrete = {
            'amount_bin': torch.tensor(amount_bin, dtype=torch.long),
            'mcc_bin': torch.tensor(tx_type, dtype=torch.long),  # Use tx type as MCC proxy
            'device_type': torch.tensor(hash(row['nameOrig']) % 10, dtype=torch.long)
        }

        # Timestamp (step = hours)
        timestamp = torch.tensor(float(row['step']) * 3600, dtype=torch.float32)  # Convert to seconds

        # Label
        label = torch.tensor(row['isFraud'], dtype=torch.float32)

        # Time slice (day buckets)
        time_slice = torch.tensor(int(row['step'] / 24), dtype=torch.long)

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


def download_paysim(data_dir: str = 'data/paysim'):
    """Download PaySim dataset using Kaggle API."""
    import os
    import zipfile

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading PaySim dataset...")
    os.system(f'kaggle datasets download -d ealaxi/paysim1 -p {data_dir}')

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
        dataset = PaySimDataset(data_dir='data/paysim', fraction=0.1)  # 10% for testing
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
        print("  python -c 'from stream_fraudx.data.paysim_loader import download_paysim; download_paysim()'")
