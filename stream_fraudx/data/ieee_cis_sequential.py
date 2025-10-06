"""
Sequential Dataset Loader for IEEE-CIS Fraud Detection

Groups transactions into sequences by card (user) for temporal modeling.
Enables testing of temporal attention and multi-scale extraction.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IEEECISSequentialDataset(Dataset):
    """
    Sequential version of IEEE-CIS dataset.

    Groups transactions by card (user) into sequences for temporal modeling.
    Each sample is a sequence of transactions from the same card.
    """

    def __init__(self,
                 data_dir: str = 'data/ieee-cis',
                 split: str = 'train',
                 seq_len: int = 10,
                 min_seq_len: int = 3,
                 stride: int = 1):
        """
        Args:
            data_dir: Directory containing train_transaction.csv
            split: 'train' or 'test'
            seq_len: Target sequence length
            min_seq_len: Minimum transactions per sequence
            stride: Stride for creating overlapping sequences
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride

        # Load data
        print(f"Loading IEEE-CIS {split} data...")
        self.df = self._load_data()

        # Group into sequences
        print(f"Grouping transactions into sequences (seq_len={seq_len})...")
        self.sequences = self._create_sequences()

        print(f"Created {len(self.sequences)} sequences from {len(self.df)} transactions")
        if 'isFraud' in self.df.columns:
            fraud_count = sum(1 for seq in self.sequences if seq['fraud_label'])
            print(f"Fraud sequences: {fraud_count}/{len(self.sequences)} ({fraud_count/len(self.sequences):.2%})")

    def _load_data(self) -> pd.DataFrame:
        """Load transaction data."""
        txn_file = self.data_dir / f'{self.split}_transaction.csv'
        if not txn_file.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {txn_file}\n"
                f"Download from: kaggle competitions download -c ieee-fraud-detection"
            )

        df = pd.read_csv(txn_file)

        # Sort by time
        if 'TransactionDT' in df.columns:
            df = df.sort_values('TransactionDT').reset_index(drop=True)

        return df

    def _create_sequences(self) -> List[Dict]:
        """Group transactions into sequences by card."""
        sequences = []

        # Group by card1 (primary card identifier)
        if 'card1' not in self.df.columns:
            print("Warning: card1 not found, using sliding window instead")
            return self._create_sliding_window_sequences()

        grouped = self.df.groupby('card1')

        for card_id, group in grouped:
            if len(group) < self.min_seq_len:
                continue

            # Sort by time within card
            if 'TransactionDT' in group.columns:
                group = group.sort_values('TransactionDT')

            # Create overlapping sequences with stride
            for start_idx in range(0, len(group) - self.min_seq_len + 1, self.stride):
                end_idx = min(start_idx + self.seq_len, len(group))
                seq_df = group.iloc[start_idx:end_idx]

                if len(seq_df) >= self.min_seq_len:
                    # Determine sequence label (fraud if ANY transaction is fraud)
                    fraud_label = False
                    if 'isFraud' in seq_df.columns:
                        fraud_label = seq_df['isFraud'].max() > 0

                    sequences.append({
                        'indices': seq_df.index.tolist(),
                        'card_id': card_id,
                        'fraud_label': fraud_label,
                        'length': len(seq_df)
                    })

        return sequences

    def _create_sliding_window_sequences(self) -> List[Dict]:
        """Create sequences using sliding window (fallback)."""
        sequences = []

        for start_idx in range(0, len(self.df) - self.min_seq_len + 1, self.stride):
            end_idx = min(start_idx + self.seq_len, len(self.df))
            seq_df = self.df.iloc[start_idx:end_idx]

            if len(seq_df) >= self.min_seq_len:
                fraud_label = False
                if 'isFraud' in seq_df.columns:
                    fraud_label = seq_df['isFraud'].max() > 0

                sequences.append({
                    'indices': seq_df.index.tolist(),
                    'card_id': start_idx,
                    'fraud_label': fraud_label,
                    'length': len(seq_df)
                })

        return sequences

    def _preprocess_features(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Extract continuous and categorical features from transaction."""
        # Continuous features
        continuous_cols = ['TransactionAmt', 'dist1', 'dist2',
                          'card1', 'card2', 'card3', 'card5',
                          'addr1', 'addr2', 'C1', 'C2']

        continuous = []
        for col in continuous_cols:
            if col in row.index:
                val = row[col]
                continuous.append(float(val) if pd.notna(val) else 0.0)

        # Pad or truncate to 10 features
        while len(continuous) < 10:
            continuous.append(0.0)
        continuous = np.array(continuous[:10], dtype=np.float32)

        # Categorical features
        categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        categorical = []

        for col in categorical_cols:
            if col in row.index and pd.notna(row[col]):
                val = hash(str(row[col])) % 100
            else:
                val = 0
            categorical.append(val)

        categorical = np.array(categorical, dtype=np.int64)

        return continuous, categorical

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of transactions.

        Returns:
            Dict with:
                - sequence_features: [seq_len, 15] (continuous + categorical)
                - timestamps: [seq_len]
                - mask: [seq_len] (1 for valid, 0 for padding)
                - label: scalar (fraud if any transaction is fraud)
        """
        seq_info = self.sequences[idx]
        indices = seq_info['indices']

        # Extract features for each transaction in sequence
        sequence_features = []
        timestamps = []

        for trans_idx in indices:
            row = self.df.iloc[trans_idx]

            # Features
            continuous, categorical = self._preprocess_features(row)
            features = np.concatenate([continuous, categorical.astype(float)])
            sequence_features.append(features)

            # Timestamp
            timestamp = row['TransactionDT'] if 'TransactionDT' in row.index else 0.0
            timestamps.append(timestamp)

        # Convert to arrays
        sequence_features = np.array(sequence_features, dtype=np.float32)  # [actual_len, 15]
        timestamps = np.array(timestamps, dtype=np.float32)  # [actual_len]

        # Padding to seq_len
        actual_len = len(sequence_features)
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        if actual_len < self.seq_len:
            # Pad with zeros
            pad_len = self.seq_len - actual_len
            sequence_features = np.vstack([
                sequence_features,
                np.zeros((pad_len, 15), dtype=np.float32)
            ])
            timestamps = np.concatenate([
                timestamps,
                np.zeros(pad_len, dtype=np.float32)
            ])
        else:
            # Truncate if longer
            sequence_features = sequence_features[:self.seq_len]
            timestamps = timestamps[:self.seq_len]

        # Label
        label = float(seq_info['fraud_label'])

        return {
            'sequence_features': torch.from_numpy(sequence_features),  # [seq_len, 15]
            'timestamps': torch.from_numpy(timestamps),  # [seq_len]
            'mask': torch.from_numpy(mask),  # [seq_len]
            'label': torch.tensor(label, dtype=torch.float32)
        }


if __name__ == '__main__':
    # Test sequential dataset
    print("Testing IEEECISSequentialDataset...")

    try:
        dataset = IEEECISSequentialDataset(
            data_dir='data/ieee-cis',
            split='train',
            seq_len=10,
            min_seq_len=3,
            stride=5
        )

        print(f"\nDataset size: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Sequence features shape: {sample['sequence_features'].shape}")
        print(f"  Timestamps shape: {sample['timestamps'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Label: {sample['label'].item()}")
        print(f"  Mask sum (actual length): {sample['mask'].sum().item()}")

        # Test batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch = next(iter(loader))

        print(f"\nBatch shapes:")
        print(f"  Sequence features: {batch['sequence_features'].shape}")  # [32, 10, 15]
        print(f"  Timestamps: {batch['timestamps'].shape}")  # [32, 10]
        print(f"  Mask: {batch['mask'].shape}")  # [32, 10]
        print(f"  Label: {batch['label'].shape}")  # [32]

        print("\n✓ Sequential dataset test passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Dataset file not found: {e}")
        print("This is expected if IEEE-CIS data is not downloaded yet.")
