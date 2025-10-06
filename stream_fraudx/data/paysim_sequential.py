"""
Sequential Dataset Loader for PaySim Fraud Detection

Groups transactions into sequences by user (nameOrig) for temporal modeling.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class PaySimSequentialDataset(Dataset):
    """
    Sequential version of PaySim dataset.

    Groups transactions by user (nameOrig) into sequences for temporal modeling.
    """

    def __init__(self,
                 data_dir: str = 'data/paysim',
                 filename: str = 'PS_20174392719_1491204439457_log.csv',
                 seq_len: int = 10,
                 min_seq_len: int = 3,
                 stride: int = 1,
                 fraction: float = 1.0):
        """
        Args:
            data_dir: Directory containing PaySim CSV
            filename: CSV filename
            seq_len: Target sequence length
            min_seq_len: Minimum transactions per sequence
            stride: Stride for creating overlapping sequences
            fraction: Fraction of data to use
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride

        # Load data
        print(f"Loading PaySim dataset...")
        csv_path = self.data_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"PaySim file not found: {csv_path}\n"
                f"Download from: kaggle datasets download -d ealaxi/paysim1"
            )

        self.df = pd.read_csv(csv_path)

        # Sample if needed
        if fraction < 1.0:
            self.df = self.df.sample(frac=fraction, random_state=42).reset_index(drop=True)

        # Sort by time
        self.df = self.df.sort_values('step').reset_index(drop=True)

        # Group into sequences
        print(f"Grouping transactions into sequences (seq_len={seq_len})...")
        self.sequences = self._create_sequences()

        print(f"Created {len(self.sequences)} sequences from {len(self.df)} transactions")
        fraud_count = sum(1 for seq in self.sequences if seq['fraud_label'])
        print(f"Fraud sequences: {fraud_count}/{len(self.sequences)} ({fraud_count/len(self.sequences):.2%})")

    def _create_sequences(self) -> List[Dict]:
        """Group transactions into sequences by user."""
        sequences = []

        # Try grouping by nameOrig first
        grouped = self.df.groupby('nameOrig')

        users_with_sequences = 0
        for user_id, group in grouped:
            if len(group) < self.min_seq_len:
                continue

            users_with_sequences += 1

            # Sort by time within user
            group = group.sort_values('step')

            # Create overlapping sequences
            for start_idx in range(0, len(group) - self.min_seq_len + 1, self.stride):
                end_idx = min(start_idx + self.seq_len, len(group))
                seq_df = group.iloc[start_idx:end_idx]

                if len(seq_df) >= self.min_seq_len:
                    # Label is fraud if ANY transaction is fraud
                    fraud_label = seq_df['isFraud'].max() > 0

                    sequences.append({
                        'indices': seq_df.index.tolist(),
                        'user_id': user_id,
                        'fraud_label': fraud_label,
                        'length': len(seq_df)
                    })

        # If no sequences found, use sliding window instead
        if len(sequences) == 0:
            print(f"Warning: Only {users_with_sequences} users have >= {self.min_seq_len} transactions")
            print(f"Using sliding window approach instead...")
            sequences = self._create_sliding_window_sequences()

        return sequences

    def _create_sliding_window_sequences(self) -> List[Dict]:
        """Create sequences using sliding window (fallback)."""
        sequences = []

        for start_idx in range(0, len(self.df) - self.min_seq_len + 1, self.stride):
            end_idx = min(start_idx + self.seq_len, len(self.df))
            seq_df = self.df.iloc[start_idx:end_idx]

            if len(seq_df) >= self.min_seq_len:
                fraud_label = seq_df['isFraud'].max() > 0

                sequences.append({
                    'indices': seq_df.index.tolist(),
                    'user_id': start_idx,
                    'fraud_label': fraud_label,
                    'length': len(seq_df)
                })

        return sequences

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

    def _extract_features(self, row: pd.Series) -> np.ndarray:
        """Extract features from a single transaction."""
        # Transaction amounts and balances
        amount = row['amount']
        old_balance_orig = row['oldbalanceOrg']
        new_balance_orig = row['newbalanceOrig']
        old_balance_dest = row['oldbalanceDest']
        new_balance_dest = row['newbalanceDest']

        # Derived features
        balance_diff_orig = new_balance_orig - old_balance_orig
        balance_diff_dest = new_balance_dest - old_balance_dest
        amount_ratio = amount / (old_balance_orig + 1)
        error_orig = abs(balance_diff_orig + amount) / (amount + 1)
        error_dest = abs(balance_diff_dest - amount) / (amount + 1)

        # Transaction type
        tx_type = self._encode_transaction_type(row['type'])

        # Combine into feature vector [15 features]
        features = np.array([
            amount,
            old_balance_orig,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest,
            balance_diff_orig,
            balance_diff_dest,
            amount_ratio,
            error_orig,
            error_dest,
            float(tx_type),
            float(row.get('isFlaggedFraud', 0)),
            0.0,  # Placeholder
            0.0,  # Placeholder
            0.0   # Placeholder
        ], dtype=np.float32)

        return features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of transactions.

        Returns:
            Dict with:
                - sequence_features: [seq_len, 15]
                - timestamps: [seq_len]
                - mask: [seq_len]
                - label: scalar
        """
        seq_info = self.sequences[idx]
        indices = seq_info['indices']

        # Extract features for each transaction
        sequence_features = []
        timestamps = []

        for trans_idx in indices:
            row = self.df.iloc[trans_idx]

            # Features
            features = self._extract_features(row)
            sequence_features.append(features)

            # Timestamp (step is in hours)
            timestamp = float(row['step']) * 3600  # Convert to seconds
            timestamps.append(timestamp)

        # Convert to arrays
        sequence_features = np.array(sequence_features, dtype=np.float32)
        timestamps = np.array(timestamps, dtype=np.float32)

        # Padding
        actual_len = len(sequence_features)
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        if actual_len < self.seq_len:
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
            sequence_features = sequence_features[:self.seq_len]
            timestamps = timestamps[:self.seq_len]

        # Label
        label = float(seq_info['fraud_label'])

        return {
            'sequence_features': torch.from_numpy(sequence_features),
            'timestamps': torch.from_numpy(timestamps),
            'mask': torch.from_numpy(mask),
            'label': torch.tensor(label, dtype=torch.float32)
        }


if __name__ == '__main__':
    # Test sequential dataset
    print("Testing PaySimSequentialDataset...")

    try:
        dataset = PaySimSequentialDataset(
            data_dir='data/paysim',
            seq_len=10,
            min_seq_len=3,
            stride=5,
            fraction=0.1  # Use 10% for quick test
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
        print(f"  Sequence features: {batch['sequence_features'].shape}")
        print(f"  Timestamps: {batch['timestamps'].shape}")
        print(f"  Mask: {batch['mask'].shape}")
        print(f"  Label: {batch['label'].shape}")

        print("\n✓ Sequential dataset test passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Dataset file not found: {e}")
        print("This is expected if PaySim data is not downloaded yet.")
