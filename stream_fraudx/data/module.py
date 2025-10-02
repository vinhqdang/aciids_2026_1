"""
Data Module for STREAM-FraudX
Handles micro-batch streaming, label-latency simulation, and unified data loading.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, Iterator, Callable
from collections import deque
import numpy as np
from pathlib import Path

from .synthetic_data import SyntheticFraudDataset, collate_fn as synthetic_collate
from .ieee_cis_loader import IEEECISDataset
from .paysim_loader import PaySimDataset
from .elliptic_loader import EllipticDataset


class LabelLatencyQueue:
    """
    Simulates label arrival delay in streaming scenarios.
    Labels arrive after a configurable delay (in number of micro-batches).
    """
    def __init__(self, delay_batches: int = 10):
        """
        Args:
            delay_batches: Number of micro-batches to delay label arrival
        """
        self.delay_batches = delay_batches
        self.queue = deque(maxlen=delay_batches * 2)

    def add_batch(self, samples: Dict[str, torch.Tensor], labels: torch.Tensor):
        """Add a batch to the label queue."""
        self.queue.append({
            'samples': samples,
            'labels': labels,
            'age': 0
        })

    def get_ready_labels(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get labels that have reached the required delay.

        Returns:
            Dictionary with samples and their delayed labels, or None
        """
        # Age all items in queue
        for item in self.queue:
            item['age'] += 1

        # Find items ready for release
        ready_items = [item for item in self.queue if item['age'] >= self.delay_batches]

        if not ready_items:
            return None

        # Remove ready items from queue
        for item in ready_items:
            self.queue.remove(item)

        # Combine ready items
        return self._combine_batches(ready_items)

    def _combine_batches(self, items: list) -> Dict[str, torch.Tensor]:
        """Combine multiple batches into one."""
        if not items:
            return None

        combined = {
            'samples': {},
            'labels': []
        }

        # Collect all labels
        for item in items:
            combined['labels'].append(item['labels'])

        combined['labels'] = torch.cat(combined['labels'], dim=0)

        # Collect sample data
        sample_keys = items[0]['samples'].keys()
        for key in sample_keys:
            values = [item['samples'][key] for item in items]
            if torch.is_tensor(values[0]):
                combined['samples'][key] = torch.cat(values, dim=0)
            elif isinstance(values[0], dict):
                # Handle nested dictionaries
                combined['samples'][key] = {}
                for subkey in values[0].keys():
                    subvalues = [v[subkey] for v in values]
                    combined['samples'][key][subkey] = torch.cat(subvalues, dim=0)

        return combined


class MicroBatchStream:
    """
    Converts a dataset into micro-batch stream with configurable window size.
    Supports time-slicing and drift window configuration.
    """
    def __init__(self,
                 dataset: Dataset,
                 microbatch_size: int = 100,
                 window_seconds: int = 30,
                 collate_fn: Optional[Callable] = None):
        """
        Args:
            dataset: Source dataset
            microbatch_size: Number of samples per micro-batch
            window_seconds: Time window for each micro-batch (simulation)
            collate_fn: Custom collate function
        """
        self.dataset = dataset
        self.microbatch_size = microbatch_size
        self.window_seconds = window_seconds
        self.collate_fn = collate_fn or synthetic_collate
        self.current_idx = 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over micro-batches."""
        self.current_idx = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next micro-batch."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # Get next batch of samples
        end_idx = min(self.current_idx + self.microbatch_size, len(self.dataset))
        samples = [self.dataset[i] for i in range(self.current_idx, end_idx)]
        self.current_idx = end_idx

        # Collate into batch
        batch = self.collate_fn(samples)

        # Add time window metadata
        batch['timestamp'] = self.current_idx * self.window_seconds
        batch['window_seconds'] = self.window_seconds

        return batch

    def reset(self):
        """Reset stream to beginning."""
        self.current_idx = 0


class StreamDataModule:
    """
    Unified data module for STREAM-FraudX.
    Supports multiple dataset backends, micro-batch streaming, and label-latency simulation.
    """
    def __init__(self,
                 dataset_name: str = 'synthetic',
                 data_dir: Optional[str] = None,
                 microbatch_size: int = 100,
                 label_delay_batches: int = 10,
                 synthetic_params: Optional[Dict] = None):
        """
        Args:
            dataset_name: 'synthetic', 'ieee-cis', 'paysim', or 'elliptic'
            data_dir: Directory containing real datasets
            microbatch_size: Size of micro-batches for streaming
            label_delay_batches: Delay for label arrival (in micro-batches)
            synthetic_params: Parameters for synthetic data generation
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.microbatch_size = microbatch_size
        self.label_delay_batches = label_delay_batches
        self.synthetic_params = synthetic_params or {}

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Streaming components
        self.label_queue = LabelLatencyQueue(delay_batches=label_delay_batches)

    def setup(self):
        """Load and prepare datasets."""
        if self.dataset_name == 'synthetic':
            self._setup_synthetic()
        elif self.dataset_name == 'ieee-cis':
            self._setup_ieee_cis()
        elif self.dataset_name == 'paysim':
            self._setup_paysim()
        elif self.dataset_name == 'elliptic':
            self._setup_elliptic()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _setup_synthetic(self):
        """Setup synthetic dataset."""
        from torch.utils.data import random_split

        params = {
            'num_samples': 50000,
            'num_nodes': 2000,
            'fraud_rate': 0.01,
            'num_continuous': 10,
            'num_categorical': 5,
            **self.synthetic_params
        }

        full_dataset = SyntheticFraudDataset(**params)

        # Split into train/val/test
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        self.collate_fn = synthetic_collate

    def _setup_ieee_cis(self):
        """Setup IEEE-CIS dataset."""
        data_path = self.data_dir / 'ieee-cis'
        if not data_path.exists():
            raise FileNotFoundError(f"IEEE-CIS data not found at {data_path}")

        self.train_dataset = IEEECISDataset(data_path, split='train')
        self.val_dataset = IEEECISDataset(data_path, split='val')
        self.test_dataset = IEEECISDataset(data_path, split='test')

        self.collate_fn = self.train_dataset.collate_fn

    def _setup_paysim(self):
        """Setup PaySim dataset."""
        data_path = self.data_dir / 'paysim'
        if not data_path.exists():
            raise FileNotFoundError(f"PaySim data not found at {data_path}")

        self.train_dataset = PaySimDataset(data_path, split='train')
        self.val_dataset = PaySimDataset(data_path, split='val')
        self.test_dataset = PaySimDataset(data_path, split='test')

        self.collate_fn = self.train_dataset.collate_fn

    def _setup_elliptic(self):
        """Setup Elliptic dataset."""
        data_path = self.data_dir / 'elliptic'
        if not data_path.exists():
            raise FileNotFoundError(f"Elliptic data not found at {data_path}")

        self.train_dataset = EllipticDataset(data_path, split='train')
        self.val_dataset = EllipticDataset(data_path, split='val')
        self.test_dataset = EllipticDataset(data_path, split='test')

        self.collate_fn = self.train_dataset.collate_fn

    def train_dataloader(self, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=0
        )

    def val_dataloader(self, batch_size: int = 256) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )

    def test_dataloader(self, batch_size: int = 256) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )

    def stream_dataloader(self, split: str = 'test', window_seconds: int = 30) -> MicroBatchStream:
        """
        Get streaming dataloader with micro-batches.

        Args:
            split: 'train', 'val', or 'test'
            window_seconds: Time window for micro-batches

        Returns:
            MicroBatchStream iterator
        """
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        return MicroBatchStream(
            dataset=dataset,
            microbatch_size=self.microbatch_size,
            window_seconds=window_seconds,
            collate_fn=self.collate_fn
        )

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for categorical features."""
        if hasattr(self.train_dataset, 'dataset'):
            # Handle Subset wrapper
            base_dataset = self.train_dataset.dataset
        else:
            base_dataset = self.train_dataset

        if hasattr(base_dataset, 'get_vocab_sizes'):
            return base_dataset.get_vocab_sizes()
        elif hasattr(base_dataset, 'categorical_vocab_sizes'):
            return base_dataset.categorical_vocab_sizes
        else:
            # Default for synthetic
            return [100] * self.synthetic_params.get('num_categorical', 5)

    def get_continuous_dims(self) -> int:
        """Get number of continuous features."""
        if hasattr(self.train_dataset, 'dataset'):
            base_dataset = self.train_dataset.dataset
        else:
            base_dataset = self.train_dataset

        if hasattr(base_dataset, 'get_continuous_dims'):
            return base_dataset.get_continuous_dims()
        elif hasattr(base_dataset, 'num_continuous'):
            return base_dataset.num_continuous
        else:
            return self.synthetic_params.get('num_continuous', 10)
