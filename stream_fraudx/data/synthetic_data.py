"""
Synthetic data generator for testing STREAM-FraudX.
Generates realistic fraud patterns with temporal drift.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple


class SyntheticFraudDataset(Dataset):
    """
    Synthetic fraud detection dataset with realistic patterns.
    """
    def __init__(self,
                 num_samples: int = 10000,
                 num_nodes: int = 1000,
                 fraud_rate: float = 0.01,
                 num_continuous: int = 10,
                 num_categorical: int = 5,
                 seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.fraud_rate = fraud_rate

        # Generate data
        self.data = self._generate_data(num_samples, num_continuous, num_categorical)

    def _generate_data(self, num_samples, num_continuous, num_categorical):
        """Generate synthetic fraud transactions."""
        data = {}

        # Generate graph structure
        src_nodes = torch.randint(0, self.num_nodes, (num_samples,))
        dst_nodes = torch.randint(0, self.num_nodes, (num_samples,))
        data['src_nodes'] = src_nodes
        data['dst_nodes'] = dst_nodes

        # Timestamps (increasing with noise)
        base_times = torch.linspace(0, 86400, num_samples)  # One day
        data['timestamps'] = base_times + torch.randn(num_samples) * 100

        # Edge attributes (for graph) - continuous features
        data['edge_attrs'] = torch.randn(num_samples, 64)

        # Discrete edge attributes for MEM pretraining
        amount_bins = torch.clamp((torch.abs(torch.randn(num_samples)) * 10).long(), 0, 49)  # 50 bins
        mcc_bins = torch.randint(0, 20, (num_samples,))  # 20 MCC categories
        device_types = torch.randint(0, 10, (num_samples,))  # 10 device types

        data['edge_attr_discrete'] = {
            'amount_bin': amount_bins,
            'mcc_bin': mcc_bins,
            'device_type': device_types
        }

        # Continuous features (amounts, etc.)
        continuous = torch.randn(num_samples, num_continuous)
        continuous[:, 0] = torch.abs(continuous[:, 0]) * 100  # Transaction amount
        data['continuous'] = continuous

        # Categorical features
        categorical = torch.randint(0, 100, (num_samples, num_categorical))
        data['categorical'] = categorical

        # Generate labels with fraud patterns
        labels = torch.zeros(num_samples)

        # Fraud pattern 1: High amount transactions
        high_amount_mask = continuous[:, 0] > 200
        labels[high_amount_mask] = (torch.rand(high_amount_mask.sum()) < 0.5).float()

        # Fraud pattern 2: Frequent transactions (velocity)
        _, counts = torch.unique(src_nodes, return_counts=True)
        frequent_nodes = torch.where(counts > 20)[0]
        for node in frequent_nodes:
            node_mask = src_nodes == node
            labels[node_mask] = (torch.rand(node_mask.sum()) < 0.3).float()

        # Random fraud to reach target rate
        current_rate = labels.mean()
        if current_rate < self.fraud_rate:
            num_add = int((self.fraud_rate - current_rate) * num_samples)
            add_idx = torch.randperm(num_samples)[:num_add]
            labels[add_idx] = 1.0

        data['labels'] = labels

        # Time slices for IRM
        time_slices = (data['timestamps'] / 3600).long()  # Hour buckets
        data['time_slices'] = time_slices

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {}
        for k, v in self.data.items():
            if k == 'edge_attr_discrete':
                item[k] = {attr_name: attr_val[idx] for attr_name, attr_val in v.items()}
            else:
                item[k] = v[idx]
        return item


def collate_fn(batch):
    """Collate function for DataLoader."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'edge_attr_discrete':
            # Handle nested dictionary
            collated[key] = {}
            for attr_name in batch[0][key].keys():
                collated[key][attr_name] = torch.stack([item[key][attr_name] for item in batch])
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated
