"""
Feature encoder registries with persistence.
Manages categorical and continuous feature encoding with state management.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import json
from collections import defaultdict


class CategoricalEncoder:
    """
    Categorical feature encoder with vocabulary management.

    Features:
    - Label encoding with unknown token handling
    - Vocabulary persistence
    - Frequency tracking for rare category handling
    """

    def __init__(self, name: str, min_frequency: int = 1, unknown_token: int = 0):
        self.name = name
        self.min_frequency = min_frequency
        self.unknown_token = unknown_token

        # Vocabulary
        self.value_to_idx: Dict[Any, int] = {}
        self.idx_to_value: Dict[int, Any] = {}
        self.value_counts: Dict[Any, int] = defaultdict(int)

        # State
        self.is_fitted = False
        self.vocab_size = 0

    def fit(self, values: np.ndarray) -> 'CategoricalEncoder':
        """
        Fit encoder to categorical values.

        Args:
            values: Array of categorical values
        """
        # Count frequencies
        unique, counts = np.unique(values, return_counts=True)

        # Reserve index 0 for unknown token
        self.value_to_idx = {self.unknown_token: 0}
        self.idx_to_value = {0: self.unknown_token}
        current_idx = 1

        # Build vocabulary
        for value, count in zip(unique, counts):
            self.value_counts[value] = count

            # Only include if meets frequency threshold
            if count >= self.min_frequency:
                self.value_to_idx[value] = current_idx
                self.idx_to_value[current_idx] = value
                current_idx += 1

        self.vocab_size = current_idx
        self.is_fitted = True

        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Transform categorical values to indices.

        Args:
            values: Array of categorical values

        Returns:
            Array of encoded indices
        """
        if not self.is_fitted:
            raise ValueError("Must fit encoder before transform")

        # Vectorized transform with unknown token handling
        encoded = np.array([
            self.value_to_idx.get(v, self.unknown_token) for v in values
        ])

        return encoded

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """
        Transform indices back to original values.

        Args:
            indices: Array of encoded indices

        Returns:
            Array of original values
        """
        if not self.is_fitted:
            raise ValueError("Must fit encoder before inverse transform")

        return np.array([self.idx_to_value.get(idx, self.unknown_token) for idx in indices])

    def get_vocab_info(self) -> Dict[str, Any]:
        """Get vocabulary information."""
        return {
            'name': self.name,
            'vocab_size': self.vocab_size,
            'num_unique_values': len(self.value_counts),
            'min_frequency': self.min_frequency,
            'rare_values_filtered': len(self.value_counts) - (self.vocab_size - 1)
        }

    def save(self, path: str):
        """Save encoder state."""
        state = {
            'name': self.name,
            'min_frequency': self.min_frequency,
            'unknown_token': self.unknown_token,
            'value_to_idx': {str(k): v for k, v in self.value_to_idx.items()},
            'idx_to_value': {int(k): str(v) for k, v in self.idx_to_value.items()},
            'value_counts': {str(k): v for k, v in self.value_counts.items()},
            'is_fitted': self.is_fitted,
            'vocab_size': self.vocab_size
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        """Load encoder state."""
        with open(path, 'r') as f:
            state = json.load(f)

        self.name = state['name']
        self.min_frequency = state['min_frequency']
        self.unknown_token = state['unknown_token']

        # Convert string keys back to original types
        self.value_to_idx = state['value_to_idx']
        self.idx_to_value = {int(k): v for k, v in state['idx_to_value'].items()}
        self.value_counts = defaultdict(int, state['value_counts'])
        self.is_fitted = state['is_fitted']
        self.vocab_size = state['vocab_size']


class ContinuousEncoder:
    """
    Continuous feature encoder with normalization and binning.

    Features:
    - Multiple encoding strategies (passthrough, binning, quantile)
    - Statistics tracking
    - Outlier handling
    """

    def __init__(self, name: str, encoding_type: str = "passthrough",
                 num_bins: int = 10, clip_outliers: bool = False):
        self.name = name
        self.encoding_type = encoding_type
        self.num_bins = num_bins
        self.clip_outliers = clip_outliers

        # Statistics
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.percentiles = None

        # Binning
        self.bin_edges = None

        self.is_fitted = False

    def fit(self, values: np.ndarray) -> 'ContinuousEncoder':
        """
        Fit encoder to continuous values.

        Args:
            values: Array of continuous values
        """
        # Compute statistics
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.min = np.min(values)
        self.max = np.max(values)
        self.percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])

        # Setup binning if needed
        if self.encoding_type == "binning":
            self.bin_edges = np.linspace(self.min, self.max, self.num_bins + 1)
        elif self.encoding_type == "quantile":
            self.bin_edges = np.percentile(
                values,
                np.linspace(0, 100, self.num_bins + 1)
            )

        self.is_fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Transform continuous values.

        Args:
            values: Array of continuous values

        Returns:
            Transformed values
        """
        if not self.is_fitted:
            raise ValueError("Must fit encoder before transform")

        # Clip outliers if requested
        if self.clip_outliers:
            values = np.clip(values, self.percentiles[0], self.percentiles[-1])

        if self.encoding_type == "passthrough":
            return values

        elif self.encoding_type in ["binning", "quantile"]:
            # Bin values
            binned = np.digitize(values, self.bin_edges[1:-1])
            return binned

        else:
            return values

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(values)
        return self.transform(values)

    def get_stats(self) -> Dict[str, float]:
        """Get feature statistics."""
        return {
            'name': self.name,
            'mean': float(self.mean) if self.mean is not None else None,
            'std': float(self.std) if self.std is not None else None,
            'min': float(self.min) if self.min is not None else None,
            'max': float(self.max) if self.max is not None else None
        }

    def save(self, path: str):
        """Save encoder state."""
        state = {
            'name': self.name,
            'encoding_type': self.encoding_type,
            'num_bins': self.num_bins,
            'clip_outliers': self.clip_outliers,
            'mean': float(self.mean) if self.mean is not None else None,
            'std': float(self.std) if self.std is not None else None,
            'min': float(self.min) if self.min is not None else None,
            'max': float(self.max) if self.max is not None else None,
            'percentiles': self.percentiles.tolist() if self.percentiles is not None else None,
            'bin_edges': self.bin_edges.tolist() if self.bin_edges is not None else None,
            'is_fitted': self.is_fitted
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        """Load encoder state."""
        with open(path, 'r') as f:
            state = json.load(f)

        self.name = state['name']
        self.encoding_type = state['encoding_type']
        self.num_bins = state['num_bins']
        self.clip_outliers = state['clip_outliers']
        self.mean = state['mean']
        self.std = state['std']
        self.min = state['min']
        self.max = state['max']
        self.percentiles = np.array(state['percentiles']) if state['percentiles'] else None
        self.bin_edges = np.array(state['bin_edges']) if state['bin_edges'] else None
        self.is_fitted = state['is_fitted']


class EncoderRegistry:
    """
    Central registry for managing all feature encoders.

    Features:
    - Automatic encoder creation
    - Batch encoding/decoding
    - State persistence
    - Encoder metadata tracking
    """

    def __init__(self):
        self.categorical_encoders: Dict[str, CategoricalEncoder] = {}
        self.continuous_encoders: Dict[str, ContinuousEncoder] = {}
        self.is_fitted = False

    def register_categorical(self, name: str, min_frequency: int = 1) -> CategoricalEncoder:
        """Register a new categorical encoder."""
        encoder = CategoricalEncoder(name, min_frequency=min_frequency)
        self.categorical_encoders[name] = encoder
        return encoder

    def register_continuous(self, name: str, encoding_type: str = "passthrough",
                           **kwargs) -> ContinuousEncoder:
        """Register a new continuous encoder."""
        encoder = ContinuousEncoder(name, encoding_type=encoding_type, **kwargs)
        self.continuous_encoders[name] = encoder
        return encoder

    def fit(self, data: Dict[str, np.ndarray]):
        """
        Fit all registered encoders.

        Args:
            data: Dictionary mapping feature names to arrays
        """
        for name, encoder in self.categorical_encoders.items():
            if name in data:
                encoder.fit(data[name])

        for name, encoder in self.continuous_encoders.items():
            if name in data:
                encoder.fit(data[name])

        self.is_fitted = True

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted encoders.

        Args:
            data: Dictionary mapping feature names to arrays

        Returns:
            Dictionary of transformed arrays
        """
        if not self.is_fitted:
            raise ValueError("Must fit registry before transform")

        transformed = {}

        for name, encoder in self.categorical_encoders.items():
            if name in data:
                transformed[name] = encoder.transform(data[name])

        for name, encoder in self.continuous_encoders.items():
            if name in data:
                transformed[name] = encoder.transform(data[name])

        return transformed

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all categorical encoders."""
        return {
            name: encoder.vocab_size
            for name, encoder in self.categorical_encoders.items()
        }

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about all encoders."""
        return {
            'categorical': {
                name: encoder.get_vocab_info()
                for name, encoder in self.categorical_encoders.items()
            },
            'continuous': {
                name: encoder.get_stats()
                for name, encoder in self.continuous_encoders.items()
            },
            'is_fitted': self.is_fitted
        }

    def save(self, directory: str):
        """Save all encoders to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save categorical encoders
        for name, encoder in self.categorical_encoders.items():
            encoder.save(dir_path / f"categorical_{name}.json")

        # Save continuous encoders
        for name, encoder in self.continuous_encoders.items():
            encoder.save(dir_path / f"continuous_{name}.json")

        # Save registry metadata
        metadata = {
            'categorical_encoders': list(self.categorical_encoders.keys()),
            'continuous_encoders': list(self.continuous_encoders.keys()),
            'is_fitted': self.is_fitted
        }

        with open(dir_path / "registry_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, directory: str):
        """Load all encoders from directory."""
        dir_path = Path(directory)

        # Load metadata
        with open(dir_path / "registry_metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load categorical encoders
        for name in metadata['categorical_encoders']:
            encoder = CategoricalEncoder(name)
            encoder.load(dir_path / f"categorical_{name}.json")
            self.categorical_encoders[name] = encoder

        # Load continuous encoders
        for name in metadata['continuous_encoders']:
            encoder = ContinuousEncoder(name)
            encoder.load(dir_path / f"continuous_{name}.json")
            self.continuous_encoders[name] = encoder

        self.is_fitted = metadata['is_fitted']
