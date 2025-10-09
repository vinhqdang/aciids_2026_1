"""
Base data loader with configurable preprocessing pipeline.
Provides composable preprocessing steps for consistent data processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, robust, none
    scale_continuous: bool = True
    scale_categorical: bool = False

    # Normalization
    normalize: bool = False
    norm_type: str = "l2"  # l1, l2, max

    # Feature engineering
    add_polynomial_features: bool = False
    polynomial_degree: int = 2
    add_interaction_features: bool = False
    max_interactions: int = 10

    # Missing values
    fill_missing: bool = True
    fill_strategy: str = "mean"  # mean, median, mode, zero

    # Encoding
    categorical_encoding: str = "label"  # label, onehot, embedding


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> 'PreprocessingStep':
        """Fit the preprocessing step to data."""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def save_state(self) -> Dict[str, Any]:
        """Save preprocessing state for persistence."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted
        }

    def load_state(self, state: Dict[str, Any]):
        """Load preprocessing state."""
        self.name = state['name']
        self.is_fitted = state['is_fitted']


class ScalingStep(PreprocessingStep):
    """Scaling preprocessing step."""

    def __init__(self, method: str = "standard"):
        super().__init__(f"scaling_{method}")
        self.method = method

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def fit(self, data: np.ndarray) -> 'ScalingStep':
        if self.scaler is not None:
            self.scaler.fit(data)
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        if self.scaler is None:
            return data
        return self.scaler.transform(data)

    def save_state(self) -> Dict[str, Any]:
        state = super().save_state()
        state['method'] = self.method
        if self.scaler is not None:
            state['scaler_params'] = {
                'mean_': getattr(self.scaler, 'mean_', None),
                'scale_': getattr(self.scaler, 'scale_', None),
                'center_': getattr(self.scaler, 'center_', None)
            }
        return state

    def load_state(self, state: Dict[str, Any]):
        super().load_state(state)
        self.method = state['method']
        if 'scaler_params' in state and self.scaler is not None:
            params = state['scaler_params']
            if params['mean_'] is not None:
                self.scaler.mean_ = params['mean_']
            if params['scale_'] is not None:
                self.scaler.scale_ = params['scale_']
            if params['center_'] is not None:
                self.scaler.center_ = params['center_']


class NormalizationStep(PreprocessingStep):
    """Normalization preprocessing step."""

    def __init__(self, norm_type: str = "l2"):
        super().__init__(f"normalization_{norm_type}")
        self.norm_type = norm_type

    def fit(self, data: np.ndarray) -> 'NormalizationStep':
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")

        if self.norm_type == "l1":
            norms = np.abs(data).sum(axis=1, keepdims=True)
        elif self.norm_type == "l2":
            norms = np.sqrt((data ** 2).sum(axis=1, keepdims=True))
        elif self.norm_type == "max":
            norms = np.abs(data).max(axis=1, keepdims=True)
        else:
            return data

        norms[norms == 0] = 1  # Avoid division by zero
        return data / norms


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline.

    Chains multiple preprocessing steps together and manages state persistence.
    """

    def __init__(self, steps: Optional[List[PreprocessingStep]] = None):
        self.steps = steps or []
        self.is_fitted = False

    def add_step(self, step: PreprocessingStep):
        """Add a preprocessing step to the pipeline."""
        self.steps.append(step)

    def fit(self, data: np.ndarray) -> 'PreprocessingPipeline':
        """Fit all steps in the pipeline."""
        current_data = data
        for step in self.steps:
            step.fit(current_data)
            current_data = step.transform(current_data)
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data through all steps."""
        if not self.is_fitted:
            raise ValueError("Must fit pipeline before transform")

        current_data = data
        for step in self.steps:
            current_data = step.transform(current_data)
        return current_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        """Save pipeline state to disk."""
        state = {
            'is_fitted': self.is_fitted,
            'steps': [step.save_state() for step in self.steps]
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load pipeline state from disk."""
        state = torch.load(path)
        self.is_fitted = state['is_fitted']
        # Note: Need to reconstruct steps with proper classes
        # This is a simplified version

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> 'PreprocessingPipeline':
        """Create pipeline from configuration."""
        pipeline = cls()

        # Add scaling
        if config.scale_continuous and config.scaling_method != "none":
            pipeline.add_step(ScalingStep(config.scaling_method))

        # Add normalization
        if config.normalize:
            pipeline.add_step(NormalizationStep(config.norm_type))

        return pipeline


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    Provides interface for loading, preprocessing, and batching fraud detection data.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.continuous_pipeline = PreprocessingPipeline.from_config(config)
        self.categorical_pipeline = PreprocessingPipeline()
        self.is_fitted = False

    @abstractmethod
    def load_raw_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from source.

        Returns:
            Tuple of (features, labels)
        """
        pass

    @abstractmethod
    def extract_features(self, raw_data: Any) -> Dict[str, np.ndarray]:
        """
        Extract continuous and categorical features.

        Returns:
            Dictionary with 'continuous' and 'categorical' keys
        """
        pass

    def fit_preprocessing(self, data: Dict[str, np.ndarray]):
        """Fit preprocessing pipelines to training data."""
        if 'continuous' in data and data['continuous'] is not None:
            self.continuous_pipeline.fit(data['continuous'])

        if 'categorical' in data and data['categorical'] is not None:
            # Categorical features usually don't need fitting for label encoding
            pass

        self.is_fitted = True

    def transform_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform features using fitted preprocessing."""
        if not self.is_fitted:
            raise ValueError("Must fit preprocessing before transform")

        transformed = {}

        if 'continuous' in data and data['continuous'] is not None:
            transformed['continuous'] = self.continuous_pipeline.transform(data['continuous'])

        if 'categorical' in data and data['categorical'] is not None:
            transformed['categorical'] = data['categorical']  # Keep as-is for now

        return transformed

    def save_preprocessing(self, path: str):
        """Save fitted preprocessing pipelines."""
        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'continuous_pipeline': self.continuous_pipeline,
            'categorical_pipeline': self.categorical_pipeline
        }
        torch.save(state, path)

    def load_preprocessing(self, path: str):
        """Load fitted preprocessing pipelines."""
        state = torch.load(path)
        self.config = state['config']
        self.is_fitted = state['is_fitted']
        self.continuous_pipeline = state['continuous_pipeline']
        self.categorical_pipeline = state['categorical_pipeline']
