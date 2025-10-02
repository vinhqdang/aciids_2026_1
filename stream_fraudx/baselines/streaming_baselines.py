"""
Streaming ML baselines using River library.
Includes: Adaptive Random Forest, Hoeffding Tree
"""

import numpy as np
from typing import Dict, Optional

try:
    from river import ensemble, tree, preprocessing, compose
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class StreamingBaseline:
    """Base class for streaming baselines."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    def prepare_features(self, batch: Dict) -> list:
        """
        Extract features from batch and convert to list of dicts for River.

        Args:
            batch: Batch dictionary with 'continuous' and 'categorical' tensors

        Returns:
            List of feature dictionaries
        """
        import torch

        batch_size = batch['continuous'].size(0) if torch.is_tensor(batch['continuous']) else len(batch['continuous'])

        samples = []
        for i in range(batch_size):
            sample = {}

            # Continuous features
            if 'continuous' in batch:
                cont = batch['continuous'][i]
                if torch.is_tensor(cont):
                    cont = cont.cpu().numpy()
                for j, val in enumerate(cont):
                    sample[f'cont_{j}'] = float(val)

            # Categorical features
            if 'categorical' in batch:
                cat = batch['categorical'][i]
                if torch.is_tensor(cat):
                    cat = cat.cpu().numpy()
                for j, val in enumerate(cat):
                    sample[f'cat_{j}'] = int(val)

            samples.append(sample)

        return samples

    def update(self, x: Dict, y: int):
        """Update model with one sample (online learning)."""
        raise NotImplementedError

    def predict_proba(self, x: Dict) -> float:
        """Predict probability for one sample."""
        raise NotImplementedError

    def train_batch(self, train_data: list, train_labels: list):
        """Train on a batch of samples."""
        for x, y in zip(train_data, train_labels):
            self.update(x, int(y))

    def predict(self, test_data: list) -> np.ndarray:
        """Predict probabilities for a batch of samples."""
        predictions = [self.predict_proba(x) for x in test_data]
        return np.array(predictions)


class AdaptiveRandomForestBaseline(StreamingBaseline):
    """
    Adaptive Random Forest for streaming data.

    Uses River's ensemble.AdaptiveRandomForestClassifier.
    """

    def __init__(self,
                 n_models: int = 10,
                 max_depth: Optional[int] = None,
                 grace_period: int = 200):
        super().__init__("AdaptiveRandomForest")

        if not RIVER_AVAILABLE:
            raise ImportError("River not installed. Install with: pip install river")

        self.model = ensemble.AdaptiveRandomForestClassifier(
            n_models=n_models,
            max_depth=max_depth,
            grace_period=grace_period,
            seed=42
        )

    def update(self, x: Dict, y: int):
        """Update model with one sample."""
        self.model.learn_one(x, y)

    def predict_proba(self, x: Dict) -> float:
        """Predict fraud probability for one sample."""
        proba = self.model.predict_proba_one(x)
        # Return probability of positive class (1)
        return proba.get(1, 0.0)


class HoeffdingTreeBaseline(StreamingBaseline):
    """
    Hoeffding Tree (VFDT) for streaming data.

    Uses River's tree.HoeffdingTreeClassifier.
    """

    def __init__(self,
                 max_depth: Optional[int] = None,
                 grace_period: int = 200,
                 delta: float = 1e-7):
        super().__init__("HoeffdingTree")

        if not RIVER_AVAILABLE:
            raise ImportError("River not installed. Install with: pip install river")

        self.model = tree.HoeffdingTreeClassifier(
            max_depth=max_depth,
            grace_period=grace_period,
            delta=delta
        )

    def update(self, x: Dict, y: int):
        """Update model with one sample."""
        self.model.learn_one(x, y)

    def predict_proba(self, x: Dict) -> float:
        """Predict fraud probability for one sample."""
        proba = self.model.predict_proba_one(x)
        return proba.get(1, 0.0)


class HoeffdingAdaptiveTreeBaseline(StreamingBaseline):
    """
    Hoeffding Adaptive Tree for streaming data with concept drift.

    Uses River's tree.HoeffdingAdaptiveTreeClassifier.
    """

    def __init__(self,
                 max_depth: Optional[int] = None,
                 grace_period: int = 200):
        super().__init__("HoeffdingAdaptiveTree")

        if not RIVER_AVAILABLE:
            raise ImportError("River not installed. Install with: pip install river")

        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            max_depth=max_depth,
            grace_period=grace_period,
            seed=42
        )

    def update(self, x: Dict, y: int):
        """Update model with one sample."""
        self.model.learn_one(x, y)

    def predict_proba(self, x: Dict) -> float:
        """Predict fraud probability for one sample."""
        proba = self.model.predict_proba_one(x)
        return proba.get(1, 0.0)
