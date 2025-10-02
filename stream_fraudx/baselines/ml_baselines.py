"""
Traditional ML baselines for fraud detection.
"""

import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class MLBaseline:
    """Base class for ML baselines."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    def prepare_features(self, batch: Dict) -> np.ndarray:
        """Extract features from batch."""
        features = []

        # Continuous features
        if 'continuous' in batch:
            cont = batch['continuous'].numpy() if hasattr(batch['continuous'], 'numpy') else batch['continuous']
            features.append(cont)

        # Categorical features (one-hot or label encoding)
        if 'categorical' in batch:
            cat = batch['categorical'].numpy() if hasattr(batch['categorical'], 'numpy') else batch['categorical']
            features.append(cat)

        return np.concatenate(features, axis=1)

    def train(self, train_data, train_labels):
        """Train the model."""
        raise NotImplementedError

    def predict(self, test_data):
        """Predict probabilities."""
        raise NotImplementedError


class RandomForestBaseline(MLBaseline):
    """Random Forest baseline."""

    def __init__(self, n_estimators=100, max_depth=10):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]


class LogisticRegressionBaseline(MLBaseline):
    """Logistic Regression baseline."""

    def __init__(self):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]
