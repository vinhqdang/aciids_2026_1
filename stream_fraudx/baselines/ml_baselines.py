"""
Traditional ML baselines for fraud detection.
Includes: Random Forest, Logistic Regression, LightGBM, XGBoost, CatBoost
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


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


class LightGBMBaseline(MLBaseline):
    """LightGBM gradient boosting baseline."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 scale_pos_weight: Optional[float] = None):
        super().__init__("LightGBM")

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight

        self.model = lgb.LGBMClassifier(**params)

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]


class XGBoostBaseline(MLBaseline):
    """XGBoost gradient boosting baseline."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 scale_pos_weight: Optional[float] = None,
                 use_gpu: bool = False):
        super().__init__("XGBoost")

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'gpu_hist' if use_gpu else 'hist'
        }

        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight

        self.model = xgb.XGBClassifier(**params)

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]


class CatBoostBaseline(MLBaseline):
    """CatBoost gradient boosting baseline."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 scale_pos_weight: Optional[float] = None,
                 use_gpu: bool = False):
        super().__init__("CatBoost")

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")

        params = {
            'iterations': n_estimators,
            'depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': 42,
            'verbose': False,
            'task_type': 'GPU' if use_gpu else 'CPU'
        }

        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight

        self.model = cb.CatBoostClassifier(**params)

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]
