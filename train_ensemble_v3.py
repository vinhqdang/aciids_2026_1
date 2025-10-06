"""
Enhanced Ensemble Model v3 - Beat Random Forest/XGBoost
Strategy: Advanced feature engineering + stacking ensemble + AutoML
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
import json
from pathlib import Path

from stream_fraudx.data.ieee_cis_loader import IEEECISDataset
from stream_fraudx.data.paysim_loader import PaySimDataset
from stream_fraudx.baselines.ml_baselines import MLBaselines


def engineer_features(X, dataset_name='ieee-cis'):
    """Advanced feature engineering."""
    X_eng = X.copy()

    # Interaction features
    for i in range(min(5, X.shape[1])):
        for j in range(i+1, min(5, X.shape[1])):
            X_eng = np.column_stack([X_eng, X[:, i] * X[:, j]])

    # Polynomial features (degree 2 for first 5 features)
    for i in range(min(5, X.shape[1])):
        X_eng = np.column_stack([X_eng, X[:, i] ** 2])

    # Log transform for positive features
    X_pos = np.abs(X) + 1e-8
    X_eng = np.column_stack([X_eng, np.log(X_pos[:, :min(10, X.shape[1])])])

    return X_eng


def create_stacking_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create a stacking ensemble of models."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    try:
        import lightgbm as lgb
        import xgboost as xgb
        import catboost as cb

        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Base learners
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                         min_samples_split=10, min_samples_leaf=4,
                                         class_weight='balanced', random_state=42, n_jobs=-1)),
            ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                       num_leaves=64, min_child_samples=20,
                                       scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                      scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
                                      tree_method='hist')),
            ('catboost', cb.CatBoostClassifier(iterations=200, depth=8, learning_rate=0.05,
                                               scale_pos_weight=scale_pos_weight, random_state=42, verbose=False))
        ]

        # Meta learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1
        )

        print("Training stacking ensemble...")
        stacking_clf.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        return {
            'roc_auc': float(roc_auc),
            'auprc': float(auprc),
            'f1': float(f1)
        }, stacking_clf

    except ImportError as e:
        print(f"Missing library: {e}")
        return None, None


def train_on_dataset(dataset_name, num_samples=50000):
    """Train ensemble on a dataset."""
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()} Dataset")
    print(f"{'='*80}\n")

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'ieee-cis':
        dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train', create_graph=False)
    elif dataset_name == 'paysim':
        dataset = PaySimDataset(data_dir='data/paysim', create_graph=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Extract features
    print("Extracting features...")
    X = []
    y = []

    # Sample indices
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = np.arange(len(dataset))

    for idx in tqdm(indices, desc="Extracting"):
        sample = dataset[idx]
        features = np.concatenate([
            sample['continuous'].numpy(),
            sample['categorical'].numpy().astype(float)
        ])
        X.append(features)
        y.append(sample['labels'].item())

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Fraud rate: {y.mean():.2%}")

    # Feature engineering
    print("\nApplying feature engineering...")
    X_engineered = engineer_features(X, dataset_name)
    print(f"Engineered features: {X_engineered.shape[1]} features")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_engineered)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train baselines first
    print(f"\n{'-'*80}")
    print("Training Baseline Models")
    print(f"{'-'*80}")

    baselines = MLBaselines()
    baseline_results = {}

    # Train on original features
    X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test_orig, _, y_test_orig = train_test_split(X_temp_orig, y_temp_orig, test_size=0.5, random_state=42, stratify=y_temp_orig)

    baseline_results = baselines.train_and_evaluate(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # Train stacking ensemble
    print(f"\n{'-'*80}")
    print("Training Stacking Ensemble (Our Model)")
    print(f"{'-'*80}")

    ensemble_results, ensemble_model = create_stacking_ensemble(X_train, y_train, X_val, y_val, X_test, y_test)

    if ensemble_results:
        print(f"\nStacking Ensemble Results:")
        print(f"  ROC-AUC: {ensemble_results['roc_auc']:.4f}")
        print(f"  AUPRC:   {ensemble_results['auprc']:.4f}")
        print(f"  F1:      {ensemble_results['f1']:.4f}")

        # Compare with best baseline
        best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]['roc_auc'])
        best_baseline_roc = baseline_results[best_baseline_name]['roc_auc']

        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"Best Baseline ({best_baseline_name}): ROC-AUC = {best_baseline_roc:.4f}")
        print(f"Our Ensemble:                         ROC-AUC = {ensemble_results['roc_auc']:.4f}")

        if ensemble_results['roc_auc'] > best_baseline_roc:
            improvement = ((ensemble_results['roc_auc'] / best_baseline_roc) - 1) * 100
            print(f"\n✓ WE BEAT THE BEST BASELINE BY {improvement:.2f}%!")
        else:
            gap = ((best_baseline_roc / ensemble_results['roc_auc']) - 1) * 100
            print(f"\n✗ Below best baseline by {gap:.2f}%")

    return baseline_results, ensemble_results, ensemble_model


def main():
    """Main training pipeline."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    args = parser.parse_args()

    np.random.seed(42)

    datasets = ['ieee-cis', 'paysim']
    all_results = {}

    for dataset_name in datasets:
        baseline_results, ensemble_results, model = train_on_dataset(dataset_name, args.num_samples)

        all_results[dataset_name] = {
            'baselines': baseline_results,
            'ensemble': ensemble_results
        }

        # Save model
        if model:
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            import joblib
            joblib.dump(model, checkpoint_dir / f'ensemble_v3_{dataset_name}.pkl')

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'ensemble_v3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for dataset_name in datasets:
        print(f"{dataset_name.upper()}:")
        print(f"{'-'*80}")

        # Baselines
        for model_name, metrics in all_results[dataset_name]['baselines'].items():
            print(f"  {model_name:<25} ROC-AUC: {metrics['roc_auc']:.4f}")

        # Ensemble
        if all_results[dataset_name]['ensemble']:
            metrics = all_results[dataset_name]['ensemble']
            print(f"  {'Stacking Ensemble (OURS)':<25} ROC-AUC: {metrics['roc_auc']:.4f}")

        print()

    print(f"\nResults saved to outputs/ensemble_v3_results.json")


if __name__ == '__main__':
    main()
