"""
Optimized Model v3 - Beat Random Forest on IEEE-CIS
Strategy: Better RF hyperparameters + feature engineering + larger sample
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
import json
from pathlib import Path

from stream_fraudx.data.ieee_cis_loader import IEEECISDataset
from stream_fraudx.data.paysim_loader import PaySimDataset
from stream_fraudx.baselines.ml_baselines import MLBaselines


def engineer_features_v2(X, dataset_name='ieee-cis'):
    """Enhanced feature engineering with domain knowledge."""
    X_eng = X.copy()

    # Original features
    # Features 0-9: continuous, 10-14: categorical

    # Transaction amount features (assuming it's in continuous features)
    if X.shape[1] >= 10:
        # Amount-based features (first continuous feature is usually amount)
        amount = X[:, 0]
        X_eng = np.column_stack([X_eng, np.log1p(amount)])  # Log amount
        X_eng = np.column_stack([X_eng, np.sqrt(amount)])   # Sqrt amount
        X_eng = np.column_stack([X_eng, amount ** 2])       # Square amount

        # Balance features
        if X.shape[1] >= 5:
            # Balance differences and ratios
            for i in range(1, min(5, X.shape[1])):
                X_eng = np.column_stack([X_eng, X[:, i] / (amount + 1)])  # Ratios

    # Interaction features (top features only)
    for i in range(min(3, X.shape[1])):
        for j in range(i+1, min(5, X.shape[1])):
            X_eng = np.column_stack([X_eng, X[:, i] * X[:, j]])

    # Statistical features
    for i in range(min(10, X.shape[1])):
        X_eng = np.column_stack([X_eng, X[:, i] ** 2])      # Square
        X_eng = np.column_stack([X_eng, np.abs(X[:, i])])   # Abs

    return X_eng


def train_optimized_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest with optimized hyperparameters."""
    print("\nTraining Optimized Random Forest...")

    # Calculate class weight
    class_weight = {
        0: 1.0,
        1: (y_train == 0).sum() / (y_train == 1).sum()
    }

    # Optimized hyperparameters
    rf = RandomForestClassifier(
        n_estimators=500,         # More trees
        max_depth=20,             # Deeper trees
        min_samples_split=20,     # Less aggressive splitting
        min_samples_leaf=10,      # Larger leaves
        max_features='sqrt',      # Feature sampling
        class_weight=class_weight,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf.fit(X_train, y_train)

    # Predict
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  N_estimators: {rf.n_estimators}")
    print(f"  Max_depth: {rf.max_depth}")
    print(f"  OOB Score: {rf.oob_score_:.4f}" if hasattr(rf, 'oob_score_') else "")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  F1: {f1:.4f}")

    return {
        'roc_auc': float(roc_auc),
        'auprc': float(auprc),
        'f1': float(f1)
    }, rf


def train_on_dataset(dataset_name, num_samples=100000):
    """Train on dataset with larger sample."""
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

    # Sample indices (stratified)
    if len(dataset) > num_samples:
        # Get all labels first for stratification
        print("Sampling with stratification...")
        all_labels = []
        sample_size = min(len(dataset), num_samples * 2)
        for idx in tqdm(range(sample_size), desc="Getting labels"):
            all_labels.append(dataset[idx]['labels'].item())
        all_labels = np.array(all_labels)

        # Stratified sampling
        indices = np.arange(sample_size)
        if len(np.unique(all_labels)) > 1:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=num_samples, random_state=42)
            train_idx, _ = next(sss.split(indices, all_labels))
            indices = train_idx
        else:
            indices = np.random.choice(sample_size, num_samples, replace=False)
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

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Fraud rate: {y.mean():.2%}")
    print(f"Fraud samples: {y.sum()}, Non-fraud: {(1-y).sum()}")

    # Feature engineering v2
    print("\nApplying advanced feature engineering...")
    X_engineered = engineer_features_v2(X, dataset_name)
    print(f"Engineered features: {X_engineered.shape[1]} features")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_engineered)

    # Split data (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} (fraud: {y_train.sum()}/{len(y_train)})")
    print(f"Val: {len(X_val)} (fraud: {y_val.sum()}/{len(y_val)})")
    print(f"Test: {len(X_test)} (fraud: {y_test.sum()}/{len(y_test)})")

    # Train standard baselines
    print(f"\n{'-'*80}")
    print("Training Standard Baseline Models")
    print(f"{'-'*80}")

    baselines = MLBaselines()

    # Use original features for baselines
    X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    _, X_test_orig, _, y_test_orig = train_test_split(
        X_temp_orig, y_temp_orig, test_size=0.5, random_state=42, stratify=y_temp_orig
    )

    baseline_results = baselines.train_and_evaluate(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # Train optimized model
    print(f"\n{'-'*80}")
    print("Training Optimized Random Forest (Our Model)")
    print(f"{'-'*80}")

    optimized_results, optimized_model = train_optimized_rf(X_train, y_train, X_test, y_test)

    # Compare
    best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]['roc_auc'])
    best_baseline_roc = baseline_results[best_baseline_name]['roc_auc']

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"Best Baseline ({best_baseline_name}): ROC-AUC = {best_baseline_roc:.4f}")
    print(f"Our Optimized RF:                     ROC-AUC = {optimized_results['roc_auc']:.4f}")

    if optimized_results['roc_auc'] > best_baseline_roc:
        improvement = ((optimized_results['roc_auc'] / best_baseline_roc) - 1) * 100
        print(f"\n✓ WE BEAT THE BEST BASELINE BY {improvement:.2f}%!")
    else:
        gap = ((best_baseline_roc / optimized_results['roc_auc']) - 1) * 100
        print(f"\n✗ Below best baseline by {gap:.2f}%")
        print("  → Need to iterate with better features or models")

    return baseline_results, optimized_results, optimized_model


def main():
    """Main training pipeline."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples')
    args = parser.parse_args()

    np.random.seed(42)

    datasets = ['ieee-cis', 'paysim']
    all_results = {}

    for dataset_name in datasets:
        baseline_results, optimized_results, model = train_on_dataset(dataset_name, args.num_samples)

        all_results[dataset_name] = {
            'baselines': baseline_results,
            'optimized_rf': optimized_results
        }

        # Save model
        if model:
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            import joblib
            joblib.dump(model, checkpoint_dir / f'optimized_rf_v3_{dataset_name}.pkl')

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'optimized_v3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for dataset_name in datasets:
        print(f"{dataset_name.upper()}:")
        print(f"{'-'*80}")

        # Baselines
        for model_name, metrics in all_results[dataset_name]['baselines'].items():
            print(f"  {model_name:<30} ROC-AUC: {metrics['roc_auc']:.4f}")

        # Optimized
        metrics = all_results[dataset_name]['optimized_rf']
        print(f"  {'Optimized RF (OURS)':<30} ROC-AUC: {metrics['roc_auc']:.4f}")
        print()

    print(f"\nResults saved to outputs/optimized_v3_results.json")


if __name__ == '__main__':
    main()
