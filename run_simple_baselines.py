"""
Simple baseline comparison for STREAM-FraudX
Tests LightGBM, XGBoost, and TabTransformer on synthetic data.
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

from stream_fraudx.data.synthetic_data import SyntheticFraudDataset
from stream_fraudx.baselines.ml_baselines import LightGBMBaseline, XGBoostBaseline, LogisticRegressionBaseline
from stream_fraudx.utils.metrics import compute_metrics

def prepare_data(num_samples=20000):
    """Prepare train/test data."""
    print(f"Generating {num_samples} samples...")
    dataset = SyntheticFraudDataset(
        num_samples=num_samples,
        num_nodes=1000,
        fraud_rate=0.05,
        num_continuous=10,
        num_categorical=5
    )

    # Extract features and labels
    features = []
    labels = []

    for i in range(len(dataset)):
        sample = dataset[i]
        # Concatenate continuous and categorical
        cont = sample['continuous'].numpy()
        cat = sample['categorical'].numpy()
        feat = np.concatenate([cont, cat])
        features.append(feat)
        labels.append(sample['labels'].item())

    X = np.array(features)
    y = np.array(labels)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Fraud rate: {y_train.mean():.3f}")

    return X_train, X_test, y_train, y_test


def run_baseline(baseline_class, name, X_train, X_test, y_train, y_test, **kwargs):
    """Run a single baseline."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    # Create and train
    baseline = baseline_class(**kwargs)
    print("Training...")
    baseline.train(X_train, y_train)

    # Predict
    print("Predicting...")
    y_pred = baseline.predict(X_test)

    # Evaluate
    metrics = compute_metrics(y_test, y_pred)

    print(f"\nResults:")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    return {
        'name': name,
        'metrics': metrics
    }


def main():
    """Run all baselines."""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(num_samples=20000)

    # Compute scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    results = []

    # Logistic Regression
    try:
        result = run_baseline(
            LogisticRegressionBaseline,
            "Logistic Regression",
            X_train, X_test, y_train, y_test
        )
        results.append(result)
    except Exception as e:
        print(f"Error: {e}")

    # LightGBM
    try:
        result = run_baseline(
            LightGBMBaseline,
            "LightGBM",
            X_train, X_test, y_train, y_test,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight
        )
        results.append(result)
    except Exception as e:
        print(f"Error: {e}")

    # XGBoost
    try:
        result = run_baseline(
            XGBoostBaseline,
            "XGBoost",
            X_train, X_test, y_train, y_test,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_gpu=torch.cuda.is_available()
        )
        results.append(result)
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'AUPRC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-" * 60)

    for result in results:
        print(f"{result['name']:<25} "
              f"{result['metrics']['auprc']:<10.4f} "
              f"{result['metrics']['roc_auc']:<10.4f} "
              f"{result['metrics']['f1']:<10.4f}")

    # Save results
    output_dir = Path('outputs/baselines')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to outputs/baselines/baseline_results.json")


if __name__ == '__main__':
    main()
