"""
Run comprehensive experiments for STREAM-FraudX and baselines.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import json
import time
from pathlib import Path

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.training.trainer import STREAMFraudXTrainer
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn
from stream_fraudx.baselines.ml_baselines import RandomForestBaseline, LogisticRegressionBaseline
from stream_fraudx.utils.metrics import compute_metrics


def run_streamfraudx(train_loader, val_loader, test_loader, device, epochs=20):
    """Run STREAM-FraudX experiments."""
    print("\n" + "="*80)
    print("STREAM-FraudX Experiment")
    print("="*80)

    # Build model
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(10))
    config.categorical_vocab_sizes = [100] * 5

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=True
    )

    # Train
    trainer = STREAMFraudXTrainer(model, device, learning_rate=1e-3)

    best_auprc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)

        if epoch % 5 == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"Epoch {epoch}: Val AUPRC={val_metrics['auprc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}")

            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']

    training_time = time.time() - start_time

    # Final evaluation
    test_metrics = trainer.evaluate(test_loader)
    test_metrics['training_time'] = training_time
    test_metrics['model'] = 'STREAM-FraudX'

    return test_metrics


def run_baseline(baseline, train_loader, test_loader):
    """Run baseline experiment."""
    print(f"\n" + "="*80)
    print(f"{baseline.name} Experiment")
    print("="*80)

    # Collect training data
    print("Collecting training data...")
    train_features = []
    train_labels = []

    for batch in train_loader:
        features = baseline.prepare_features(batch)
        labels = batch['labels'].numpy()
        train_features.append(features)
        train_labels.append(labels)

    X_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    print(f"Training on {len(X_train)} samples...")

    # Train
    start_time = time.time()
    baseline.train(X_train, y_train)
    training_time = time.time() - start_time

    # Test
    print("Evaluating...")
    test_features = []
    test_labels = []

    for batch in test_loader:
        features = baseline.prepare_features(batch)
        labels = batch['labels'].numpy()
        test_features.append(features)
        test_labels.append(labels)

    X_test = np.concatenate(test_features, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    # Predict
    y_scores = baseline.predict(X_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_scores)
    metrics['training_time'] = training_time
    metrics['model'] = baseline.name

    return metrics


def print_results(results):
    """Print comparison table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Model':<20} {'AUPRC':<10} {'ROC-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
    print("-"*80)

    for result in results:
        print(f"{result['model']:<20} "
              f"{result['auprc']:<10.4f} "
              f"{result['roc_auc']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['training_time']:<10.2f}")


def main():
    """Run all experiments."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path('outputs/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data =====
    print("\nLoading data...")
    dataset = SyntheticFraudDataset(
        num_samples=10000,
        num_nodes=1000,
        fraud_rate=0.02,  # 2% fraud rate for better evaluation
        num_continuous=10,
        num_categorical=5,
        seed=42
    )

    # Split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ===== Experiments =====
    results = []

    # 1. STREAM-FraudX
    try:
        result = run_streamfraudx(train_loader, val_loader, test_loader, device, epochs=20)
        results.append(result)
    except Exception as e:
        print(f"STREAM-FraudX failed: {e}")

    # 2. Random Forest
    try:
        rf = RandomForestBaseline(n_estimators=100, max_depth=10)
        result = run_baseline(rf, train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"RandomForest failed: {e}")

    # 3. Logistic Regression
    try:
        lr = LogisticRegressionBaseline()
        result = run_baseline(lr, train_loader, test_loader)
        results.append(result)
    except Exception as e:
        print(f"LogisticRegression failed: {e}")

    # ===== Results =====
    print_results(results)

    # Save results
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/experiment_results.json")

    return results


if __name__ == '__main__':
    results = main()
