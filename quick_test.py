"""
Quick test to verify STREAM-FraudX implementation.
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
from stream_fraudx.baselines.ml_baselines import RandomForestBaseline
from stream_fraudx.utils.metrics import compute_metrics


def main():
    device = torch.device('cpu')  # Use CPU for quick test
    print(f"Device: {device}")
    print("="*60)

    # Create output directory
    output_dir = Path('outputs/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data (small dataset) =====
    print("\n[1/4] Loading data...")
    dataset = SyntheticFraudDataset(
        num_samples=2000,  # Small dataset for quick test
        num_nodes=200,
        fraud_rate=0.05,  # 5% fraud
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ===== STREAM-FraudX =====
    print("\n[2/4] Training STREAM-FraudX...")
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(10))
    config.categorical_vocab_sizes = [100] * 5

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=True
    )

    print(f"Model parameters: {model.get_model_size()['total']:,}")

    trainer = STREAMFraudXTrainer(model, device, learning_rate=1e-3)

    start_time = time.time()
    for epoch in range(1, 11):  # 10 epochs
        train_metrics = trainer.train_epoch(train_loader, epoch)

        if epoch % 5 == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"Epoch {epoch}: Val AUPRC={val_metrics['auprc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}")

    training_time = time.time() - start_time

    test_metrics = trainer.evaluate(test_loader)

    print(f"\n[3/4] STREAM-FraudX Results:")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Training time: {training_time:.2f}s")

    # ===== Random Forest Baseline =====
    print(f"\n[4/4] Training RandomForest baseline...")
    rf = RandomForestBaseline(n_estimators=50, max_depth=10)

    # Collect training data
    train_features = []
    train_labels = []
    for batch in train_loader:
        features = rf.prepare_features(batch)
        labels = batch['labels'].numpy()
        train_features.append(features)
        train_labels.append(labels)

    X_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    start_time = time.time()
    rf.train(X_train, y_train)
    rf_training_time = time.time() - start_time

    # Test
    test_features = []
    test_labels = []
    for batch in test_loader:
        features = rf.prepare_features(batch)
        labels = batch['labels'].numpy()
        test_features.append(features)
        test_labels.append(labels)

    X_test = np.concatenate(test_features, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    y_scores = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test, y_scores)

    print(f"\nRandomForest Results:")
    print(f"  AUPRC: {rf_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall: {rf_metrics['recall']:.4f}")
    print(f"  F1: {rf_metrics['f1']:.4f}")
    print(f"  Training time: {rf_training_time:.2f}s")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'AUPRC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-"*50)
    print(f"{'STREAM-FraudX':<20} {test_metrics['auprc']:<10.4f} {test_metrics['roc_auc']:<10.4f} {test_metrics['f1']:<10.4f}")
    print(f"{'RandomForest':<20} {rf_metrics['auprc']:<10.4f} {rf_metrics['roc_auc']:<10.4f} {rf_metrics['f1']:<10.4f}")

    # Save results
    results = {
        'stream_fraudx': {
            'metrics': test_metrics,
            'training_time': training_time
        },
        'random_forest': {
            'metrics': rf_metrics,
            'training_time': rf_training_time
        }
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/test_results.json")


if __name__ == '__main__':
    main()
