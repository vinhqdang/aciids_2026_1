"""
Simplified quick test without IRM loss.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import json
import time
from pathlib import Path

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.losses.focal_losses import CombinedFocalLoss
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn
from stream_fraudx.baselines.ml_baselines import RandomForestBaseline
from stream_fraudx.utils.metrics import compute_metrics
import torch.optim as optim
from tqdm import tqdm


def main():
    device = torch.device('cpu')
    print(f"Device: {device}")
    print("="*60)

    output_dir = Path('outputs/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\n[1/4] Loading data...")
    dataset = SyntheticFraudDataset(
        num_samples=2000,
        num_nodes=200,
        fraud_rate=0.05,
        num_continuous=10,
        num_categorical=5,
        seed=42
    )

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

    # Model
    print("\n[2/4] Training STREAM-FraudX...")
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(10))
    config.categorical_vocab_sizes = [100] * 5

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=False  # Disable adapters for simpler test
    ).to(device)

    print(f"Model parameters: {model.get_model_size()['total']:,}")

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = CombinedFocalLoss()

    start_time = time.time()
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/10')
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else
                    ({k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                    for k, v in batch.items()}

            logits = model(batch)
            loss = loss_fn(logits, batch['labels'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})

        if epoch % 5 == 0:
            model.eval()
            all_scores = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else
                            ({k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                            for k, v in batch.items()}

                    logits = model(batch, update_memory=False)
                    scores = torch.sigmoid(logits)

                    all_scores.extend(scores.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

            metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
            print(f"Epoch {epoch}: Val AUPRC={metrics['auprc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    training_time = time.time() - start_time

    # Test
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else
                    ({k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                    for k, v in batch.items()}

            logits = model(batch, update_memory=False)
            scores = torch.sigmoid(logits)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    test_metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    print(f"\n[3/4] STREAM-FraudX Results:")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Training time: {training_time:.2f}s")

    # Random Forest
    print(f"\n[4/4] Training RandomForest baseline...")
    rf = RandomForestBaseline(n_estimators=50, max_depth=10)

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

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'AUPRC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-"*50)
    print(f"{'STREAM-FraudX':<20} {test_metrics['auprc']:<10.4f} {test_metrics['roc_auc']:<10.4f} {test_metrics['f1']:<10.4f}")
    print(f"{'RandomForest':<20} {rf_metrics['auprc']:<10.4f} {rf_metrics['roc_auc']:<10.4f} {rf_metrics['f1']:<10.4f}")

    # Save
    results = {
        'stream_fraudx': {**test_metrics, 'training_time': training_time},
        'random_forest': {**rf_metrics, 'training_time': rf_training_time}
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/test_results.json")
    return results


if __name__ == '__main__':
    main()
