"""
Complete experimental pipeline for STREAM-FraudX.
Runs experiments on all datasets and saves results.
"""

import torch
import numpy as np
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.losses.focal_losses import AsymmetricFocalLoss  # Use simpler loss
from stream_fraudx.baselines.ml_baselines import RandomForestBaseline, LogisticRegressionBaseline
from stream_fraudx.utils.metrics import compute_metrics
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn

import torch.optim as optim
from tqdm import tqdm


def train_streamfraudx(dataset_name, train_loader, val_loader, test_loader, device, epochs=30):
    """Train STREAM-FraudX model."""

    print(f"\n{'='*80}")
    print(f"Training STREAM-FraudX on {dataset_name}")
    print(f"{'='*80}")

    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(10))
    config.categorical_vocab_sizes = [100] * 5

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=False  # Disable adapters for stable training
    ).to(device)

    print(f"Model parameters: {model.get_model_size()['total']:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = AsymmetricFocalLoss()  # Simpler loss to avoid gradient issues

    best_auprc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for batch in pbar:
            try:
                # Move to device
                batch = {k: v.to(device) if torch.is_tensor(v) else
                        ({k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                        for k, v in batch.items()}

                # Forward pass
                logits = model(batch)
                loss = loss_fn(logits, batch['labels'])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

            except RuntimeError as e:
                print(f"\nError in training batch: {e}")
                continue

        # Validation
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
            print(f"\nEpoch {epoch}: Val AUPRC={metrics['auprc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

            if metrics['auprc'] > best_auprc:
                best_auprc = metrics['auprc']

    training_time = time.time() - start_time

    # Final test evaluation
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
    test_metrics['training_time'] = training_time
    test_metrics['model'] = 'STREAM-FraudX'
    test_metrics['dataset'] = dataset_name

    return test_metrics


def train_baseline(baseline, dataset_name, train_loader, test_loader):
    """Train baseline model."""

    print(f"\n{'='*80}")
    print(f"Training {baseline.name} on {dataset_name}")
    print(f"{'='*80}")

    # Collect data
    train_features = []
    train_labels = []

    for batch in tqdm(train_loader, desc='Collecting training data'):
        features = baseline.prepare_features(batch)
        labels = batch['labels'].numpy()
        train_features.append(features)
        train_labels.append(labels)

    X_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    # Train
    start_time = time.time()
    baseline.train(X_train, y_train)
    training_time = time.time() - start_time

    # Test
    test_features = []
    test_labels = []

    for batch in tqdm(test_loader, desc='Evaluating'):
        features = baseline.prepare_features(batch)
        labels = batch['labels'].numpy()
        test_features.append(features)
        test_labels.append(labels)

    X_test = np.concatenate(test_features, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    y_scores = baseline.predict(X_test)

    metrics = compute_metrics(y_test, y_scores)
    metrics['training_time'] = training_time
    metrics['model'] = baseline.name
    metrics['dataset'] = dataset_name

    return metrics


def run_on_dataset(dataset_name, dataset, device, batch_size=256):
    """Run all experiments on a single dataset."""

    print(f"\n{'#'*80}")
    print(f"# Dataset: {dataset_name}")
    print(f"# Size: {len(dataset):,} transactions")
    print(f"{'#'*80}")

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    results = []

    # 1. STREAM-FraudX
    try:
        result = train_streamfraudx(dataset_name, train_loader, val_loader, test_loader, device, epochs=20)
        results.append(result)
        print(f"\n✓ STREAM-FraudX: AUPRC={result['auprc']:.4f}")
    except Exception as e:
        print(f"\n✗ STREAM-FraudX failed: {e}")

    # 2. Random Forest
    try:
        rf = RandomForestBaseline(n_estimators=100, max_depth=10)
        result = train_baseline(rf, dataset_name, train_loader, test_loader)
        results.append(result)
        print(f"\n✓ RandomForest: AUPRC={result['auprc']:.4f}")
    except Exception as e:
        print(f"\n✗ RandomForest failed: {e}")

    # 3. Logistic Regression
    try:
        lr = LogisticRegressionBaseline()
        result = train_baseline(lr, dataset_name, train_loader, test_loader)
        results.append(result)
        print(f"\n✓ LogisticRegression: AUPRC={result['auprc']:.4f}")
    except Exception as e:
        print(f"\n✗ LogisticRegression failed: {e}")

    return results


def main(args):
    """Run all experiments."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = []

    # Experiment 1: Synthetic Data (small, for quick validation)
    print("\n" + "="*80)
    print("Experiment 1: Synthetic Data (Validation)")
    print("="*80)

    synthetic_dataset = SyntheticFraudDataset(
        num_samples=5000,
        num_nodes=500,
        fraud_rate=0.05,
        num_continuous=10,
        num_categorical=5,
        seed=42
    )

    results = run_on_dataset('Synthetic', synthetic_dataset, device, batch_size=128)
    all_results.extend(results)

    # Try real datasets if available
    datasets_to_try = [
        ('PaySim', 'stream_fraudx.data.paysim_loader', 'PaySimDataset',
         {'data_dir': 'data/paysim', 'fraction': 0.05}),  # 5% for speed
        ('IEEE-CIS', 'stream_fraudx.data.ieee_cis_loader', 'IEEECISDataset',
         {'data_dir': 'data/ieee-cis', 'split': 'train'}),
        ('Elliptic', 'stream_fraudx.data.elliptic_loader', 'EllipticDataset',
         {'data_dir': 'data/elliptic', 'use_labeled_only': True})
    ]

    for dataset_name, module_name, class_name, kwargs in datasets_to_try:
        try:
            print(f"\n" + "="*80)
            print(f"Experiment: {dataset_name}")
            print("="*80)

            module = __import__(module_name, fromlist=[class_name])
            DatasetClass = getattr(module, class_name)
            dataset = DatasetClass(**kwargs)

            results = run_on_dataset(dataset_name, dataset, device, batch_size=256)
            all_results.extend(results)

        except FileNotFoundError:
            print(f"\n⚠ {dataset_name} not found, skipping...")
        except Exception as e:
            print(f"\n✗ {dataset_name} failed: {e}")

    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='results_experiment.json')
    args = parser.parse_args()

    results = main(args)
