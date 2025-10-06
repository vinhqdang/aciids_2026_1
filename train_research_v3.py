"""
Training script for STREAM-FraudX Research Model v3

Novel deep learning architecture for fraud detection.
Goal: Beat XGBoost on IEEE-CIS (0.8790) and PaySim (0.9731) datasets.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import Dataset, DataLoader

from stream_fraudx.data.ieee_cis_loader import IEEECISDataset
from stream_fraudx.data.paysim_loader import PaySimDataset
from stream_fraudx.models.stream_fraudx_research import STREAMFraudXResearchSimple
from stream_fraudx.baselines.ml_baselines import MLBaselines

warnings.filterwarnings('ignore')


def engineer_features(X, dataset_name='ieee-cis'):
    """Advanced feature engineering."""
    X_eng = X.copy()

    # Interaction features
    for i in range(min(5, X.shape[1])):
        for j in range(i + 1, min(5, X.shape[1])):
            X_eng = np.column_stack([X_eng, X[:, i] * X[:, j]])

    # Polynomial features
    for i in range(min(3, X.shape[1])):
        X_eng = np.column_stack([X_eng, X[:, i] ** 2])

    # Log features
    for i in range(min(3, X.shape[1])):
        X_safe = np.abs(X[:, i]) + 1e-8
        X_eng = np.column_stack([X_eng, np.log(X_safe)])

    # Statistical features
    X_eng = np.column_stack([X_eng, np.mean(X, axis=1)])
    X_eng = np.column_stack([X_eng, np.std(X, axis=1)])
    X_eng = np.column_stack([X_eng, np.max(X, axis=1)])
    X_eng = np.column_stack([X_eng, np.min(X, axis=1)])

    return X_eng


class FraudDataset(Dataset):
    """PyTorch dataset for fraud detection."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(train_loader)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    return avg_loss, auc


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()

            # Store predictions
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        # F1 with optimal threshold
        best_f1 = 0.0
        for threshold in np.linspace(0.1, 0.9, 9):
            preds_binary = (all_preds >= threshold).astype(int)
            f1 = f1_score(all_labels, preds_binary, zero_division=0)
            best_f1 = max(best_f1, f1)

    except:
        auc = 0.0
        auprc = 0.0
        best_f1 = 0.0

    return avg_loss, auc, auprc, best_f1


def train_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    input_dim,
    num_epochs=50,
    batch_size=256,
    lr=0.001,
    patience=10,
    device='cuda'
):
    """Train STREAM-FraudX research model."""

    # Create datasets
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = STREAMFraudXResearchSimple(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    ).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Training STREAM-FraudX Research Model...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc, val_auprc, val_f1 = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'checkpoints/best_research_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('checkpoints/best_research_model.pt'))
    test_loss, test_auc, test_auprc, test_f1 = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Results:")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print(f"  AUPRC: {test_auprc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Training time: {train_time:.2f}s")

    return {
        'roc_auc': test_auc,
        'auprc': test_auprc,
        'f1': test_f1,
        'train_time': train_time
    }


def process_dataset(dataset_name, dataset, num_samples, device):
    """Process a single dataset."""
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} Dataset")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading {dataset_name} dataset...")

    # Extract features
    print("Extracting features...")
    X = []
    y = []

    # Sample indices (stratified)
    from tqdm import tqdm
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

    # Feature engineering
    print(f"\nApplying advanced feature engineering...")
    X_engineered = engineer_features(X)
    print(f"Engineered features: {X_engineered.shape[1]} features")

    # Split data
    from sklearn.model_selection import train_test_split

    # First split: train + val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_engineered, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp  # 0.1765 * 0.85 ≈ 0.15
    )

    print(f"Train: {len(y_train)} (fraud: {y_train.sum():.0f}/{len(y_train)})")
    print(f"Val: {len(y_val)} (fraud: {y_val.sum():.0f}/{len(y_val)})")
    print(f"Test: {len(y_test)} (fraud: {y_test.sum():.0f}/{len(y_test)})")

    # Train baseline models
    print(f"\n{'-'*80}")
    print(f"Training Standard Baseline Models")
    print(f"{'-'*80}\n")

    baselines = MLBaselines()
    baseline_results = baselines.train_and_evaluate(X_train, y_train, X_test, y_test)

    for model_name, metrics in baseline_results.items():
        print(f"{model_name}:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}, AUPRC: {metrics['auprc']:.4f}, "
              f"F1: {metrics['f1']:.4f}, Time: {metrics['train_time']:.2f}s")

    # Train research model
    print(f"\n{'-'*80}")
    print(f"Training STREAM-FraudX Research Model (Our Novel Architecture)")
    print(f"{'-'*80}\n")

    research_results = train_model(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        input_dim=X_engineered.shape[1],
        num_epochs=100,
        batch_size=256,
        lr=0.001,
        patience=15,
        device=device
    )

    # Comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")

    # Find best baseline
    best_baseline_name = max(baseline_results.items(), key=lambda x: x[1]['roc_auc'])
    best_baseline_auc = best_baseline_name[1]['roc_auc']
    best_baseline_name = best_baseline_name[0]

    our_auc = research_results['roc_auc']
    diff_pct = ((our_auc - best_baseline_auc) / best_baseline_auc) * 100

    print(f"Best Baseline ({best_baseline_name}): ROC-AUC = {best_baseline_auc:.4f}")
    print(f"Our STREAM-FraudX Research:  ROC-AUC = {our_auc:.4f}")
    print()

    if our_auc > best_baseline_auc:
        print(f"✓ BEAT baseline by {diff_pct:.2f}%!")
    else:
        print(f"✗ Below best baseline by {-diff_pct:.2f}%")
        print(f"  → Continue iterating with better architecture or training")

    # Return all results
    return {
        'baseline_results': baseline_results,
        'research_results': research_results,
        'best_baseline': best_baseline_name,
        'best_baseline_auc': best_baseline_auc,
        'our_auc': our_auc,
        'improvement': diff_pct
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    print("="*80)
    print("STREAM-FraudX Research Model v3")
    print("Novel Deep Learning Architecture for Fraud Detection")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Samples per dataset: {args.num_samples}")

    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Process IEEE-CIS
    ieee_dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train', create_graph=False)
    ieee_results = process_dataset('IEEE-CIS', ieee_dataset, args.num_samples, args.device)

    # Process PaySim
    paysim_dataset = PaySimDataset(data_dir='data/paysim', create_graph=False)
    paysim_results = process_dataset('PaySim', paysim_dataset, args.num_samples, args.device)

    # Save results
    all_results = {
        'ieee_cis': ieee_results,
        'paysim': paysim_results,
        'config': {
            'num_samples': args.num_samples,
            'device': args.device
        }
    }

    with open('outputs/research_v3_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        json.dump(convert_to_serializable(all_results), f, indent=2)

    # Summary
    print(f"\n\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    print("IEEE-CIS:")
    print("-" * 80)
    for model_name, metrics in ieee_results['baseline_results'].items():
        print(f"  {model_name:30s} ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  {'STREAM-FraudX Research (OURS)':30s} ROC-AUC: {ieee_results['our_auc']:.4f}")
    if ieee_results['our_auc'] > ieee_results['best_baseline_auc']:
        print(f"  ✓ BEAT {ieee_results['best_baseline']} by {ieee_results['improvement']:.2f}%!")
    else:
        print(f"  ✗ Below {ieee_results['best_baseline']} by {-ieee_results['improvement']:.2f}%")

    print("\nPaySim:")
    print("-" * 80)
    for model_name, metrics in paysim_results['baseline_results'].items():
        print(f"  {model_name:30s} ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  {'STREAM-FraudX Research (OURS)':30s} ROC-AUC: {paysim_results['our_auc']:.4f}")
    if paysim_results['our_auc'] > paysim_results['best_baseline_auc']:
        print(f"  ✓ BEAT {paysim_results['best_baseline']} by {paysim_results['improvement']:.2f}%!")
    else:
        print(f"  ✗ Below {paysim_results['best_baseline']} by {-paysim_results['improvement']:.2f}%")

    print(f"\n\nResults saved to outputs/research_v3_results.json\n\n")


if __name__ == '__main__':
    main()
