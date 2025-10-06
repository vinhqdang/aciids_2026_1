"""
Training script for STREAM-FraudX with SEQUENTIAL DATA

This script tests the FULL novel architecture with:
- Temporal-aware attention mechanism
- Multi-scale feature extraction (CNN + LSTM)
- Sequential transaction patterns

This is the CORRECT way to test the research contributions.
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
from torch.utils.data import DataLoader, random_split

from stream_fraudx.data.ieee_cis_sequential import IEEECISSequentialDataset
from stream_fraudx.data.paysim_sequential import PaySimSequentialDataset
from stream_fraudx.models.stream_fraudx_research import STREAMFraudXResearch
from stream_fraudx.baselines.ml_baselines import MLBaselines

warnings.filterwarnings('ignore')


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

    for batch in train_loader:
        # Move to device
        x = batch['sequence_features'].to(device)  # [batch, seq_len, 15]
        timestamps = batch['timestamps'].to(device)  # [batch, seq_len]
        mask = batch['mask'].to(device)  # [batch, seq_len]
        y = batch['label'].to(device)  # [batch]

        optimizer.zero_grad()

        # Forward pass with temporal attention
        logits, attention_weights = model(x, timestamps, mask)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y.cpu().numpy())

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
        for batch in data_loader:
            x = batch['sequence_features'].to(device)
            timestamps = batch['timestamps'].to(device)
            mask = batch['mask'].to(device)
            y = batch['label'].to(device)

            logits, _ = model(x, timestamps, mask)
            loss = criterion(logits, y)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y.cpu().numpy())

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


def train_baselines(dataset, num_samples=10000):
    """Train baseline models on sequential data (flattened)."""
    print("\nTraining baseline models (using flattened sequences)...")

    # Flatten sequences to single-transaction format for baselines
    X = []
    y = []

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        sample = dataset[idx]
        features = sample['sequence_features']  # [seq_len, 15]
        mask = sample['mask']  # [seq_len]
        label = sample['label'].item()

        # Use last valid transaction in sequence
        valid_len = int(mask.sum().item())
        if valid_len > 0:
            last_transaction = features[valid_len - 1].numpy()
            X.append(last_transaction)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train baselines
    baselines = MLBaselines()
    results = baselines.train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\nBaseline results:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: ROC-AUC={metrics['roc_auc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")

    return results


def train_sequential_model(
    train_dataset,
    val_dataset,
    test_dataset,
    input_dim=15,
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    patience=10,
    device='cuda'
):
    """Train STREAM-FraudX on sequential data."""

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = STREAMFraudXResearch(
        input_dim=input_dim,
        embedding_dim=64,
        cnn_channels=32,
        lstm_hidden=64,
        num_transformer_layers=2,
        num_heads=4,
        d_ff=256,
        dropout=0.2
    ).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"\nTraining STREAM-FraudX Sequential Model...")
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
            torch.save(model.state_dict(), 'checkpoints/best_sequential_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('checkpoints/best_sequential_model.pt'))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ieee-cis', choices=['ieee-cis', 'paysim'],
                        help='Dataset to use')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Sequence length')
    parser.add_argument('--min_seq_len', type=int, default=3,
                        help='Minimum sequence length')
    parser.add_argument('--num_sequences', type=int, default=10000,
                        help='Number of sequences to use (for quick testing)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    print("="*80)
    print("STREAM-FraudX Sequential Training (Testing Novel Architecture!)")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Device: {args.device}")

    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Load sequential dataset
    if args.dataset == 'ieee-cis':
        print("\nLoading IEEE-CIS sequential dataset...")
        full_dataset = IEEECISSequentialDataset(
            data_dir='data/ieee-cis',
            split='train',
            seq_len=args.seq_len,
            min_seq_len=args.min_seq_len,
            stride=5
        )
    else:
        print("\nLoading PaySim sequential dataset...")
        full_dataset = PaySimSequentialDataset(
            data_dir='data/paysim',
            seq_len=args.seq_len,
            min_seq_len=args.min_seq_len,
            stride=5,
            fraction=0.2  # Use 20% of PaySim for faster training
        )

    # Limit dataset size if needed
    if args.num_sequences < len(full_dataset):
        indices = np.random.choice(len(full_dataset), args.num_sequences, replace=False)
        from torch.utils.data import Subset
        full_dataset = Subset(full_dataset, indices)
        print(f"Using {args.num_sequences} sequences out of total available")

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train baselines
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)
    baseline_results = train_baselines(full_dataset, num_samples=min(10000, len(full_dataset)))

    # Train sequential model
    print("\n" + "="*80)
    print("STREAM-FraudX SEQUENTIAL MODEL (NOVEL ARCHITECTURE)")
    print("="*80)
    sequential_results = train_sequential_model(
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim=15,
        num_epochs=args.epochs,
        batch_size=32,
        lr=0.001,
        patience=10,
        device=args.device
    )

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    best_baseline_name = max(baseline_results.items(), key=lambda x: x[1]['roc_auc'])
    best_baseline_auc = best_baseline_name[1]['roc_auc']
    best_baseline_name = best_baseline_name[0]

    our_auc = sequential_results['roc_auc']
    diff_pct = ((our_auc - best_baseline_auc) / best_baseline_auc) * 100

    print(f"Best Baseline ({best_baseline_name}): ROC-AUC = {best_baseline_auc:.4f}")
    print(f"STREAM-FraudX Sequential:        ROC-AUC = {our_auc:.4f}")
    print()

    if our_auc > best_baseline_auc:
        print(f"✓ BEAT baseline by {diff_pct:.2f}%!")
        print(f"  Novel architecture with temporal attention WORKS!")
    else:
        print(f"✗ Below best baseline by {-diff_pct:.2f}%")
        print(f"  Continue iterating with:")
        print(f"  - More sequences")
        print(f"  - Longer training")
        print(f"  - Better hyperparameters")

    # Save results
    results = {
        'dataset': args.dataset,
        'seq_len': args.seq_len,
        'baseline_results': baseline_results,
        'sequential_results': sequential_results,
        'best_baseline': best_baseline_name,
        'best_baseline_auc': best_baseline_auc,
        'our_auc': our_auc,
        'improvement': diff_pct
    }

    output_file = f'outputs/sequential_{args.dataset}_results.json'
    with open(output_file, 'w') as f:
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
