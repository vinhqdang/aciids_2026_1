"""
STREAM-FraudX v3: Production Training Pipeline
Goal: Beat XGBoost on ROC-AUC for IEEE-CIS and PaySim datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from stream_fraudx.data.ieee_cis_loader import IEEECISDataset
from stream_fraudx.data.paysim_loader import PaySimDataset
from stream_fraudx.models.stream_fraudx import STREAMFraudX
from stream_fraudx.losses.focal_losses import CombinedFocalLoss
from stream_fraudx.baselines.ml_baselines import MLBaselines


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate function for batching."""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'edge_attr_discrete':
            # Handle nested dict
            collated[key] = {
                k: torch.stack([b[key][k] for b in batch])
                for k in batch[0][key].keys()
            }
        else:
            collated[key] = torch.stack([b[key] for b in batch])

    return collated


def create_dataloaders(dataset_name, batch_size=256, num_samples=50000):
    """Create train/val/test dataloaders."""
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} Dataset")
    print(f"{'='*80}")

    # Load dataset
    if dataset_name == 'ieee-cis':
        dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train', create_graph=True)
    elif dataset_name == 'paysim':
        dataset = PaySimDataset(data_dir='data/paysim', create_graph=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample if needed
    if num_samples and len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset_subset = Subset(dataset, indices)
    else:
        dataset_subset = dataset
        indices = np.arange(len(dataset))

    # Split into train/val/test
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, dataset


def compute_metrics(y_true, y_scores):
    """Compute evaluation metrics."""
    y_pred = (y_scores > 0.5).astype(int)

    return {
        'roc_auc': roc_auc_score(y_true, y_scores),
        'auprc': average_precision_score(y_true, y_scores),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_labels = []
    all_scores = []

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()}

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch)

        # Compute loss
        loss = criterion(logits.squeeze(), batch['labels'])

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        all_labels.extend(labels)
        all_scores.extend(scores)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    return avg_loss, metrics


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                    {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
                    for k, v in batch.items()}

            # Forward pass
            logits = model(batch)
            loss = criterion(logits.squeeze(), batch['labels'])

            # Track metrics
            total_loss += loss.item()
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_labels.extend(labels)
            all_scores.extend(scores)

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))

    return avg_loss, metrics


def train_stream_fraudx(dataset_name, config):
    """Train STREAM-FraudX model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        dataset_name,
        batch_size=config['batch_size'],
        num_samples=config['num_samples']
    )

    # Initialize model
    model = STREAMFraudX(
        graph_node_dim=config['node_dim'],
        graph_hidden_dim=config['hidden_dim'],
        graph_num_layers=config['num_gnn_layers'],
        tabular_embedding_dim=config['tabular_dim'],
        tabular_num_layers=config['num_transformer_layers'],
        tabular_num_heads=config['num_heads'],
        fusion_hidden_dim=config['hidden_dim'],
        head_hidden_dim=config['hidden_dim'] // 2
    ).to(device)

    # Loss and optimizer
    criterion = CombinedFocalLoss(
        alpha=config['focal_alpha'],
        beta=config['focal_beta'],
        gamma_pos=config['focal_gamma_pos'],
        gamma_neg=config['focal_gamma_neg']
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] / 10
    )

    # Training loop
    best_val_roc_auc = 0
    best_model_state = None

    print(f"\n{'='*80}")
    print("Training STREAM-FraudX")
    print(f"{'='*80}")

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"Train - Loss: {train_loss:.4f}, ROC-AUC: {train_metrics['roc_auc']:.4f}, "
              f"AUPRC: {train_metrics['auprc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}, "
              f"AUPRC: {val_metrics['auprc']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['roc_auc'] > best_val_roc_auc:
            best_val_roc_auc = val_metrics['roc_auc']
            best_model_state = model.state_dict().copy()
            print(f"  → New best model! ROC-AUC: {best_val_roc_auc:.4f}")

    # Test with best model
    model.load_state_dict(best_model_state)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*80}")
    print("Test Results")
    print(f"{'='*80}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"AUPRC:   {test_metrics['auprc']:.4f}")
    print(f"F1:      {test_metrics['f1']:.4f}")

    return test_metrics, best_model_state


def train_baselines(dataset_name, num_samples=50000):
    """Train baseline models (XGBoost, LightGBM, LogReg)."""
    print(f"\n{'='*80}")
    print(f"Training Baselines on {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    if dataset_name == 'ieee-cis':
        dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train', create_graph=False)
    elif dataset_name == 'paysim':
        dataset = PaySimDataset(data_dir='data/paysim', create_graph=False)

    # Subsample
    if num_samples and len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = np.arange(len(dataset))

    # Extract features and labels
    X = []
    y = []
    for idx in tqdm(indices, desc="Extracting features"):
        sample = dataset[idx]
        # Concatenate continuous and categorical features
        features = np.concatenate([
            sample['continuous'].numpy(),
            sample['categorical'].numpy().astype(float)
        ])
        X.append(features)
        y.append(sample['labels'].item())

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train baselines
    baselines = MLBaselines()
    results = baselines.train_and_evaluate(X_train, y_train, X_test, y_test)

    return results


def main():
    """Main training pipeline."""
    set_seed(42)

    # Configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    args = parser.parse_args()

    config = {
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'node_dim': 128,
        'tabular_dim': 128,
        'hidden_dim': 256,
        'num_gnn_layers': 3,
        'num_transformer_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'focal_alpha': 0.7,
        'focal_beta': 0.3,
        'focal_gamma_pos': 0.0,
        'focal_gamma_neg': 2.0
    }

    datasets = ['ieee-cis', 'paysim']
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing {dataset_name.upper()} Dataset")
        print(f"{'#'*80}")

        # Train baselines
        baseline_results = train_baselines(dataset_name, config['num_samples'])

        # Train STREAM-FraudX
        stream_fraudx_results, model_state = train_stream_fraudx(dataset_name, config)

        # Save results
        all_results[dataset_name] = {
            'baselines': baseline_results,
            'stream_fraudx': stream_fraudx_results
        }

        # Save model checkpoint
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(model_state, checkpoint_dir / f'stream_fraudx_v3_{dataset_name}.pth')

    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}\n")

    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"{'-'*80}")

        # Baselines
        print("\nBaselines:")
        for model_name, metrics in all_results[dataset_name]['baselines'].items():
            print(f"  {model_name:<20} ROC-AUC: {metrics['roc_auc']:.4f}  "
                  f"AUPRC: {metrics['auprc']:.4f}  F1: {metrics['f1']:.4f}")

        # STREAM-FraudX
        metrics = all_results[dataset_name]['stream_fraudx']
        print(f"\nSTREAM-FraudX:")
        print(f"  {'STREAM-FraudX v3':<20} ROC-AUC: {metrics['roc_auc']:.4f}  "
              f"AUPRC: {metrics['auprc']:.4f}  F1: {metrics['f1']:.4f}")

        # Check if beat XGBoost
        xgb_roc_auc = all_results[dataset_name]['baselines']['XGBoost']['roc_auc']
        stream_roc_auc = metrics['roc_auc']

        if stream_roc_auc > xgb_roc_auc:
            improvement = ((stream_roc_auc / xgb_roc_auc) - 1) * 100
            print(f"\n  ✓ Beat XGBoost by {improvement:.2f}%!")
        else:
            gap = ((xgb_roc_auc / stream_roc_auc) - 1) * 100
            print(f"\n  ✗ Below XGBoost by {gap:.2f}%")

    # Save results to JSON
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'v3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to outputs/v3_results.json")


if __name__ == '__main__':
    main()
