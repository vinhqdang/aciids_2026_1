#!/usr/bin/env python
"""Quick neural model test."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn
from stream_fraudx.models.stream_fraudx import STREAMFraudX
from stream_fraudx.utils.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

print("\n" + "="*80)
print("STREAM-FraudX NEURAL MODEL TEST")
print("="*80)

# Generate data
print("\n[1/4] Generating dataset...")
dataset = SyntheticFraudDataset(
    num_samples=5000,
    num_nodes=1000,
    fraud_rate=0.05,
    num_continuous=10,
    num_categorical=5
)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create model
print("\n[2/4] Creating STREAM-FraudX model...")
model = STREAMFraudX(
    continuous_dims=list(range(10)),
    categorical_vocab_sizes=[100] * 5,
    use_adapters=False
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Training
print("\n[3/4] Training (10 epochs)...")
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_val_auprc = 0
for epoch in range(1, 11):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'].float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            preds = torch.sigmoid(outputs).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(batch['labels'].cpu().numpy())

    val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
    print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Val AUPRC={val_metrics['auprc']:.4f}, Val ROC-AUC={val_metrics['roc_auc']:.4f}")

    if val_metrics['auprc'] > best_val_auprc:
        best_val_auprc = val_metrics['auprc']

# Test
print("\n[4/4] Testing...")
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(batch['labels'].cpu().numpy())

test_metrics = compute_metrics(np.array(test_labels), np.array(test_preds))

print("\n✅ STREAM-FraudX RESULTS:")
print(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
print(f"  AUPRC:      {test_metrics['auprc']:.4f}")
print(f"  F1 Score:   {test_metrics['f1']:.4f}")
print(f"  Precision:  {test_metrics['precision']:.4f}")
print(f"  Recall:     {test_metrics['recall']:.4f}")
print("="*80)
