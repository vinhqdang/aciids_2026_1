#!/usr/bin/env python
"""Quick test - bypass logger to get actual results fast."""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from functools import partial

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn
from stream_fraudx.baselines.ml_baselines import RandomForestBaseline, XGBoostBaseline
from stream_fraudx.utils.metrics import compute_metrics

print("\n" + "="*80)
print("ACTUAL RESULTS - STREAM-FraudX v4")
print("="*80)

# Generate data
print("\n[1/3] Generating synthetic dataset (5K samples)...")
dataset = SyntheticFraudDataset(
    num_samples=5000,
    num_nodes=1000,
    fraud_rate=0.05,
    num_continuous=10,
    num_categorical=5
)

# Split
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# Prepare data for baselines
def extract_features(dataset):
    X, y = [], []
    for sample in dataset:
        feat = np.concatenate([
            sample['continuous'].numpy(),
            sample['categorical'].numpy()
        ])
        X.append(feat)
        y.append(sample['labels'].item())
    return np.array(X), np.array(y)

X_train, y_train = extract_features(train_dataset)
X_test, y_test = extract_features(test_dataset)

# Test 1: Random Forest
print("\n[2/3] Training Random Forest...")
rf = RandomForestBaseline(n_estimators=100, max_depth=10)
rf.train(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_metrics = compute_metrics(y_test, rf_pred)

print("\n✅ RANDOM FOREST RESULTS:")
print(f"  ROC-AUC:    {rf_metrics['roc_auc']:.4f}")
print(f"  AUPRC:      {rf_metrics['auprc']:.4f}")
print(f"  F1 Score:   {rf_metrics['f1']:.4f}")
print(f"  Precision:  {rf_metrics['precision']:.4f}")
print(f"  Recall:     {rf_metrics['recall']:.4f}")

# Test 2: XGBoost
print("\n[3/3] Training XGBoost...")
xgb = XGBoostBaseline(n_estimators=100, max_depth=6)
xgb.train(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_metrics = compute_metrics(y_test, xgb_pred)

print("\n✅ XGBOOST RESULTS:")
print(f"  ROC-AUC:    {xgb_metrics['roc_auc']:.4f}")
print(f"  AUPRC:      {xgb_metrics['auprc']:.4f}")
print(f"  F1 Score:   {xgb_metrics['f1']:.4f}")
print(f"  Precision:  {xgb_metrics['precision']:.4f}")
print(f"  Recall:     {xgb_metrics['recall']:.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"\n{'Model':<20} {'ROC-AUC':<12} {'AUPRC':<12} {'F1':<12}")
print("-" * 80)
print(f"{'Random Forest':<20} {rf_metrics['roc_auc']:<12.4f} {rf_metrics['auprc']:<12.4f} {rf_metrics['f1']:<12.4f}")
print(f"{'XGBoost':<20} {xgb_metrics['roc_auc']:<12.4f} {xgb_metrics['auprc']:<12.4f} {xgb_metrics['f1']:<12.4f}")
print("="*80)
