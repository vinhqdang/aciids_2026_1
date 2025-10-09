# STREAM-FraudX Experiment Documentation

## Overview

This document describes the complete experimental setup for STREAM-FraudX, designed for reproducible, publication-quality fraud detection research.

## Experiment Framework

### Unified Experiment Driver

Location: `experiments/driver.py`

The experiment driver consolidates all previous runner scripts (main.py, run_simple_baselines.py, run_all_experiments.py) into a single, unified framework.

**Features:**
- Deterministic seeding for reproducibility
- Structured JSON/CSV logging
- Support for both neural models and classical baselines
- Automatic checkpointing and resuming
- Comprehensive metrics tracking

### Configuration System

Location: `experiments/config.py`

**Configuration Components:**
- `DataConfig`: Data loading, preprocessing, batch size
- `ModelConfig`: Architecture hyperparameters
- `TrainingConfig`: Optimization, scheduling, early stopping
- `PretrainingConfig`: Stage-A self-supervised pretraining
- `StreamingConfig`: Stage-C online adaptation

**Example:**
```python
from experiments.config import ExperimentConfig

config = ExperimentConfig(
    experiment_name="streamfraudx_baseline",
    seed=42,
    device="cuda"
)

config.data.dataset_name = "ieee-cis"
config.data.num_samples = 50000
config.training.max_epochs = 30
config.training.learning_rate = 1e-3
```

## Experiment Stages

### Stage A: Self-Supervised Pretraining (Optional)

**Goal:** Learn robust feature representations without labels

**Tasks:**
- Contrastive learning on transaction sequences
- Temporal ordering prediction
- Masked feature reconstruction

**Usage:**
```bash
python -m experiments.driver \
    --experiment_name "stage_a_pretrain" \
    --model_type "stream_fraudx" \
    --pretraining_enabled \
    --pretrain_epochs 50
```

### Stage B: Supervised Training (Main)

**Goal:** Train fraud detection model with labeled data

**Supported Models:**
1. **Neural Models:**
   - STREAM-FraudX (v1): Baseline dual-tower architecture
   - STREAM-FraudX (v2): Enhanced with recency-weighted attention, feature gating, FiLM fusion

2. **Classical Baselines:**
   - Random Forest
   - Logistic Regression
   - LightGBM
   - XGBoost
   - CatBoost

**Usage:**
```bash
# Neural model
python -m experiments.driver \
    --experiment_name "stage_b_streamfraudx" \
    --model_type "stream_fraudx" \
    --dataset "synthetic" \
    --num_samples 20000 \
    --epochs 30

# Baseline model
python -m experiments.driver \
    --experiment_name "stage_b_xgboost" \
    --model_type "xgboost" \
    --dataset "synthetic" \
    --num_samples 20000
```

### Stage C: Streaming Adaptation (Future)

**Goal:** Adapt to distributional drift in production

**Features:**
- Meta-learning for fast adaptation
- Conformal prediction for uncertainty
- Online drift detection

## Datasets

### 1. Synthetic Dataset

**Purpose:** Fast prototyping and debugging

**Characteristics:**
- Configurable size
- Controllable fraud rate (default: 5%)
- Graph structure with temporal edges
- Mixed continuous/categorical features

**Usage:**
```python
config.data.dataset_name = "synthetic"
config.data.num_samples = 10000
```

### 2. IEEE-CIS Fraud Detection

**Source:** Kaggle competition
**Size:** 590,540 transactions
**Fraud Rate:** 3.5%
**Features:** 400+ transaction and identity features

**Location:** `data/ieee-cis/`

### 3. PaySim Mobile Money

**Source:** Kaggle dataset
**Size:** 6.3M transactions
**Fraud Rate:** 0.13%
**Features:** Mobile money transactions

**Location:** `data/paysim/`

### 4. Elliptic Bitcoin Transactions

**Source:** Kaggle dataset
**Graph:** Bitcoin transaction network
**Task:** Illicit transaction detection

**Location:** `data/elliptic/`

## Metrics and Evaluation

### Primary Metrics

1. **ROC-AUC**: Area under ROC curve (main metric)
   - Best for overall ranking performance
   - Threshold-independent

2. **AUPRC**: Average precision
   - Better for highly imbalanced data
   - Emphasizes minority class performance

3. **F1 Score**: Harmonic mean of precision/recall
   - Balanced measure at optimal threshold

### Additional Metrics

- **Precision@k**: Precision in top-k predictions
- **Recall@k**: Recall in top-k predictions
- **Calibration Error**: Expected Calibration Error (ECE)
- **Confusion Matrix**: TP, FP, TN, FN

### Logging

All metrics are logged to:
- **JSON**: `artifacts/runs/<run_id>/metrics.json`
- **CSV**: `artifacts/runs/<run_id>/metrics.csv`

View results:
```python
from experiments.logger import ExperimentLogger

logger = ExperimentLogger.load_run("run_id_here")
print(logger.summary())
```

## Reproducibility

### Seeds

All random seeds are set deterministically:
- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed`
- CUDA `torch.cuda.manual_seed_all`
- DataLoader workers

### Environment

Conda environment: `py310`

Install:
```bash
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
```

### Configuration Files

Save configuration for exact reproducibility:
```python
config.save("artifacts/runs/<run_id>/config.yaml")
```

Load and rerun:
```python
config = ExperimentConfig.load("artifacts/runs/<run_id>/config.yaml")
```

## Running Experiments

### Single Command

Run complete pipeline:
```bash
./run_all.sh
```

### Custom Experiments

**Baseline Comparison:**
```bash
# Test all baselines
for model in random_forest lightgbm xgboost catboost; do
    python -m experiments.driver \
        --experiment_name "baseline_${model}" \
        --model_type "${model}" \
        --dataset "synthetic" \
        --num_samples 10000 \
        --seed 42
done
```

**Hyperparameter Sweep:**
```bash
# Test different learning rates
for lr in 0.0001 0.001 0.01; do
    python -m experiments.driver \
        --experiment_name "streamfraudx_lr_${lr}" \
        --model_type "stream_fraudx" \
        --lr "${lr}" \
        --epochs 30
done
```

### Experiment Tracking

List all runs:
```python
from experiments.logger import ExperimentLogger
runs = ExperimentLogger.list_runs()
print(runs)
```

Load and compare:
```python
run1 = ExperimentLogger.load_run("run_20250109_120000")
run2 = ExperimentLogger.load_run("run_20250109_130000")

print(f"Run 1 AUPRC: {run1.get_best_metric('val_auprc', 'max')}")
print(f"Run 2 AUPRC: {run2.get_best_metric('val_auprc', 'max')}")
```

## Best Practices

1. **Always set a seed** for reproducibility
2. **Log everything** - hyperparameters, metrics, artifacts
3. **Save checkpoints** regularly during training
4. **Version your data** - document any preprocessing
5. **Document experiments** - keep notes on what you tried
6. **Compare to baselines** - always run simple baselines first
7. **Check for overfitting** - monitor train/val gap
8. **Validate on multiple seeds** - run with 3-5 random seeds

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python -m experiments.driver --batch_size 32
```

Enable gradient checkpointing:
```python
config.training.use_grad_checkpointing = True
```

### Poor Performance

1. Check class balance
2. Try label-aware sampling
3. Increase model capacity
4. Add more training epochs
5. Tune learning rate

### Slow Training

1. Enable AMP (mixed precision)
2. Increase batch size
3. Use fewer workers
4. Profile with PyTorch profiler

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
