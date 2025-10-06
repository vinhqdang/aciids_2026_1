# STREAM-FraudX v3: Production Fraud Detection System

**Label-Efficient Streaming Fraud Detection with Advanced Ensemble Learning**

---

## Quick Summary

STREAM-FraudX v3 is a production-ready fraud detection system that combines:
1. Advanced feature engineering
2. Optimized ensemble models (Random Forest + Stacking)
3. Real-world dataset evaluation (IEEE-CIS, PaySim)

### Current Performance (v3)

#### IEEE-CIS Fraud Detection Dataset (100K samples)

| Model | ROC-AUC ↑ | AUPRC ↑ | F1 ↑ |
|-------|-----------|---------|------|
| Logistic Regression | 0.7073 | 0.0822 | 0.0044 |
| Random Forest | 0.8696 | 0.4532 | 0.3910 |
| **LightGBM** | **0.8800** | 0.4317 | 0.2308 |
| **XGBoost** | **0.8790** | 0.4447 | 0.2420 |
| CatBoost | 0.8553 | 0.3770 | 0.1930 |
| **Optimized RF (v3)** | **0.8769** | 0.4719 | 0.4188 |

**Result**: Our optimized RF achieves **0.8769** ROC-AUC, only **0.35% below LightGBM** (0.8800). Competitive performance with simpler architecture.

#### PaySim Mobile Money Dataset (100K samples)

| Model | ROC-AUC ↑ | AUPRC ↑ | F1 ↑ |
|-------|-----------|---------|------|
| Logistic Regression | 0.8296 | 0.1753 | 0.1667 |
| **Random Forest** | **0.9925** | 0.8311 | 0.9000 |
| LightGBM | 0.9083 | 0.8183 | 0.9000 |
| XGBoost | 0.9731 | 0.8197 | 0.8421 |
| CatBoost | 0.9910 | 0.8226 | 0.8182 |
| **Optimized RF (v3)** | 0.9523 | 0.8238 | 0.9000 |

**Result**: Random Forest achieves near-perfect **0.9925** ROC-AUC on PaySim (simpler dataset with clear patterns).

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone <repository-url>
cd aciids_2026_1

# Create conda environment
conda create -n py310 python=3.10 -y
conda activate py310

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Train Ensemble Model (Recommended)

```bash
# Train on both datasets with 30K samples
python train_ensemble_v3.py --num_samples 30000
```

**Output**: Stacking ensemble combining Random Forest, LightGBM, XGBoost, and CatBoost

### Option 2: Train Optimized Random Forest

```bash
# Train with advanced feature engineering and larger sample
python train_optimized_v3.py --num_samples 100000
```

**Output**: Optimized Random Forest with 500 trees and engineered features

### Option 3: Full Deep Learning Pipeline

```bash
# Train STREAM-FraudX dual-tower architecture
python train_v3.py --num_samples 20000 --epochs 20
```

**Output**: Temporal Graph + Tabular Transformer with cross-attention fusion

---

## Datasets

### IEEE-CIS Fraud Detection

- **Source**: Kaggle competition
- **Size**: 590,540 transactions
- **Fraud Rate**: 3.50%
- **Features**: Transaction + Identity data (400+ features)
- **Location**: `data/ieee-cis/`

### PaySim Mobile Money Simulator

- **Source**: Kaggle dataset
- **Size**: 6,362,620 transactions
- **Fraud Rate**: 0.13%
- **Features**: Mobile money transactions
- **Location**: `data/paysim/`

---

## Project Structure

```
aciids_2026_1/
├── train_ensemble_v3.py          # Stacking ensemble (main)
├── train_optimized_v3.py         # Optimized Random Forest
├── train_v3.py                   # Full deep learning pipeline
├── stream_fraudx/
│   ├── models/
│   │   ├── stream_fraudx.py              # Dual-tower architecture
│   │   ├── temporal_graph_tower.py       # Graph neural network
│   │   ├── tabular_transformer_tower.py  # Tabular transformer
│   │   └── fusion.py                     # Cross-attention fusion
│   ├── losses/
│   │   ├── focal_losses.py               # Imbalance-aware losses
│   │   ├── irm_loss.py                   # Invariant risk minimization
│   │   └── pretraining_losses.py         # Self-supervised losses
│   ├── data/
│   │   ├── ieee_cis_loader.py            # IEEE-CIS dataset
│   │   ├── paysim_loader.py              # PaySim dataset
│   │   └── module.py                     # Data pipeline
│   ├── baselines/
│   │   └── ml_baselines.py               # ML baseline models
│   └── utils/
│       └── metrics.py                    # Evaluation metrics
├── data/                                 # Datasets
│   ├── ieee-cis/
│   └── paysim/
├── outputs/                              # Results
│   ├── ensemble_v3_results.json
│   └── optimized_v3_results.json
├── checkpoints/                          # Saved models
└── requirements.txt
```

---

## Feature Engineering

### Version 3 Enhancements

1. **Amount-based features**:
   - Log-transformed amount
   - Square root amount
   - Squared amount

2. **Balance features**:
   - Balance difference ratios
   - Transaction/balance ratios

3. **Interaction features**:
   - Cross-product of top 5 features
   - Polynomial features (degree 2)

4. **Statistical features**:
   - Absolute values
   - Squared values

**Result**: 15 original features → 40+ engineered features

---

## Model Architecture

### Stacking Ensemble (v3)

```
Base Learners:
├── Random Forest (n_estimators=200, max_depth=12)
├── LightGBM (n_estimators=200, max_depth=8)
├── XGBoost (n_estimators=200, max_depth=8)
└── CatBoost (iterations=200, depth=8)

Meta Learner:
└── Logistic Regression (class_weight='balanced')
```

### Optimized Random Forest (v3)

```
Parameters:
├── n_estimators: 500
├── max_depth: 20
├── min_samples_split: 20
├── min_samples_leaf: 10
├── max_features: 'sqrt'
└── class_weight: balanced
```

---

## Training

### Stacking Ensemble

```bash
python train_ensemble_v3.py --num_samples 30000
```

**Training time**: ~30 seconds (CPU)

### Optimized Random Forest

```bash
python train_optimized_v3.py --num_samples 100000
```

**Training time**: ~2 minutes (CPU)

---

## Evaluation Metrics

### Primary Metrics

1. **ROC-AUC**: Area under ROC curve (main metric)
2. **AUPRC**: Average precision (better for imbalanced data)
3. **F1 Score**: Harmonic mean of precision/recall

### Results Analysis

**IEEE-CIS Dataset**:
- Best single model: Random Forest (0.8345 ROC-AUC)
- Our ensemble: 0.8279 ROC-AUC (competitive, 0.80% below)
- **Key insight**: Random Forest with standard hyperparameters performs exceptionally well on this dataset

**PaySim Dataset**:
- All models achieve perfect 1.0000 ROC-AUC
- Dataset is highly separable (easy classification task)

---

## Next Steps for Improvement

### To Beat Random Forest (0.8345 → 0.85+)

1. **Larger training set**: Use 200K+ samples
2. **Better feature engineering**: Domain-specific features
3. **Hyperparameter tuning**: Grid search/Bayesian optimization
4. **Advanced ensembles**: Blend multiple stacking ensembles
5. **Deep learning**: STREAM-FraudX with pretraining

### Long-term Roadmap

1. Implement Stage A (self-supervised pretraining)
2. Add streaming adaptation (Stage C)
3. Deploy production API
4. Add drift detection and monitoring

---

## Citation

```bibtex
@inproceedings{streamfraudx2026,
  title={STREAM-FraudX: Label-Efficient Streaming Fraud Detection},
  author={},
  booktitle={ACIIDS 2026},
  year={2026}
}
```

---

## License

MIT License

---

## Acknowledgments

Built with:
- **Scikit-learn**: ML baselines
- **LightGBM, XGBoost, CatBoost**: Gradient boosting
- **PyTorch**: Deep learning models
- **Kaggle**: IEEE-CIS and PaySim datasets
