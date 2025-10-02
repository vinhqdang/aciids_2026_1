# Real Fraud Detection Datasets for STREAM-FraudX

This document describes the three real-world fraud detection datasets used in STREAM-FraudX experiments, as specified in planv1.md.

## Overview

| Dataset | Type | Size | Fraud Rate | Best For |
|---------|------|------|------------|----------|
| **IEEE-CIS** | Tabular | 590K txns | 3.5% | Rich categorical features, device/email patterns |
| **PaySim** | Temporal Stream | 6.3M txns | 0.13% | Temporal drift, streaming simulation |
| **Elliptic** | Graph | 203K nodes | 21% (labeled) | Graph structure, temporal evolution |

## 1. IEEE-CIS Fraud Detection

**Source**: [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)

### Description
Large-scale anonymized credit card fraud detection dataset from IEEE Computational Intelligence Society.

### Features
- **Transaction data** (590,540 records):
  - `TransactionDT`: Time delta from reference point
  - `TransactionAmt`: Transaction amount
  - `ProductCD`: Product code
  - `card1-6`: Card information (anonymized)
  - `addr1-2`: Address information
  - `dist1-2`: Distance features
  - `P_emaildomain`, `R_emaildomain`: Email domains
  - `C1-C14`: Counting features
  - `D1-D15`: Time delta features
  - `M1-M9`: Match features

- **Identity data** (144,233 records):
  - `id_01-38`: Identity features
  - `DeviceType`, `DeviceInfo`: Device information

### Graph Construction
Build co-occurrence graph connecting:
- Card IDs ↔ Email domains
- Card IDs ↔ Address IDs
- Device IDs ↔ Accounts

### Why Use This Dataset
✓ Rich categorical features (high-cardinality)
✓ Device/identity patterns for relational learning
✓ Real-world imbalanced data (3.5% fraud)
✓ Industry-standard benchmark

### Expected Results
- **Target**: AUPRC 0.85-0.90 (beating LightGBM baseline ~0.82)
- **Precision@100**: >0.90
- **Inference**: <20ms per transaction

---

## 2. PaySim Mobile Money Simulator

**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Description
Synthetic mobile money transaction dataset based on real data from an African mobile money service. Simulates 30 days of transactions with temporal patterns and concept drift.

### Features
- **Size**: 6,362,620 transactions
- **Time span**: 743 hours (30+ days)
- **Fraud rate**: 0.13% (8,213 fraudulent transactions)

### Transaction Types
1. `CASH-IN`: Deposit money
2. `CASH-OUT`: Withdraw money
3. `DEBIT`: Debit card payment
4. `PAYMENT`: Merchant payment
5. `TRANSFER`: P2P transfer

### Features
- `step`: Time step (hour)
- `type`: Transaction type
- `amount`: Transaction amount
- `nameOrig`: Customer ID
- `oldbalanceOrg`, `newbalanceOrig`: Balance before/after
- `nameDest`: Recipient ID
- `oldbalanceDest`, `newbalanceDest`: Recipient balance
- `isFraud`: Label
- `isFlaggedFraud`: Flagged by business rules

### Graph Structure
- **Nodes**: ~8M unique customers/merchants
- **Edges**: 6.3M transactions connecting nodes
- **Temporal**: Natural time ordering for streaming experiments

### Why Use This Dataset
✓ Temporal evolution with drift
✓ Graph structure (customer ↔ merchant)
✓ Streaming simulation (hour-by-hour)
✓ Large scale for scalability testing

### Expected Results
- **Target**: AUPRC 0.70-0.80 (highly imbalanced)
- **Drift resistance**: <2% AUPRC drop week-over-week
- **Throughput**: >1000 txn/sec

---

## 3. Elliptic Bitcoin Transaction Graph

**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### Description
Bitcoin transaction graph dataset for detecting illicit activity (money laundering, ransomware, etc.). Transactions are labeled as licit, illicit, or unknown across 49 time steps.

### Statistics
- **Nodes**: 203,769 Bitcoin transactions
- **Edges**: 234,355 payment flows
- **Time steps**: 49 (≈2 weeks per step)
- **Labels**:
  - Illicit: 4,545 (2%)
  - Licit: 42,019 (21%)
  - Unknown: 157,205 (77%)

### Features
- **166 features per node**:
  - 1 feature: Time step
  - 93 features: Local features (transaction-specific)
  - 72 features: Aggregated features (neighborhood statistics)
  - All features anonymized for privacy

### Graph Properties
- **Directed**: Edges represent Bitcoin flow
- **Temporal**: Transactions span multiple time steps
- **Sparse**: Average degree ~2.3
- **Evolving**: New nodes/edges in each time step

### Why Use This Dataset
✓ True graph structure (not constructed)
✓ Temporal evolution across 49 time steps
✓ Tests graph tower under realistic conditions
✓ Semi-supervised (77% unlabeled) → good for pretraining

### Expected Results
- **Target**: AUPRC 0.75-0.85 on labeled set
- **Graph tower benefit**: +10-15% over tabular-only
- **Pretraining benefit**: +5-8% with self-supervision

---

## Dataset Comparison

### Complementary Strengths

| Aspect | IEEE-CIS | PaySim | Elliptic |
|--------|----------|--------|----------|
| **Feature richness** | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| **Graph structure** | Constructed | Natural | Native |
| **Temporal patterns** | ★★★☆☆ | ★★★★★ | ★★★★★ |
| **Scale** | Medium | Large | Medium |
| **Label coverage** | 100% | 100% | 23% |
| **Realism** | High | Medium | High |

### Why Use All Three?

1. **IEEE-CIS**: Tests tabular transformer on rich categorical features
2. **PaySim**: Tests streaming adaptation and drift handling
3. **Elliptic**: Tests temporal graph tower on native graph structure

**Together**: Comprehensive evaluation across tabular, temporal, and graph modalities.

---

## Download Instructions

### Prerequisites

```bash
# 1. Install Kaggle API
conda run -n py310 pip install kaggle

# 2. Setup Kaggle credentials
# Visit https://www.kaggle.com/account → Create New API Token
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Accept dataset/competition rules on Kaggle website
```

### Download All Datasets

```bash
# Automated download
python download_datasets.py

# Or individually:
python -c "from stream_fraudx.data.ieee_cis_loader import download_ieee_cis; download_ieee_cis()"
python -c "from stream_fraudx.data.paysim_loader import download_paysim; download_paysim()"
python -c "from stream_fraudx.data.elliptic_loader import download_elliptic; download_elliptic()"
```

### Verify Downloads

```bash
ls -lh data/*/
# Should see:
#   data/ieee-cis/train_transaction.csv (144MB)
#   data/paysim/PS_20174392719_1491204439457_log.csv (493MB)
#   data/elliptic/elliptic_txs_features.csv (109MB)
```

---

## Data Loaders

Each dataset has a dedicated PyTorch Dataset loader:

```python
from stream_fraudx.data.ieee_cis_loader import IEEECISDataset
from stream_fraudx.data.paysim_loader import PaySimDataset
from stream_fraudx.data.elliptic_loader import EllipticDataset

# IEEE-CIS
dataset = IEEECISDataset(data_dir='data/ieee-cis', split='train', create_graph=True)

# PaySim (use fraction for faster experiments)
dataset = PaySimDataset(data_dir='data/paysim', fraction=0.1, create_graph=True)

# Elliptic (labeled only)
dataset = EllipticDataset(data_dir='data/elliptic', use_labeled_only=True)
```

All loaders return consistent format compatible with STREAM-FraudX:
```python
{
    'src_nodes': torch.Tensor,      # Graph source node
    'dst_nodes': torch.Tensor,      # Graph destination node
    'edge_attrs': torch.Tensor,     # Edge features (64-dim)
    'continuous': torch.Tensor,     # Continuous features (10-dim)
    'categorical': torch.Tensor,    # Categorical features (5-dim)
    'timestamps': torch.Tensor,     # Transaction time
    'labels': torch.Tensor,         # Fraud label (0/1)
    'time_slices': torch.Tensor     # Time bucket for IRM
}
```

---

## Experimental Protocol

### Streaming Simulation

1. **Time-ordered splits**:
   - Train: First 60% by time
   - Validation: Next 20%
   - Test: Last 20%

2. **Micro-batch processing**:
   - Process transactions in 30-60 second windows
   - Simulate label latency (1-24 hour delay)
   - Daily label budget (100-500 queries)

3. **Drift injection**:
   - Week-over-week evaluation
   - Track AUPRC degradation
   - Test adaptation speed

### Evaluation Metrics

As per planv1.md:

**Primary**:
- AUPRC (average precision)
- Precision@k (k = daily review capacity)
- F1 at calibrated threshold

**Secondary**:
- ROC-AUC
- FPR@Precision=0.9
- Detection delay
- Inference latency
- Drift resistance

---

## Expected Performance Targets

Based on planv1.md specifications:

| Dataset | Baseline AUPRC | STREAM-FraudX Target | Improvement |
|---------|---------------|---------------------|-------------|
| IEEE-CIS | 0.82 (LGBM) | 0.87-0.89 | +5-7 pts |
| PaySim | 0.65 (RF) | 0.72-0.78 | +7-13 pts |
| Elliptic | 0.68 (GCN) | 0.75-0.82 | +7-14 pts |

**Key advantages**:
1. Graph + Tabular fusion
2. Label-efficient pretraining
3. Fast drift adaptation
4. Active learning with conformal uncertainty

---

## Citation

If you use these datasets, please cite:

**IEEE-CIS**:
```bibtex
@misc{ieee-cis-fraud-detection,
  title={IEEE-CIS Fraud Detection},
  author={IEEE Computational Intelligence Society and Vesta Corporation},
  year={2019},
  url={https://www.kaggle.com/c/ieee-fraud-detection}
}
```

**PaySim**:
```bibtex
@inproceedings{lopez2016paysim,
  title={PaySim: A financial mobile money simulator for fraud detection},
  author={Lopez-Rojas, Edgar Alonso and Elmir, Ahmad and Axelsson, Stefan},
  booktitle={28th European Modeling and Simulation Symposium},
  year={2016}
}
```

**Elliptic**:
```bibtex
@article{weber2019anti,
  title={Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics},
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E},
  journal={arXiv preprint arXiv:1908.02591},
  year={2019}
}
```
