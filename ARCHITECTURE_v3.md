# STREAM-FraudX v3: Novel Deep Learning Architecture

## Overview

STREAM-FraudX v3 is a novel deep learning architecture for fraud detection that combines **multi-scale feature extraction** and **temporal-aware attention** to capture fraud patterns at different granularities and time scales.

## Research Contributions

### 1. Temporal-Aware Attention Mechanism (`temporal_attention.py`)

**Problem**: Standard transformers treat all temporal positions equally, failing to capture the time-decay nature of fraud patterns where recent transactions are more relevant.

**Solution**: Time-decay attention that exponentially weights recent transactions higher while preserving long-range dependencies.

**Key Innovation**:
```python
attention_scores = softmax(Q @ K.T / sqrt(d) - alpha * time_decay)
```

Where `time_decay` is learned from transaction timestamps:
- Recent transactions → Lower decay → Higher attention
- Distant transactions → Higher decay → Lower attention

**Architecture Components**:
- `TemporalAttention`: Multi-head attention with learnable time-decay parameter
- `TemporalTransformerBlock`: Transformer block with temporal attention + feed-forward
- `TemporalTransformer`: Stack of temporal transformer blocks

**Benefits**:
- Emphasizes recent fraud patterns (e.g., rapid successive transactions)
- Maintains context from historical behavior
- Learns optimal time-decay schedule during training

### 2. Multi-Scale Feature Extraction (`multiscale_extractor.py`)

**Problem**: Fraud patterns exist at multiple temporal granularities:
- **Micro-scale** (3 transactions): Immediate patterns (rapid fire fraud)
- **Meso-scale** (7 transactions): Short-term behavioral patterns (testing phase)
- **Macro-scale** (15 transactions): Long-term trends (account takeover)

**Solution**: Parallel CNNs with different receptive fields + multi-timescale LSTMs

**Architecture**:

```
Input Features
    ↓
[Micro CNN (k=3)] → Immediate transaction patterns
[Meso CNN  (k=7)] → Short-term behavioral patterns
[Macro CNN (k=15)] → Long-term trends
    ↓
Concatenate & Fuse
    ↓
[Short-term LSTM] → Full sequence processing
[Long-term LSTM]  → Downsampled (2x) sequence processing
    ↓
Fuse Multi-Scale Features
```

**Key Components**:
- `MultiScaleCNN`: Three parallel 1D CNNs with kernel sizes [3, 7, 15]
- `MultiScaleLSTM`: Dual-timescale bidirectional LSTMs
- `MultiScaleExtractor`: Complete multi-scale feature extraction pipeline

**Benefits**:
- Captures patterns at multiple granularities simultaneously
- No information loss from aggressive downsampling
- Hierarchical representation learning

### 3. Complete Research Model (`stream_fraudx_research.py`)

**Full Architecture**:

```
Input (Transaction Features)
    ↓
Input Embedding (Linear + LayerNorm + GELU)
    ↓
Multi-Scale Extractor
    ├─ Multi-Scale CNN (3 parallel paths)
    └─ Multi-Scale LSTM (2 timescales)
    ↓
Temporal Transformer
    ├─ Temporal Attention (with time decay)
    └─ Feed-Forward Network
    ↓
Global Pooling (Max + Mean)
    ↓
Classification Head (3-layer MLP)
    ↓
Fraud Prediction
```

**Model Variants**:
1. `STREAMFraudXResearch`: Full sequential model with temporal features
2. `STREAMFraudXResearchSimple`: Simplified single-transaction model for baseline

**Training Details**:
- **Loss**: Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Early Stopping**: Patience of 15 epochs
- **Batch Size**: 256

## Novel Contributions for Publication

### 1. Time-Decay Attention for Financial Fraud

**Contribution**: First work to incorporate learnable temporal decay into attention mechanisms for fraud detection.

**Technical Details**:
- Learns time-decay parameter α during training
- Embeds relative time differences into attention scores
- Maintains both recency bias and long-term context

**Expected Impact**: 1-2% improvement over standard transformers

### 2. Multi-Granularity Pattern Extraction

**Contribution**: Hierarchical feature extraction at micro/meso/macro scales specifically designed for fraud patterns.

**Technical Details**:
- Parallel CNN paths with receptive fields optimized for fraud timescales
- Dual-timescale LSTM for short and long-term dependencies
- Feature fusion preserving all granularity information

**Expected Impact**: 2-3% improvement over single-scale models

### 3. Unified Deep Learning Framework

**Contribution**: End-to-end trainable architecture combining multi-scale extraction and temporal attention.

**Advantages over Traditional Methods**:
- **vs. XGBoost/LightGBM**: Captures temporal dependencies and sequential patterns
- **vs. Standard LSTMs**: Multi-scale feature extraction + attention mechanism
- **vs. Standard Transformers**: Time-aware attention with learned decay

**Expected Performance**:
- **IEEE-CIS**: Target ROC-AUC > 0.89 (vs. XGBoost 0.8790)
- **PaySim**: Target ROC-AUC > 0.98 (vs. XGBoost 0.9731)

## Implementation Details

### Feature Engineering

Advanced feature engineering applied before model training:
1. **Interaction features**: Cross-products of top 5 features
2. **Polynomial features**: Squared terms for key numerical features
3. **Log features**: Log-transformed amounts and ratios
4. **Statistical features**: Mean, std, max, min across all features

Result: 15 raw features → 40+ engineered features

### Training Pipeline (`train_research_v3.py`)

**Workflow**:
1. Load IEEE-CIS and PaySim datasets
2. Stratified sampling (to maintain fraud rate)
3. Advanced feature engineering
4. Train baseline models (LogReg, RF, LightGBM, XGBoost, CatBoost)
5. Train STREAM-FraudX Research Model
6. Compare results against best baseline

**Data Splits**:
- Training: 70%
- Validation: 15%
- Test: 15%

**Metrics**:
- **Primary**: ROC-AUC (main comparison metric)
- **Secondary**: AUPRC (better for imbalanced data)
- **Tertiary**: F1 Score (with optimal threshold tuning)

## Results (To Be Updated)

### IEEE-CIS Dataset

| Model | ROC-AUC | AUPRC | F1 Score |
|-------|---------|-------|----------|
| XGBoost (baseline) | 0.8790 | 0.4447 | 0.2420 |
| LightGBM (best) | 0.8800 | 0.4317 | 0.2308 |
| **STREAM-FraudX Research** | **TBD** | **TBD** | **TBD** |

**Target**: Beat 0.8800 (LightGBM)

### PaySim Dataset

| Model | ROC-AUC | AUPRC | F1 Score |
|-------|---------|-------|----------|
| Random Forest (best) | 0.9925 | 0.8311 | 0.9000 |
| XGBoost | 0.9731 | 0.8197 | 0.8421 |
| **STREAM-FraudX Research** | **TBD** | **TBD** | **TBD** |

**Target**: Beat 0.9925 (Random Forest)

## Future Work

### Phase 2: Self-Supervised Pretraining
- Masked transaction prediction
- Contrastive learning on unlabeled fraud data
- Expected: +2-3% improvement

### Phase 3: Advanced Techniques
- Graph Neural Networks for transaction networks
- Adversarial training for robustness
- Meta-learning for drift adaptation
- Expected: +3-5% improvement

## File Structure

```
stream_fraudx/
├── models/
│   ├── temporal_attention.py          # Novel time-decay attention
│   ├── multiscale_extractor.py        # Multi-granularity features
│   └── stream_fraudx_research.py      # Complete research model
├── data/
│   ├── ieee_cis_loader.py             # IEEE-CIS dataset loader
│   └── paysim_loader.py               # PaySim dataset loader
└── baselines/
    └── ml_baselines.py                # Baseline models wrapper

train_research_v3.py                   # Training script (main)
RESEARCH_PLAN.md                       # Research strategy
ARCHITECTURE_v3.md                     # This file
```

## Usage

### Train Research Model

```bash
# Train on 50K samples (fast iteration)
python train_research_v3.py --num_samples 50000

# Train on 100K samples (full evaluation)
python train_research_v3.py --num_samples 100000

# Use CPU instead of GPU
python train_research_v3.py --device cpu
```

### Test Novel Components

```bash
# Test temporal attention
python -m stream_fraudx.models.temporal_attention

# Test multi-scale extractor
python -m stream_fraudx.models.multiscale_extractor

# Test complete research model
python -m stream_fraudx.models.stream_fraudx_research
```

## Citation

```bibtex
@inproceedings{streamfraudx2026,
  title={STREAM-FraudX: Multi-Scale Temporal Attention for Fraud Detection},
  author={},
  booktitle={ACIIDS 2026},
  year={2026},
  note={Novel architecture combining time-decay attention and
        multi-granularity feature extraction}
}
```

## License

MIT License

## Acknowledgments

This research builds upon:
- **Temporal Attention**: Inspired by Transformer-XL and time-aware models
- **Multi-Scale CNNs**: Inspired by Inception architecture and multi-resolution analysis
- **Fraud Detection**: IEEE-CIS Kaggle competition and academic fraud detection research
