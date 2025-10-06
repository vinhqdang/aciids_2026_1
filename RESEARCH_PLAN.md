# STREAM-FraudX Research Plan: Beat XGBoost with Novel Architecture

## Goal
Create a novel deep learning architecture that beats XGBoost (0.8790 ROC-AUC) on IEEE-CIS and PaySim datasets through architectural innovation, not just hyperparameter tuning.

## Current Baseline to Beat
- **IEEE-CIS**: XGBoost 0.8790, LightGBM 0.8800
- **PaySim**: XGBoost 0.9731, Random Forest 0.9925

## Research Strategy: Novel Components

### 1. Temporal-Aware Attention Mechanism
**Problem**: Standard transformers don't capture fraud temporal patterns well
**Solution**: Time-decay attention that weights recent transactions higher

```python
attention_score = softmax(Q @ K.T / sqrt(d) - alpha * time_decay)
```

### 2. Multi-Scale Feature Learning
**Problem**: Fraud patterns exist at multiple granularities (transaction, hourly, daily)
**Solution**: Multi-scale CNN + LSTM hybrid

```
[Transaction] → Conv1D(kernel=3) → features_micro
              → Conv1D(kernel=10) → features_meso
              → Conv1D(kernel=50) → features_macro
              → Concatenate → Dense
```

### 3. Contrastive Fraud Representation Learning
**Problem**: Limited labeled fraud samples
**Solution**: Self-supervised contrastive learning on transaction embeddings

```python
# Pull fraud transactions together, push normal apart
loss_contrastive = -log(exp(sim(fraud_i, fraud_j)) / sum(exp(sim(fraud_i, all_k))))
```

### 4. Adversarial Fraud Detection
**Problem**: Fraudsters adapt, models don't
**Solution**: Train with adversarial examples

```python
# Generate adversarial perturbations
x_adv = x + epsilon * sign(grad_x(loss))
loss_total = loss(x) + lambda * loss(x_adv)
```

### 5. Graph Neural Network for Transaction Networks
**Problem**: XGBoost ignores relational structure
**Solution**: GNN on co-occurrence graph (card → merchant → email)

```python
# Message passing
h_node = aggregate([h_neighbor for neighbor in N(node)])
h_node_new = update(h_node, h_self)
```

## Implementation Plan

### Phase 1: Core Architecture (Beat XGBoost by 1-2%)
1. Temporal attention transformer
2. Multi-scale feature extraction
3. End-to-end training with focal loss

**Expected**: 0.89+ ROC-AUC

### Phase 2: Self-Supervised Pretraining (Beat XGBoost by 3-5%)
1. Contrastive learning on unlabeled data
2. Masked transaction prediction
3. Fine-tune on labeled data

**Expected**: 0.91+ ROC-AUC

### Phase 3: Advanced Techniques (Beat XGBoost by 5-10%)
1. Graph neural networks
2. Adversarial training
3. Meta-learning for drift adaptation

**Expected**: 0.93+ ROC-AUC

## Novel Contributions for Publication

1. **Temporal-Decay Attention**: Novel attention mechanism for financial time series
2. **Multi-Scale Fraud Detection**: CNN-LSTM hybrid for hierarchical patterns
3. **Contrastive Fraud Learning**: Self-supervised approach for imbalanced fraud data
4. **Adversarial Robustness**: Defense against adversarial fraud attacks
5. **Graph-Temporal Fusion**: Combining GNN + Temporal transformers

## Success Metrics

- **Primary**: Beat XGBoost ROC-AUC by ≥2% on both datasets
- **Secondary**: Novel architecture with theoretical justification
- **Tertiary**: Ablation studies showing each component's contribution

## Timeline

- Hours 1-2: Implement temporal attention transformer
- Hours 3-4: Add multi-scale feature extraction
- Hours 5-6: Train and evaluate, iterate if needed
- Hours 7-8: Self-supervised pretraining (if time permits)
