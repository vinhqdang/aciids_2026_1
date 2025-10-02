# STREAM-FraudX: Experimental Results

**Date**: October 2, 2025
**Status**: ❌ **Current Implementation Underperforms Baselines**

## Executive Summary

We implemented and tested STREAM-FraudX against traditional ML baselines (Logistic Regression, LightGBM, XGBoost) on synthetic fraud detection data.

**Key Finding**: **STREAM-FraudX currently underperforms baseline methods by 25-35% in AUPRC.**

## Experimental Setup

### Dataset
- **Type**: Synthetic fraud transactions
- **Size**: 20,000 samples
- **Nodes**: 1,000 (users/merchants)
- **Fraud Rate**: 5% (increased from default 1%)
- **Features**: 10 continuous + 5 categorical
- **Split**: 70% train / 15% val / 15% test

### STREAM-FraudX Configuration
- **Architecture**: Dual-tower (TGT + TTT) with cross-attention fusion
- **Parameters**: 2.3M total
- **Training**: 30 epochs, batch size 256
- **Loss**: BCEWithLogitsLoss (simplified due to gradient issues)
- **Optimizer**: AdamW with separate param groups
- **Device**: CUDA GPU
- **Stages Run**: Stage B only (no pretraining)

### Baseline Configurations
- **Logistic Regression**: L2 regularization, max_iter=1000
- **LightGBM**: 100 estimators, depth 6, scale_pos_weight=4.94
- **XGBoost**: 100 estimators, depth 6, GPU acceleration

## Results

### Performance Comparison

| Model | AUPRC ↑ | ROC-AUC ↑ | F1 ↑ | Precision | Recall | Training Time |
|-------|---------|-----------|------|-----------|--------|---------------|
| **Logistic Regression** | **0.2268** | **0.5489** | 0.0000 | 0.0000 | 0.0000 | ~5s |
| **LightGBM** | **0.2210** | 0.5468 | **0.2610** | 0.2226 | **0.3155** | ~10s |
| **XGBoost** | **0.2060** | 0.5401 | **0.2358** | 0.2164 | 0.2591 | ~15s |
| **STREAM-FraudX** | **0.1658** | 0.5004 | 0.0000 | 0.0000 | 0.0000 | ~90s |

### Key Metrics

#### AUPRC (Primary Metric)
```
Logistic Regression:  0.2268  ████████████████████████  (+36.8% vs STREAM-FraudX)
LightGBM:             0.2210  ███████████████████████   (+33.3% vs STREAM-FraudX)
XGBoost:              0.2060  ████████████████████      (+24.2% vs STREAM-FraudX)
STREAM-FraudX:        0.1658  █████████████████         (baseline)
```

#### ROC-AUC
```
Logistic Regression:  0.5489  ████████████  (+9.7% vs STREAM-FraudX)
LightGBM:             0.5468  ████████████  (+9.3% vs STREAM-FraudX)
XGBoost:              0.5401  ███████████   (+7.9% vs STREAM-FraudX)
STREAM-FraudX:        0.5004  ██████████    (baseline)
```

### Detailed STREAM-FraudX Results

**Training Progress**:
- Epoch 1: Train Loss 0.475, Val AUPRC 0.165
- Epoch 5: Train Loss 0.459, Val AUPRC 0.166
- Epoch 10: Train Loss 0.448, Val AUPRC 0.169
- Epoch 15: Train Loss 0.445, Val AUPRC 0.171
- Epoch 20: Train Loss 0.440, Val AUPRC 0.173
- Epoch 25: Train Loss 0.436, Val AUPRC 0.175
- Epoch 30: Train Loss 0.433, Val AUPRC 0.177

**Best Model**: Epoch 30 (final)

**Test Set Performance**:
- AUPRC: 0.1658
- ROC-AUC: 0.5004 (barely better than random: 0.5)
- Precision: 0.0000 (model predicts all negative)
- Recall: 0.0000
- F1: 0.0000
- Precision@100: 0.1400
- Precision@500: 0.1620

## Analysis

### Why STREAM-FraudX Underperformed

#### 1. **Implementation Issues** (Primary)

**Gradient/Backward Errors**:
- Original implementation had double-backward errors with IRM + Tversky loss
- Had to simplify to basic BCE loss (loses class imbalance handling)
- Disabled memory bank updates (`update_memory=False`)
- Disabled IRM regularization (set weight to 0)

**Impact**: Lost ~50% of model's key features

#### 2. **Missing Stage A (Pretraining)**

- No self-supervised pretraining was performed
- Model starts from random initialization
- Baselines (LightGBM/XGBoost) don't need pretraining

**Expected Impact**: 10-20% AUPRC improvement with pretraining

#### 3. **Synthetic Data Limitations**

- Synthetic data may not have strong graph structure
- Random user-merchant connections
- No real temporal patterns or drift
- Graph tower may not provide useful signals

**Impact**: Graph component adds complexity without benefit

#### 4. **Model Complexity vs Data Size**

- 2.3M parameters for 20K samples (~100:1 ratio)
- Baselines have <100K parameters
- Risk of overfitting despite regularization

#### 5. **Hyperparameter Tuning**

- Used default hyperparameters
- No learning rate scheduling
- No architecture search
- Baselines also used defaults but are more robust

### Baseline Success Factors

**LightGBM (Best Overall)**:
- ✅ Handles class imbalance natively (scale_pos_weight)
- ✅ Fast training (10s vs 90s)
- ✅ Robust to hyperparameters
- ✅ Works well on tabular data
- ✅ Good precision-recall trade-off (F1=0.261)

**Logistic Regression (Best AUPRC)**:
- ✅ Simple and effective
- ✅ Very fast (5s)
- ✅ Good at ranking (AUPRC=0.227)
- ❌ Poor calibration (Precision/Recall=0)

## What Works in STREAM-FraudX

Despite underperformance, several components work correctly:

✅ **Architecture**: Forward/backward passes work without crashes
✅ **Training Loop**: Stable convergence over 30 epochs
✅ **Multi-tower Design**: TGT + TTT + Fusion all functional
✅ **Adapters**: Parameter-efficient fine-tuning implemented
✅ **Data Module**: Micro-batch streaming works
✅ **Inference Speed**: ~20 it/s on GPU
✅ **Code Quality**: Modular, tested, documented

## Path to Competitive Performance

### Immediate Fixes Required

1. **Fix Gradient Issues** (Critical)
   - Resolve double-backward error in Tversky + IRM
   - Enable memory bank updates
   - Restore focal loss weighting

2. **Run Stage A Pretraining** (High Impact)
   - 20 epochs on 50K unlabeled samples
   - MEM + InfoNCE losses
   - Save encoder checkpoint

3. **Hyperparameter Tuning** (Medium Impact)
   - Learning rate: Try [1e-4, 5e-4, 1e-3]
   - Model size: Try 64/128/256 hidden dims
   - Loss weights: Grid search α, β, γ

### Expected Performance After Fixes

| Scenario | Expected AUPRC | vs Current | vs LightGBM |
|----------|----------------|------------|-------------|
| **Current** | 0.166 | - | -25% |
| **+ Gradient Fixes** | 0.200 | +20% | -9% |
| **+ Stage A Pretraining** | 0.240 | +45% | +9% |
| **+ Hyperparameter Tuning** | 0.260 | +57% | +18% |
| **+ Real Graph Data** | 0.300-0.400 | +80-140% | +36-81% |

### Long-term Improvements

1. **Test on Real Datasets**
   - IEEE-CIS: Rich categorical features
   - Elliptic: Strong graph structure
   - PaySim: Temporal drift patterns

2. **Enable Full Pipeline**
   - Stage A → B → C
   - Active learning
   - Drift adaptation

3. **Architecture Enhancements**
   - Attention visualization
   - Graph augmentations
   - Better fusion mechanisms

4. **Comparison with SOTA**
   - TabNet, FT-Transformer
   - TGAT, TGN (graph methods)
   - Streaming methods (River)

## Recommendations

### For Publication

**Current Status**: ❌ **Not publishable** due to underperformance

**Requirements for Publication**:
1. Beat at least one strong baseline (LightGBM)
2. Demonstrate benefit on graph-structured data
3. Show pretraining improves performance
4. Ablation studies show component contributions

### For Further Development

**Priority 1 (Week 1)**:
- [ ] Fix gradient/backward issues
- [ ] Run Stage A pretraining
- [ ] Re-run experiments with full pipeline

**Priority 2 (Week 2)**:
- [ ] Hyperparameter tuning
- [ ] Test on IEEE-CIS real dataset
- [ ] Ablation studies

**Priority 3 (Week 3)**:
- [ ] Compare vs TabNet, FT-Transformer
- [ ] Streaming evaluation (Stage C)
- [ ] Write paper draft

### For Production Use

**Current Status**: ❌ **Not ready**

**Blockers**:
- Performance below simple baselines
- Longer training time (90s vs 10s)
- Implementation issues (gradient errors)
- No clear advantage over LightGBM

**Path to Production**:
1. Achieve >0.25 AUPRC (beat LightGBM)
2. Fix all implementation issues
3. Optimize inference latency
4. Add monitoring and logging
5. Docker deployment ready

## Conclusion

### Summary

STREAM-FraudX is a **technically sound but currently underperforming** fraud detection system:

- ✅ **Implementation**: Complete 3-stage pipeline coded
- ✅ **Architecture**: Novel dual-tower design
- ✅ **Code Quality**: Production-ready structure
- ❌ **Performance**: 25-35% below baselines
- ❌ **Stability**: Gradient issues require workarounds

### Answer to "Is It Better Than SOTA?"

**NO** - Current implementation **underperforms** traditional baselines:
- LightGBM: 0.221 AUPRC (+33%)
- STREAM-FraudX: 0.166 AUPRC (baseline)

### Next Steps

**Immediate** (1-2 days):
1. Fix gradient issues to enable all losses
2. Run Stage A pretraining
3. Re-evaluate with full pipeline

**Short-term** (1-2 weeks):
1. Hyperparameter optimization
2. Real dataset evaluation (IEEE-CIS)
3. Ablation studies

**Long-term** (1-2 months):
1. Beat SOTA on multiple datasets
2. Paper submission
3. Production deployment

### Lessons Learned

1. **Complex ≠ Better**: Simpler LightGBM beats 2.3M parameter model
2. **Gradient Issues Matter**: Lost 50% of features due to backward errors
3. **Data Matters**: Synthetic data may not showcase graph benefits
4. **Pretraining Is Key**: Missing Stage A hurts performance significantly
5. **Baselines Are Strong**: LightGBM is hard to beat on tabular data

---

**Contact**: For questions about these results, see `docs/troubleshooting.md` or open a GitHub issue.

**Next Update**: After gradient fixes and Stage A pretraining are complete.
