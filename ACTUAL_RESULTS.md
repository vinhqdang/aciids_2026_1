# STREAM-FraudX v4 - ACTUAL EXPERIMENTAL RESULTS

**Date:** 2025-10-10
**Dataset:** Synthetic (5,000 samples, 5% fraud rate)
**Seed:** 42 (deterministic, reproducible)

---

## ACTUAL BASELINE RESULTS ✅

### Test Setup
- **Train samples:** 3,500 (70%)
- **Test samples:** 1,500 (30%)
- **Features:** 10 continuous + 5 categorical = 15 total
- **Class balance:** ~5% fraud (highly imbalanced)

### Random Forest Baseline
```
ROC-AUC:    0.7417
AUPRC:      0.3202
F1 Score:   0.2022
Precision:  0.4500
Recall:     0.1304
```

**Analysis:**
- Achieves **74.17% ROC-AUC** on highly imbalanced data
- AUPRC of 0.3202 is 6.4x better than random (baseline = 0.05)
- Conservative: High precision (45%), low recall (13%)

### XGBoost Baseline
```
ROC-AUC:    0.7553
AUPRC:      0.3414
F1 Score:   0.3784
Precision:  0.5000
Recall:     0.3043
```

**Analysis:**
- **Best baseline**: 75.53% ROC-AUC
- Better recall (30.43%) than Random Forest (13.04%)
- F1 score nearly 2x better (0.3784 vs 0.2022)
- More balanced precision/recall trade-off

---

## COMPARISON TABLE

| Model          | ROC-AUC | AUPRC  | F1     | Precision | Recall |
|----------------|---------|--------|--------|-----------|--------|
| Random Forest  | 0.7417  | 0.3202 | 0.2022 | 0.4500    | 0.1304 |
| **XGBoost**    | **0.7553** | **0.3414** | **0.3784** | **0.5000** | **0.3043** |

**Winner:** XGBoost achieves best performance across all metrics

---

## STREAM-FraudX v4 Neural Model

### Architecture
- **Temporal Graph Tower (v2):**
  - Recency-weighted attention
  - Hot-node caching
  - 2M+ parameters
- **Tabular Transformer Tower (v2):**
  - Feature gating
  - FT-Transformer blocks
- **FiLM Fusion (v2):**
  - Bidirectional conditioning

### Status
⏳ **Currently Training** (10 epochs, ~5-10 minutes on GPU)

Model is running with:
- Device: CUDA (GPU)
- Optimizer: AdamW (lr=1e-3)
- Loss: BCEWithLogitsLoss
- Batch size: 64

### Expected Performance
Based on architectural improvements (attention, gating, FiLM):
- **Target ROC-AUC:** 0.78-0.82 (3-8% improvement over XGBoost)
- **Target AUPRC:** 0.36-0.40

*Results will be available once training completes.*

---

## KEY FINDINGS

1. **XGBoost is strong baseline**: 75.53% ROC-AUC on synthetic data
2. **Imbalance challenge**: Only 5% fraud rate makes this difficult
3. **AUPRC more informative**: Better metric than ROC-AUC for imbalanced data
4. **Precision-recall tradeoff**: XGBoost balances better than RF

## REPRODUCIBILITY ✅

All results are **fully reproducible** with:
```bash
# Baseline results
conda run -n py310 python quick_test.py

# Neural model
conda run -n py310 python quick_neural_test.py
```

**Seeds:** All random sources set to 42
- Python `random`
- NumPy
- PyTorch (CPU + CUDA)

---

## COMPARISON TO README CLAIMS

**README claimed (v3 results, 5K sequences):**
- IEEE-CIS: CatBoost 0.7037 → STREAM-FraudX 0.7372 (+4.76%)
- PaySim: RF 0.5410 → STREAM-FraudX 0.9398 (+73.72%)

**Our v4 results (5K samples, synthetic):**
- XGBoost: 0.7553 (baseline)
- STREAM-FraudX v4: ⏳ Training...

*Note: Different datasets (synthetic vs IEEE-CIS/PaySim), so direct comparison not valid.*

---

## NEXT STEPS

1. ✅ Baseline results obtained
2. ⏳ Complete neural model training
3. 📊 Generate comparison plots
4. 📄 Update README with v4 results
5. 🚀 Test on real datasets (IEEE-CIS, PaySim)

---

**Generated:** 2025-10-10
**Framework:** STREAM-FraudX v4 (PLANv4 implementation)
