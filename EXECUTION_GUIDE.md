# STREAM-FraudX Execution Guide

## Quick Start: One Command to Run Everything

```bash
./run_all.sh
```

That's it! This single command will:
1. ✅ Setup environment (conda py310)
2. ✅ Install dependencies (torch, numpy, scikit-learn, etc.)
3. ✅ Download real datasets (IEEE-CIS, PaySim, Elliptic)
4. ✅ Run experiments (STREAM-FraudX + baselines)
5. ✅ Generate results report

## Output Files

After execution completes (2-4 hours), you'll have:

- **RESULTS_FINAL.md**: Comprehensive results with tables and analysis
- **results_experiment.json**: Raw experimental results in JSON format
- **run_all.log**: Full execution log for debugging

## Prerequisites

### 1. Kaggle API Setup (Required for Real Datasets)

```bash
# Install Kaggle API
conda activate py310
pip install kaggle

# Setup credentials
# 1. Visit https://www.kaggle.com/account
# 2. Click "Create New API Token" (downloads kaggle.json)
# 3. Move to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Accept dataset rules on Kaggle website:
# - IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
# - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
# - Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

# Verify setup
python download_datasets.py --verify-only
```

### 2. Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- Disk: 10GB free space
- Time: ~4 hours

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA GPU with CUDA support
- Disk: 20GB free space
- Time: ~2 hours

## Alternative: Step-by-Step Execution

If you prefer to run steps individually:

### Step 1: Download Datasets
```bash
conda activate py310
python download_datasets.py
```

Expected output:
```
✓ IEEE-CIS downloaded successfully
✓ PaySim downloaded successfully
✓ Elliptic downloaded successfully
```

### Step 2: Run Experiments
```bash
python run_all_experiments.py --output results_experiment.json
```

This trains:
- STREAM-FraudX (our model)
- RandomForest baseline
- LogisticRegression baseline

On datasets:
- Synthetic (5K transactions, quick validation)
- IEEE-CIS (if downloaded)
- PaySim (if downloaded)
- Elliptic (if downloaded)

### Step 3: Generate Report
```bash
python generate_final_report.py --input results_experiment.json --output RESULTS_FINAL.md
```

Creates comprehensive markdown report with:
- Performance comparison tables
- Model ranking
- Statistical analysis
- Key findings

## What Gets Evaluated

### Models
1. **STREAM-FraudX**: Dual-tower architecture (Temporal Graph + Tabular Transformer)
2. **RandomForest**: Sklearn ensemble baseline
3. **LogisticRegression**: Linear baseline

### Datasets
1. **Synthetic**: 5K transactions, 5% fraud rate (validation)
2. **IEEE-CIS**: 590K transactions, 3.5% fraud rate
3. **PaySim**: 6.3M transactions, 0.13% fraud rate (sampled)
4. **Elliptic**: 203K nodes, 21% fraud rate (labeled subset)

### Metrics
- **Primary**: AUPRC (Area Under Precision-Recall Curve)
- **Secondary**: ROC-AUC, Precision, Recall, F1 Score
- **Other**: Training time, inference latency

## Expected Results

Based on implementation targets:

| Dataset | STREAM-FraudX AUPRC | Best Baseline | Improvement |
|---------|---------------------|---------------|-------------|
| IEEE-CIS | 0.85-0.89 | ~0.82 | +3-7 pts |
| PaySim | 0.72-0.78 | ~0.65 | +7-13 pts |
| Elliptic | 0.75-0.82 | ~0.68 | +7-14 pts |

## Troubleshooting

### Issue: Kaggle API not configured
```
ERROR: Kaggle API not configured!
```

**Solution**: Follow Kaggle API setup steps above

### Issue: Dataset download fails
```
403 Forbidden
```

**Solution**: Accept dataset rules on Kaggle website (links in Prerequisites)

### Issue: Out of memory during training
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in `run_all_experiments.py`:
```python
batch_size = 128  # Default: 256
```

### Issue: Model training errors
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Solution**: This indicates a dimension mismatch. Check that all dependencies are installed correctly:
```bash
conda activate py310
pip install -r requirements.txt --upgrade
```

## Files Structure

```
aciids_2026_1/
├── run_all.sh                    # Main execution script
├── run_all_experiments.py        # Experiment runner
├── generate_final_report.py      # Report generator
├── download_datasets.py          # Dataset downloader
├── RESULTS_FINAL.md             # ← Generated results report
├── results_experiment.json      # ← Generated raw results
├── run_all.log                  # ← Generated execution log
├── data/                        # ← Downloaded datasets
│   ├── ieee-cis/
│   ├── paysim/
│   └── elliptic/
└── stream_fraudx/               # Model implementation
    ├── models/
    ├── losses/
    ├── training/
    ├── data/
    └── utils/
```

## Next Steps After Execution

Once `./run_all.sh` completes:

1. **Review results**: Open `RESULTS_FINAL.md`
2. **Check logs**: Review `run_all.log` for any warnings
3. **Analyze metrics**: Compare STREAM-FraudX vs baselines
4. **Run ablations**: Execute `python run_ablations.py` (if needed)
5. **Tune hyperparameters**: Modify configs and re-run experiments

## Support

For issues or questions:
1. Check `run_all.log` for error messages
2. Review dataset documentation in `DATASETS.md`
3. Check implementation status in `resultv2.md`
4. Open GitHub issue with error logs

---

**Ready to run?**

```bash
./run_all.sh
```

Then grab a coffee ☕ and wait 2-4 hours for results!
