# STREAM-FraudX Improvement Plan

## Step 1 – Establish Reproducible Baselines
- Build a shared experiment driver connecting `main.py`, `run_simple_baselines.py`, and `stream_fraudx/utils/metrics.py` to log per-epoch metrics for both STREAM-FraudX and ML baselines.
- Add deterministic data splits and seed control so runs can be replicated exactly.
- Export metrics to structured artifacts (JSON/CSV) to compare learning curves beyond final scores.

## Step 2 – Enhance Data Pipeline
- Generalize the loaders in `stream_fraudx/data` into a unified interface with configurable preprocessing, normalization, and feature engineering.
- Implement dynamic graph windowing and categorical encoding caches so relational drift signals are preserved.
- Persist fitted transforms for reuse during inference and streaming adaptation.

## Step 3 – Strengthen Model Architecture
- Replace mean neighbor pooling in `TemporalGraphTower` with recency-weighted attention and add GPU-friendly caching for hot nodes.
- Upgrade the tabular tower with feature gating and FT-Transformer style attention for expressive tabular interactions.
- Extend the fusion module with residual FiLM-style modulation to better calibrate cross-tower signals.

## Step 4 – Fix Training Signals and Optimization
- Reinstate imbalance-aware objectives by swapping BCE for `CombinedFocalLoss`, restoring IRM penalties, and adding label-aware sampling.
- Integrate Stage-A pretraining checkpoints before supervised fine-tuning, and enable mixed precision, warmup/cosine scheduling, EMA, and SWA for stability.
- Introduce gradient clipping/checkpointing policies tuned for deeper towers to avoid memory spikes during longer sequences.

## Step 5 – Evaluation, Streaming, and Documentation
- Expand the streaming adaptation loop so adapters update on conformal-selected labels and track online performance drift.
- Add calibration curves, precision@k dashboards, and richer reporting in `generate_final_report.py` and related scripts.
- Revise the README with a full Stage A→B→C runbook, command snippets, and refreshed benchmarks demonstrating wins over XGBoost.
