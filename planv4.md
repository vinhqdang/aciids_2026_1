# STREAM-FraudX Recovery & Publication Plan (v4)

## Guiding Principles
- **Single source of truth** for training/evaluation to guarantee reproducibility.
- **Modular upgrades** that unlock PLANv3 objectives without rewriting the entire stack.
- **Evidence-first reporting**: every architectural claim must be backed by logged experiments and curated artifacts.

## Workstream 1 – Rebuild the Experiment Driver
1. Consolidate `main.py`, `run_simple_baselines.py`, and `run_all_experiments.py` into a unified runner (e.g., `experiments/driver.py`).
2. Enforce deterministic seeds (PyTorch, NumPy, Python RNG, DataLoader workers) and record them in run metadata.
3. Implement a shared `ExperimentLogger` that captures per-epoch metrics, hyper-parameters, wall-clock timings, and artifact paths for **both** neural models and classical baselines.
4. Emit standardized JSON + CSV outputs (`artifacts/runs/<run_id>/metrics.{json,csv}`) and wire the CLI to resume/inspect past runs.

## Workstream 2 – Modernize the Data Pipeline
1. Refactor `stream_fraudx/data` into configurable loaders with composable preprocessing steps (scaling, normalization, feature engineering).
2. Add categorical/continuous encoder registries that persist fitted transforms (`artifacts/preprocessing/*.pt`) for reuse during inference/streaming.
3. Implement dynamic graph windowing & caching: window sizes driven by config, plus reusable categorical encoding caches to preserve drift signals.
4. Expand dataset configs to include schema metadata, so the runner can infer feature dims automatically.

## Workstream 3 – Deliver the Planned Architecture Upgrades
1. Replace mean neighbor pooling in `TemporalGraphTower` with recency-weighted attention plus a GPU-friendly hot-node cache.
2. Upgrade the tabular tower to include feature gating and FT-Transformer-style attention blocks; expose toggles for ablations.
3. Extend fusion with residual FiLM-style modulation (conditioning tabular signals on graph context and vice versa).
4. Update model configuration objects so the new components can be enabled/disabled from the experiment driver.

## Workstream 4 – Fix Training Signals & Optimization Stack
1. Reinstate `CombinedFocalLoss`, IRM penalties, and label-aware sampling inside the unified trainer.
2. Integrate Stage-A pretraining checkpoints before supervised fine-tuning, with hooks for adapter-only updates.
3. Enable AMP (`torch.cuda.amp`), warmup + cosine scheduling, EMA, SWA, gradient clipping, and gradient checkpointing where memory helps.
4. Make the trainer serializable: checkpoints must capture optimizer, schedulers, EMA/SWA states, and Stage info.

## Workstream 5 – Streaming, Evaluation, and Reporting
1. Repair Stage-C streaming adaptation: fix the meta-adapter loss interface, ensure conformal selection works with the new logger, and track online drift metrics.
2. Augment reporting with calibration curves, precision@k dashboards, and rolling streaming metrics; store plots under `artifacts/reports/`.
3. Update `generate_final_report.py` to compile all structured artifacts and comparative tables automatically.
4. Rewrite the README with a Stage A→B→C runbook, single-command launcher examples, dependency setup (conda + CUDA), and refreshed benchmarks demonstrating wins over XGBoost.

## Workstream 6 – Publication Readiness
1. Curate a “Top Conference” experiment suite (datasets, seeds, configs) and document it in `docs/experiments.md`.
2. Produce an implementation report summarizing architectural changes, optimization gains, and streaming performance improvements (linking to artifacts).
3. Establish a pre-submission checklist (code style, tests, reproducibility scripts) and integrate it into CI before tagging the release branch.

## Immediate Next Steps (Sprint 1)
1. Scaffold the new experiment driver + logger and migrate at least one neural run and one baseline run onto it.
2. Draft the data-loader config schema and implement transform persistence for the synthetic dataset as a proof of concept.
3. Patch the streaming adaptation loss call so Stage C executes end-to-end with the current model (even before architectural upgrades).

Deliverables from Sprint 1 will unblock the remaining PLANv3 commitments while giving us observable progress toward publication readiness.
