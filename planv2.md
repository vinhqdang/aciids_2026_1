# planv2.md

## Goal
Close the gap between the implementation and the STREAM-FraudX specification in `planv1.md` by delivering the missing functionality (self-supervised pretraining, streaming adaptation, active labeling, baselines) and aligning documentation and code quality.

## Workstreams

1. **Data & Pretraining Foundations**
   - Extend `stream_fraudx/data/synthetic_data.py` to emit the discrete edge attributes required by MEM (e.g., amount bins, MCC, device type) and to simulate label latency / drift windows consistent with planv1.
   - Introduce a configurable data module that can switch between synthetic stream replay and offline datasets; ensure collate function supports masked attributes and micro-batch slicing.
   - Wire up `MaskedEdgeModelingLoss` so it builds the right heads once the attribute vocabulary sizes are known.

2. **Self-Supervised Pretraining (Stage A)**
   - Implement a dedicated pretraining loop that consumes the extended data loader, leverages `PretrainingLoss`, and persists the encoder/backbone weights.
   - Provide CLI switches in `main.py` (or a new `pretrain.py`) to run Stage A separately, as well as an integrated mode that automatically pretrains before fine-tuning when checkpoints are absent.

3. **Supervised Fine-Tuning + Adapters (Stage B)**
   - Refactor `STREAMFraudXTrainer` so Stage B loads the Stage A checkpoint, freezes the backbone as needed, and optimizes only the head/adapters per planv1.
   - Integrate adapters per-layer rather than the current single-call shortcut (inject adapters inside TGT message functions and TTT blocks). Ensure meta-adapter state tracking is correct and fix the missing `Optional` import in `adapters.py`.
   - Expose loss-weight scheduling for AFL/Tversky/IRM to match paper hyperparameters.

4. **Streaming Adaptation & Meta-Learning (Stage C)**
   - Instantiate `StreamingAdaptation` inside `main.py` (or a new `serve_stream.py`) to process micro-batches, call the active learner, and perform periodic Reptile updates on the adapters.
   - Hook up `DriftDetector` to trigger adapter resets / meta updates and log drift events.
   - Simulate label latency by deferring labels for selected samples according to the daily budget constraints, updating adapters once labels arrive.

5. **Active Labeling with Conformal Prediction**
   - Connect `ConformalPredictor` calibration updates to validation/test streams and feed uncertainty scores into `ActiveLearner.select_for_labeling` during Stage C.
   - Implement a budget-aware label queue that enforces the daily cap and records which transactions were queried, matching planv1’s acquisition strategy (uncertainty × business cost × diversity).

6. **Baselines, Ablations, and Evaluation Protocol**
   - Populate `stream_fraudx/baselines/` with wrappers for LightGBM, XGBoost, CatBoost, TabTransformer, TGAT/TGN, CARE-GNN, and streaming baselines (River’s ARF/Hoeffding). Provide a unified evaluation script that replays the same data stream and logs AUPRC / Precision@k metrics.
   - Add automated ablation runners that toggle off each component listed in planv1 (graph-only, tabular-only, no adapters, etc.).
   - Ensure metrics cover calibration error and latency targets; add lightweight tests to verify metric calculations.

7. **Documentation & Tooling**
   - Update `README.md` with accurate instructions for the three-stage pipeline, streaming demo, and baseline scripts (include end-to-end commands).
   - Document adapter/meta-learning internals and data schema (e.g., `docs/architecture.md`).
   - Provide configuration files or Hydra/OmegaConf setup for experiment tracking, and outline expected outputs (checkpoints, logs).

8. **Quality & CI**
   - Add unit tests for key modules (Time2Vec, adapters, active learner budgeting) and an integration test that runs a short stream cycle.
   - Configure linters/formatters (black, isort, ruff) and set up a pytest workflow; ensure CUDA availability is handled gracefully in tests.

## Deliverables
- Updated code implementing Stages A–C with active labeling and baselines.
- Revised documentation and configs reflecting the full pipeline.
- Tests and scripts that validate core functionality and ablations.
- Changelog/notes summarizing alignment with `planv1.md`.
