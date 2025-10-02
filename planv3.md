# planv3.md

## Goal
Deliver a production-ready STREAM-FraudX pipeline that fully satisfies planv2 requirements, fixes current blockers (training crash, missing streaming integration), and produces reproducible final reports for both synthetic and real datasets.

## Current Gaps (from planv2 + repo audit)
- Stage A data module lacks label-latency simulation, micro-batch slicing, and configurable backends for real datasets.
- Stage B fine-tuning ignores pretrained checkpoints, keeps the full backbone trainable, and still triggers the Tversky/IRM double-backward failure.
- Stage C streaming adaptation and active learner utilities are not wired into any executable entry point.
- Baseline/evaluation suite missing gradient-boosting, deep, and streaming baselines; no ablation runners.
- Documentation, final report generation, and README instructions are stale; no architecture docs or troubleshooting notes.
- No automated validation (tests/linters), and requirements/conda steps are not enforced via scripts.
- `final_result.md` (promised by run scripts) is absent; only `resultv2.md` exists and is outdated.

## Workstreams & Key Tasks

### 1. Data & Pretraining Foundations
- Implement modular data module (`stream_fraudx/data/module.py`) that can switch between synthetic replay and real loaders with micro-batch emitting and label-delay simulation.
- Extend synthetic generator to support configurable drift windows, latency queues, and masked-attribute vocab export.
- Add collate utilities for micro-batch slicing and MEM target vocab sizes; ensure CUDA-safe dtype handling.
- Harden `pretrain.py` to accept dataset configs, save encoder-only checkpoint, and log MEM coverage statistics.

### 2. Stage B Fine-Tuning & Gradient Fix
- Add checkpoint loader that freezes backbone and trains adapters/head when Stage A weights exist; expose CLI switches.
- Debug Tversky/IRM backward failure (sanity test with BCE, detach pattern, or rework loss to avoid global state); add targeted unit test covering the fix.
- Restructure trainer to support distinct Stage B schedule (loss weighting, optimizer groups) and record validation curves.

### 3. Stage C Streaming Adaptation & Active Labeling
- Create `serve_stream.py` (or extend `main.py`) to instantiate `StreamingAdaptation`, `ActiveLearner`, `DriftDetector`, and run micro-batch loop with simulated latency.
- Implement label queue / budget enforcement and log drift/reset events.
- Persist meta-adapter checkpoints and provide resume hooks.
- Add smoke test that replays a short synthetic stream end-to-end.

### 4. Baselines, Ablations, and Metrics Expansion
- Implement gradient-boosting baselines (LightGBM, XGBoost, CatBoost), deep baselines (TabTransformer, TGAT/TGN), and streaming baselines (River ARF/Hoeffding).
- Build unified evaluation harness that replays identical data streams, computes calibration metrics, latency, and precision@k.
- Add ablation runner (`run_ablations.py`) toggling graph/tabular/fusion/adapters/pretraining/active-learning components.

### 5. Documentation, Reporting, and Artifacts
- Update README with accurate Stage A→C instructions, environment setup, and single-command pipelines.
- Create `docs/architecture.md`, `docs/troubleshooting.md`, and dataset notes covering schema, latency simulation, and adapter internals.
- Ensure pipeline scripts generate `final_result.md` (or rename consistently) summarizing experiments; keep `resultv2.md` as historical snapshot.
- Automate final report generation to include baseline comparisons, drift logs, and calibration metrics.

### 6. Tooling, QA, and Release Prep
- Add pytest suites for losses, adapters, active learner, and streaming loop; include minimal synthetic fixtures.
- Configure linters/formatters (black, isort, ruff) plus CI script (GitHub Actions or local). Ensure CUDA optionality is respected.
- Pin requirements/conda environment (environment.yml) and add sanity scripts for dependency checks.
- Prepare release checklist: regenerated checkpoints, logs, and updated `CHANGELOG.md` summarizing planv3 completion.

## Milestones
1. **M1 – Foundations Ready (Data + Stage A)**: Data module, latency simulation, hardened pretraining, encoder checkpoint saved.
2. **M2 – Stable Stage B**: Training loop runs end-to-end with adapters + IRM/Tversky fix, CLI supports checkpoint warm-start.
3. **M3 – Streaming MVP**: Streaming adaptation executable runs synthetic stream with active labeling, produces logs and adapter checkpoints.
4. **M4 – Evaluation Suite**: Baselines/ablations produce metrics + final report, regression scripts in place.
5. **M5 – Documentation & QA**: README/docs refreshed, tests/linters passing, final artifacts (`final_result.md`, checkpoints) published.

## Deliverables
- `stream_fraudx/data/module.py`, updated synthetic loader, and config-driven datasets.
- Updated trainers/scripts (`main.py`, `pretrain.py`, `serve_stream.py`) with Stage A→C orchestration.
- Baseline modules and ablation runners with recorded metrics.
- `final_result.md` (generated) + refreshed README and docs.
- Test suite + CI script demonstrating pipeline health.
- Release notes capturing changes vs planv2.
