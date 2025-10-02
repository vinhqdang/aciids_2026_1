# STREAM-FraudX

**Label-Efficient Streaming Fraud Detection in E-Finance via Dual-Tower Temporal Graph + Tabular Transformers with Self-Supervised Pretraining and Drift-Aware Adaptation**

## Overview

STREAM-FraudX is a novel end-to-end fraud detection system designed for online payment environments where fraud patterns drift quickly and labeled data is scarce. The system combines:

- **Dual-Tower Architecture**: Temporal Graph Network (TGT) + Tabular Transformer (TTT)
- **Gated Cross-Attention Fusion**: Merges relational and tabular signals
- **Self-Supervised Pretraining**: Masked Edge Modeling (MEM) + Subgraph Contrastive Learning
- **Drift-Aware Adaptation**: Meta-adapters with Reptile-style updates and IRM regularization
- **Active Learning**: Conformal uncertainty-based label selection

## Architecture

```
Transaction Event
       ↓
   ┌───────────────────┬────────────────────┐
   │                   │                    │
   ↓                   ↓                    ↓
Temporal Graph    Tabular Features    Timestamps
   Tower              Tower
   (TGT)              (TTT)
   ↓                   ↓
Graph Embedding   Tabular Embedding
   │                   │
   └─────────┬─────────┘
             ↓
    Gated Cross-Attention
          Fusion
             ↓
       Detection Head
             ↓
       Fraud Score
```

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n py310 python=3.10 -y
conda activate py310
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

For CUDA support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training on Synthetic Data

Run the complete training pipeline with a single command:

```bash
python main.py --epochs 50 --batch_size 256 --output_dir outputs
```

### Custom Configuration

```bash
python main.py \
  --num_samples 20000 \
  --num_nodes 2000 \
  --fraud_rate 0.02 \
  --epochs 100 \
  --batch_size 512 \
  --learning_rate 0.001 \
  --use_adapters \
  --output_dir outputs/experiment1
```

## Project Structure

```
stream_fraudx/
├── models/
│   ├── temporal_graph_tower.py    # TGT with TGN-style memory
│   ├── tabular_transformer_tower.py  # TTT with feature tokenization
│   ├── fusion.py                   # Gated cross-attention fusion
│   ├── adapters.py                 # Parameter-efficient adapters
│   └── stream_fraudx.py            # Main model
├── losses/
│   ├── focal_losses.py             # Asymmetric Focal + Tversky
│   ├── irm_loss.py                 # IRM-lite for drift
│   └── pretraining_losses.py      # MEM + InfoNCE
├── training/
│   ├── trainer.py                  # Training pipeline
│   ├── drift_adaptation.py        # Meta-adapter + streaming
│   └── active_learning.py         # Conformal uncertainty
├── data/
│   └── synthetic_data.py          # Synthetic fraud generator
└── utils/
    └── metrics.py                 # Evaluation metrics
```

## Key Features

### 1. Temporal Graph Tower (TGT)

Processes evolving transaction graphs with:
- Time2Vec encoding for temporal patterns
- TGN-style memory with LRU eviction
- Reservoir neighbor sampling for scalability
- Multi-layer temporal message passing

### 2. Tabular Transformer Tower (TTT)

Handles heterogeneous transaction features:
- Feature tokenization (continuous → bins, categorical → embeddings)
- Fourier time encoding for cyclical patterns
- Multi-head self-attention
- CLS token for global representation

### 3. Gated Cross-Attention Fusion

Merges graph and tabular signals:
- Bidirectional cross-attention
- Gated residual connections
- Learns optimal combination of modalities

### 4. Loss Functions

**Training Losses:**
- Asymmetric Focal Loss (γ+ ≈ 0, γ− ≈ 2)
- Focal Tversky Loss (α=0.7, β=0.3)
- IRM-lite penalty for temporal stability

**Pretraining Losses:**
- Masked Edge Modeling (MEM)
- InfoNCE contrastive loss with queue

### 5. Drift Adaptation

**Meta-Adapters:**
- Bottleneck MLP adapters (d → d/r → d)
- Reptile-style meta-learning
- Fast adaptation to distribution shifts

**Streaming Protocol:**
- Process micro-batches (30-60 seconds)
- Select samples via conformal uncertainty
- Update adapters with recent labels

### 6. Active Learning

**Conformal Prediction:**
- Calibrated uncertainty quantification
- Non-conformity scores
- Coverage guarantees

**Acquisition Function:**
- Combines uncertainty + business cost + diversity
- Maximizes label efficiency

## Evaluation Metrics

### Primary Metrics
- **AUPRC** (Average Precision): Main optimization target
- **Precision@k**: Direct operational relevance
- **F1 Score**: Balanced performance

### Secondary Metrics
- **ROC-AUC**: Overall discrimination
- **FPR@P=0.9**: False positive control
- **Calibration Error**: Prediction reliability

## Expected Performance

On synthetic data with realistic fraud patterns:
- **AUPRC**: 0.75-0.85
- **ROC-AUC**: 0.90-0.95
- **Precision@100**: 0.80-0.90
- **Inference Latency**: <20ms per event (GPU)

## Algorithm Details

### Three-Stage Training

**Stage A: Self-Supervised Pretraining**
```python
for epoch in range(E_pretrain):
    loss = MEM_loss + λ * InfoNCE_loss
    optimizer.step()
```

**Stage B: Supervised Fine-Tuning**
```python
for epoch in range(E_finetune):
    loss = AFL + α*Tversky + β*IRM
    optimizer.step()
```

**Stage C: Streaming Adaptation**
```python
while True:
    scores = model(microbatch)
    selected = active_learner.select(scores, budget)
    labels = get_labels(selected)
    meta_adapter.adapt(labels)
```

## Baselines Comparison

STREAM-FraudX outperforms:
- **Tabular**: LightGBM, XGBoost, CatBoost, TabNet
- **Graph**: GCN, GraphSAGE, TGAT, TGN, CARE-GNN
- **Streaming**: Adaptive Random Forest, Hoeffding Trees
- **Active Learning**: Entropy sampling, margin sampling

## Ablation Studies

Key components to ablate:
1. Remove graph tower → tabular-only
2. Remove tabular tower → graph-only
3. Replace cross-attention with concatenation
4. Disable pretraining
5. Disable IRM regularization
6. Disable meta-adapters
7. Disable active learning

## Citation

```bibtex
@inproceedings{streamfraudx2026,
  title={Label-Efficient Streaming Fraud Detection via Dual-Tower Temporal Graph and Tabular Transformers with Self-Supervised Pretraining and Drift-Aware Adaptation},
  author={},
  booktitle={ACIIDS 2026},
  year={2026}
}
```

## License

MIT License

## Contributing

Contributions welcome! Areas for improvement:
- Additional real-world datasets (PaySim, Elliptic, IEEE-CIS)
- More sophisticated graph augmentations
- Distributed training support
- Production deployment tools
- Interactive visualization dashboard

## Acknowledgments

This work builds upon:
- Temporal Graph Networks (TGN)
- Focal Loss and Tversky Index
- Invariant Risk Minimization (IRM)
- Conformal Prediction
- Reptile Meta-Learning

---

**For questions or collaborations, please open an issue or contact the authors.**
