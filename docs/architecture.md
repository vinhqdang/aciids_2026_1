# STREAM-FraudX Architecture Documentation

## Overview

STREAM-FraudX is a multi-tower deep learning system for real-time fraud detection with three key stages:

1. **Stage A**: Self-supervised pretraining
2. **Stage B**: Supervised fine-tuning
3. **Stage C**: Streaming adaptation with active learning

## System Architecture

### 1. Data Flow

```
Raw Transaction → Data Module → Micro-Batches → Model → Predictions
                        ↓
                 Label Queue (delayed)
                        ↓
                Active Learner Selection
                        ↓
                Meta-Adapter Update
```

### 2. Model Architecture

#### 2.1 Temporal Graph Tower (TGT)

**Purpose**: Captures relational patterns and temporal graph structure

**Components**:
- **Node Features**: Transaction attributes embedded
- **Edge Features**: Interaction types, amounts, timestamps
- **Time2Vec Encoding**: Periodic + linear temporal encoding
- **Memory Bank**: TGN-style memory with LRU eviction (capacity: 1000)
- **Message Passing**: 3-layer temporal GNN
- **Aggregation**: Mean pooling over neighbors

**Dimensions**:
- Node embedding: 128
- Edge embedding: 64
- Memory dimension: 128

**Code Location**: `stream_fraudx/models/temporal_graph_tower.py`

#### 2.2 Tabular Transformer Tower (TTT)

**Purpose**: Processes heterogeneous tabular features

**Components**:
- **Feature Tokenization**:
  - Continuous features → Binning (20 bins) + Embedding (dim 32)
  - Categorical features → Embedding (vocab-dependent, dim 32)
- **Positional Encoding**: Fourier features for time
- **Self-Attention**: 8 heads, 3 layers
- **CLS Token**: Global transaction representation

**Dimensions**:
- Token embedding: 128
- Feedforward: 512
- Dropout: 0.1

**Code Location**: `stream_fraudx/models/tabular_transformer_tower.py`

#### 2.3 Gated Cross-Attention Fusion

**Purpose**: Combines graph and tabular representations

**Mechanism**:
1. Query TGT with TTT: `Q_tgt = W_q(TTT), K_tgt=V_tgt=TGT`
2. Query TTT with TGT: `Q_ttt = W_q(TGT), K_ttt=V_ttt=TTT`
3. Gated combination: `Fused = α·Attended_tgt + (1-α)·Attended_ttt`

**Dimensions**:
- Attention dimension: 256
- Gate: learnable sigmoid

**Code Location**: `stream_fraudx/models/fusion.py`

#### 2.4 Adapters (Optional)

**Purpose**: Parameter-efficient fine-tuning and adaptation

**Types**:
- **LoRA**: Low-rank decomposition (rank 8)
  - `Δh = BA(h)` where A ∈ R^{d×r}, B ∈ R^{r×d}
- **Prefix**: Prepended learnable tokens (length 10)

**Insertion Points**:
- After each transformer layer in TTT
- After message passing in TGT

**Code Location**: `stream_fraudx/models/adapters.py`

### 3. Loss Functions

#### 3.1 Training Losses (Stage B)

**Combined Focal Loss**:
```python
AFL = -α(1-p)^γ log(p)              # Asymmetric Focal
Tversky = 1 - TP/(TP + αFN + βFP)   # Tversky Index
Loss = AFL + λ·Tversky
```

**IRM Regularization**:
```python
IRM_penalty = Var_e[∇_w R^e(w)]     # Variance of gradients across environments
Total = Loss + β·IRM_penalty
```

**Gradient Fix**: Use detach() to prevent double-backward:
```python
embeddings_detached = {k: v.detach().requires_grad_(True)
                      for k, v in embeddings.items()}
irm_loss = compute_irm(embeddings_detached)
```

**Code Location**: `stream_fraudx/losses/`

#### 3.2 Pretraining Losses (Stage A)

**Masked Edge Modeling (MEM)**:
- Mask 15% of edges
- Predict masked attributes: amount_bin, mcc_bin, device_type
- Cross-entropy loss per attribute

**Subgraph Contrastive (InfoNCE)**:
- Create augmented views (dropout edges, mask features)
- Maximize agreement: `L = -log[exp(sim(z1,z2)/τ) / Σ exp(sim(z1,z_neg)/τ)]`
- Queue size: 2048

**Code Location**: `stream_fraudx/losses/pretraining_losses.py`

### 4. Training Pipeline

#### 4.1 Stage A: Pretraining

**Objective**: Learn general representations from unlabeled data

**Procedure**:
1. Load unlabeled transaction stream
2. For each batch:
   - Generate masked edges
   - Compute MEM loss
   - Create augmented views
   - Compute InfoNCE loss
   - Backprop combined loss
3. Save encoder weights

**Hyperparameters**:
- Epochs: 20
- Batch size: 512
- Learning rate: 1e-3
- MEM weight: 1.0
- Contrastive weight: 1.0

**Output**: `pretrained_model.pt` (encoder weights only)

#### 4.2 Stage B: Fine-Tuning

**Objective**: Adapt to fraud detection with labeled data

**Procedure**:
1. Load pretrained encoder
2. (Optional) Freeze backbone, train adapters+head only
3. For each epoch:
   - Forward pass
   - Compute supervised loss + IRM penalty
   - Backprop with gradient clipping
   - Evaluate on validation set
4. Save best model

**Hyperparameters**:
- Epochs: 30
- Batch size: 256
- Learning rate: 1e-3 (backbone), 2e-3 (adapters/head)
- IRM weight: 0.1
- Gradient clip: 1.0

**Output**: `best_model.pt` (full model)

#### 4.3 Stage C: Streaming

**Objective**: Continuous adaptation to drift with limited labels

**Components**:

1. **Micro-Batch Processing**:
   - Window: 30 seconds
   - Size: 100 transactions

2. **Label Latency Queue**:
   - Delay: 10 micro-batches (~5 minutes)
   - Simulates real-world labeling delay

3. **Active Learning**:
   - **Conformal Uncertainty**: Calibrated prediction intervals
   - **Acquisition**: Uncertainty + business cost + diversity
   - **Budget**: 100 labels/day

4. **Meta-Adapter**:
   - **Algorithm**: Reptile (simplified MAML)
   - **Inner loop**: 5 gradient steps on labeled buffer
   - **Outer loop**: Meta-update toward adapted weights
   - **Meta-LR**: 0.01, Inner-LR: 0.001

5. **Drift Detection**:
   - Monitor: Calibration error over 100-sample window
   - Threshold: 10% relative drop
   - Action: Reset adapters to meta-initialization

**Hyperparameters**:
- Micro-batch size: 100
- Label delay: 10 batches
- Daily budget: 100
- Adaptation frequency: every 10 batches
- Meta-LR: 0.01
- Inner-LR: 0.001
- Inner steps: 5

**Output**: `streaming_results.json`, `meta_adapter.pt`

### 5. Data Module

#### 5.1 Dataset Abstraction

**Unified Interface**:
```python
class StreamDataModule:
    def setup():                    # Load datasets
    def train_dataloader():         # Standard training
    def val_dataloader():           # Validation
    def test_dataloader():          # Test evaluation
    def stream_dataloader():        # Micro-batch streaming
```

**Supported Datasets**:
- `synthetic`: Generated on-the-fly
- `ieee-cis`: IEEE-CIS Fraud Detection
- `paysim`: PaySim Mobile Money
- `elliptic`: Elliptic Bitcoin

**Code Location**: `stream_fraudx/data/module.py`

#### 5.2 Micro-Batch Streaming

**MicroBatchStream**:
- Slices dataset into fixed-size windows
- Adds timestamp metadata
- Supports reset for multiple passes

**Label Latency Queue**:
- Delays label availability by N batches
- Tracks sample age
- Returns labels when ready

### 6. Baselines

#### 6.1 Traditional ML

**Gradient Boosting**:
- LightGBM (CPU/GPU)
- XGBoost (CPU/GPU)
- CatBoost (CPU/GPU)

**Others**:
- Random Forest
- Logistic Regression

**Feature Preparation**:
- Flatten tabular features
- No graph structure used

**Code Location**: `stream_fraudx/baselines/ml_baselines.py`

#### 6.2 Deep Learning

**MLP**:
- 3 hidden layers: [256, 128, 64]
- Batch normalization + Dropout (0.2)

**TabTransformer**:
- Embedding dim: 32
- Transformer: 8 heads, 3 layers
- Classification head: 128 → 1

**Code Location**: `stream_fraudx/baselines/deep_baselines.py`

#### 6.3 Streaming

**River Models**:
- Adaptive Random Forest (10 trees)
- Hoeffding Adaptive Tree
- Single-pass online learning

**Code Location**: `stream_fraudx/baselines/streaming_baselines.py`

## Performance Characteristics

### Computational Complexity

**Training (per epoch)**:
- TGT: O(E·d²) where E = edges, d = embedding dim
- TTT: O(N·L²·d) where N = batch, L = sequence length
- Fusion: O(N·d²)

**Inference (per sample)**:
- Forward pass: ~15ms (GPU), ~50ms (CPU)
- Memory bank lookup: O(log K) where K = memory size

### Model Size

**Full Model**:
- Total parameters: ~5-10M
- TGT: ~2M
- TTT: ~2M
- Fusion: ~500K
- Adapters: ~100K (rank 8)

**Adapter-Only Fine-Tuning**:
- Trainable: ~600K (12% of full model)
- Frozen: ~4.4M

### Memory Requirements

**Training**:
- GPU: 8-12 GB (batch 256)
- CPU: 16 GB RAM

**Inference**:
- GPU: 2-4 GB
- CPU: 4 GB RAM

## Extension Points

### Adding New Datasets

1. Create loader in `stream_fraudx/data/`:
   ```python
   class MyDataset(Dataset):
       def __getitem__(self, idx):
           return {
               'continuous': ...,
               'categorical': ...,
               'labels': ...,
               'src_nodes': ...,
               'dst_nodes': ...
           }
   ```

2. Add to `StreamDataModule._setup_mydataset()`

3. Register in CLI: `--dataset mydataset`

### Adding New Baselines

1. Inherit from base class:
   ```python
   class MyBaseline(MLBaseline):
       def __init__(self, **kwargs):
           super().__init__("MyBaseline")
           self.model = MyModel(**kwargs)

       def train(self, train_data, train_labels):
           self.model.fit(train_data, train_labels)

       def predict(self, test_data):
           return self.model.predict_proba(test_data)[:, 1]
   ```

2. Add to baseline runner

### Adding New Losses

1. Define loss in `stream_fraudx/losses/`:
   ```python
   class MyLoss(nn.Module):
       def forward(self, logits, labels):
           ...
           return loss
   ```

2. Integrate in trainer

## References

- **TGN**: Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs", ICML 2020
- **IRM**: Arjovsky et al., "Invariant Risk Minimization", arXiv 2019
- **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018
- **Conformal Prediction**: Vovk et al., "Algorithmic Learning in a Random World", 2005
