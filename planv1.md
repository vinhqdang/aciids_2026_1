Topic

Label-efficient streaming fraud detection in e-finance via Dual-Tower Temporal Graph + Tabular Transformers with Self-Supervised Pretraining and Drift-Aware Adaptation

Short name: STREAM-FraudX

Problem & Contribution (What’s new)

Online payment fraud drifts quickly (new merchants, user behaviors, attack vectors). Classic tabular GBMs are strong offline, but (a) miss relational patterns (money flow), (b) erode under concept drift, (c) struggle with label scarcity and latency.

STREAM-FraudX is a new end-to-end algorithm that combines:

Dual-Tower architecture

Temporal Graph Tower (TGT): a lightweight Temporal Graph Network over the evolving transaction graph (accounts/merchants/devices as nodes; transfers as timestamped edges).

Tabular Transformer Tower (TTT): a compact transformer for raw transaction features (amount, MCC, device, geo, time encodings).

Gated Cross-Attention Fusion to merge relational and tabular signals per event.

Label-efficient pretraining (no fraud labels needed):

Masked Edge Modeling (MEM): predict masked edge attributes/types.

Subgraph Contrastive (InfoNCE): pull temporally coherent neighborhoods together, push apart negatives across time/shards.

Drift-Aware Online Adaptation

Meta-Adapter: small parameter-efficient adapters in both towers, updated with a Reptile-style meta-gradient using most-recent windows.

Invariant Time Slicing (IRM-lite): regularize representations to be stable across adjacent time slices to reduce spurious temporal correlations.

Cost-sensitive detection objective

Optimizes AUPRC and precision@k proxies via Asymmetric Focal Loss + Calibrated Focal Tversky; aligns with ops goals (high precision at small k to minimize false positives to analysts).

Active Labeling with Conformal Uncertainty

Under a daily label budget, query labels for highest non-conformity + business cost items to maximize learning from limited feedback.

Architecture Overview

Event at time t: a transaction (u → v, features x).
We produce a fraud score s_t.

TGT (Temporal Graph Tower)

Node embeddings 
ℎ
𝑢
,
ℎ
𝑣
h
u
    ​

,h
v
    ​

 updated by a temporal message function 
𝑚
𝑡
=
𝜙
(
[
ℎ
𝑢
,
ℎ
𝑣
,
time2vec
(
𝑡
)
,
𝑒
_
attrs
]
)
m
t
    ​

=ϕ([h
u
    ​

,h
v
    ​

,time2vec(t),e_attrs]).

Use TGN-style memory with reservoir neighbor sampling for scalability.

Parameter-efficient Adapter-TGT blocks (bottleneck MLPs) for online updates.

TTT (Tabular Transformer Tower)

Tokenize features: continuous → learned bins; categoricals → embeddings; time → Fourier/time2vec.

2–3 transformer layers, Adapter-TTT modules for fast drift updates.

Fusion & Head

Gated Cross-Attention: Q from TTT, K/V from TGT (and vice versa) → concatenation → gated residual.

Head: 2-layer MLP → sigmoid score 
𝑠
𝑡
s
t
    ​

.

Losses

Pretrain: 
𝐿
𝑀
𝐸
𝑀
+
𝜆
𝐿
𝐼
𝑛
𝑓
𝑜
𝑁
𝐶
𝐸
L
MEM
    ​

+λL
InfoNCE
    ​


Supervised: 
𝐿
𝐴
𝐹
𝐿
+
𝛼
𝐿
𝑇
𝑣
𝑒
𝑟
𝑠
𝑘
𝑦
+
𝛽
𝐿
𝐼
𝑅
𝑀
L
AFL
    ​

+αL
Tversky
    ​

+βL
IRM
    ​


Active step: uses conformal non-conformity score for querying.

Training & Inference Workflow

Stage A — Self-Supervised Pretraining (weeks of unlabeled data)

Build rolling temporal graph (edges are transactions with timestamps).

Train towers jointly on MEM + InfoNCE with negative sampling across time buckets.

Stage B — Supervised Fine-Tuning (limited labels)
3. Fine-tune with cost-sensitive losses on labeled windows; early stop by AUPRC on next-week validation.

Stage C — Streaming Adaptation (production)
4. For each new micro-batch (e.g., 10–60 seconds):

Update TGT memory; run inference.

If daily label budget > 0, collect labels on queried items; run Meta-Adapter updates (few steps, low LR) without touching backbone.

Inference:
O(1) neighbor fetch with bounded reservoir; dual-tower forward; return score + calibrated threshold (via conformal quantile).

Pseudocode (High-Level)
# ===== Model =====
class STREAMFraudX(nn.Module):
    def __init__(self, backbone_tgt, backbone_ttt, adapters_tgt, adapters_ttt, fusion, head):
        self.tgt = backbone_tgt        # temporal graph encoder (with memory)
        self.ttt = backbone_ttt        # tabular transformer
        self.atgt = adapters_tgt       # PEFT adapters (few params)
        self.attt = adapters_ttt
        self.fusion = fusion           # gated cross-attention
        self.head = head               # MLP -> sigmoid

    def forward(self, batch_events, graph_state):
        z_graph = self.tgt(batch_events, graph_state)      # node/edge encs
        z_tab   = self.ttt(batch_events.features)
        z_fuse  = self.fusion(q=z_tab, kv=z_graph)         # cross-attn
        return torch.sigmoid(self.head(z_fuse))

# ===== Pretraining =====
for epoch in range(E_mem):
    batch = sample_unlabeled_stream()
    loss_mem = masked_edge_modeling_loss(model.tgt, batch.graph)
    loss_nce = subgraph_contrastive_loss(model, batch)
    (loss_mem + lambda_ * loss_nce).backward(); opt.step()

# ===== Supervised Fine-tune =====
for epoch in range(E_sup):
    for batch in labeled_loader:
        s = model(batch, graph_state)
        loss = asymmetric_focal_loss(s, batch.y) + alpha*tversky_loss(s,batch.y) + beta*irm_lite(model, batch, time_slices=batch.t)
        loss.backward(); opt.step()

# ===== Streaming Adaptation =====
while True:
    batch = next_microbatch()                   # last 30s events
    scores = model(batch, graph_state)
    emit(scores)
    Q = select_for_label(scores, conformal_uncertainty, business_cost, budget=B_daily)
    labels = get_labels(Q)                      # delayed/partial labels OK
    # Meta-adapter quick update
    for step in range(K):
        s = model(Q, graph_state)
        loss = asymmetric_focal_loss(s, labels)
        meta_opt.step(loss, params=model.adapters_only())  # freeze backbones

Datasets

Use at least two to show generality; one graph-temporal, one tabular-rich.

IEEE-CIS Fraud Detection (Kaggle) — large, anonymized transactional tabular dataset; highly imbalanced; rich device/ID features.

Use sliding windows to emulate streaming; build co-occurrence graph (card/device/email/IP ↔ account/merchant).

PaySim (mobile money simulator) — event stream with users, merchants, amounts; good for temporal drift experiments.

Elliptic Bitcoin Transactions — graph with labels over time (illicit vs licit); ideal to stress the temporal graph tower.

(If internal TymeX data is available later, you can add it as a held-out third dataset for external validity.)

Metrics (optimize for operations & fairness)

Primary

AUPRC (Average Precision) — robust under class imbalance.

Precision@k (k = number of daily manual reviews) — directly maps to analyst workload.

F1@threshold chosen by conformal calibration.

Secondary

ROC-AUC (report but don’t optimize).

False Positive Rate at fixed Precision (FPR@P=0.9).

Detection Delay (mean time from event to flag under streaming).

Latency & Throughput (p95 inference latency, tx/s).

Stability under Drift: AUPRC drop from week t to t+4 (lower is better).

Fairness (optional): disparate impact across regions/segments if available.

Baselines (to beat)

Tabular strong baselines

LightGBM and XGBoost with aggressive categorical handling / target encoding.

CatBoost (handles high-cardinality categoricals well).

TabTransformer / TabNet (deep tabular).

Graph/temporal baselines

GCN/GraphSAGE on static transaction graphs.

TGAT / EvolveGCN / TGN for temporal graphs (no fusion, no adapters).

CARE-GNN (robust to camouflaged fraud in graphs).

Streaming baselines

Adaptive Random Forest / Hoeffding Trees (River library) for drift.

GBM with periodic retrain (daily) — strong practical baseline.

Active learning

Uncertainty sampling via entropy; random sampling (control).

We will compare apples-to-apples on the same streaming protocol (same windowing, label latency, daily budget).

Why STREAM-FraudX should win

Relational + Tabular Fusion catches rings & mules that tabular-only miss.

Label-efficient pretraining extracts structure from vast unlabeled flows, boosting cold-start.

Meta-Adapters give minutes-level drift response without expensive full retrains.

Cost-sensitive losses + conformal calibration maximize precision where it matters (review queue).

Active labeling spends scarce labels on the most informative + costly cases.

Ablations (required for the paper)

Remove graph tower → tabular-only.

Remove tabular tower → graph-only.

No cross-attention (concat only).

No pretraining; no IRM-lite; no meta-adapters.

Active learning off vs on (various budgets).

Swap losses (plain BCE vs AFL/Tversky).

Pretraining: MEM only vs InfoNCE only.

Implementation Notes for Developers

Scale & speed

Neighbor sampling: bounded reservoir (e.g., 20 most recent neighbors per node).

Memory: keyed by node id; evict LRU.

Mixed precision; compile model; batch size by microbatch seconds.

Feature store: pre-embed high-card categorical IDs with hashing trick.

Parameter-efficient adapters

Use bottleneck MLP (e.g., d → d/r → d with GELU; r=8–16).

Separate adapters per tower, plus a tiny fusion-adapter on cross-attention projections.

Pretraining

MEM: mask 15% edges’ attributes (amount bin, MCC bin, device type), predict with cross-entropy.

InfoNCE: views are (i) temporal random walk, (ii) edge-drop + time-jitter; temperature τ=0.2–0.5; queue size 2–8k.

Losses

Asymmetric Focal Loss: γ+ ≈ 0, γ− ≈ 2; weight positives higher.

Tversky: α=0.7, β=0.3 to penalize FPs.

IRM-lite: penalize variance of class-conditional means across time buckets.

Conformal

Maintain sliding calibration set of recent predictions; compute quantile qα for non-conformity; threshold = function(qα, budget).

Serving

Stateless forward + stateful graph memory (Redis/RocksDB).

Version adapters daily; can roll back easily.

Expected Results (targets you can put in the paper)

+3–7 AUPRC points over LightGBM and +2–5 over best deep baseline on IEEE-CIS.

Higher Precision@k at identical review capacity (k equal to 0.5–1.0‰ of daily traffic).

< 20 ms p95 inference per event on A100/consumer GPU with neighbor cap 20 and 3-layer small towers.

< 1% weekly AUPRC degradation under synthetic drift vs 5–10% for static models.

Paper Outline (ACIIDS-ready)

Intro (problem, drift, label scarcity; contributions list).

Related Work (tabular fraud; graph fraud; drift; active learning).

Method: STREAM-FraudX (architecture, objectives, training, complexity).

Experimental Setup (datasets, stream protocol, label latency, budgets).

Results (main table; precision-recall curves; drift plots; latency).

Ablations & Analysis (adapters impact; pretraining benefits).

Conclusion (ops impact; limitations; future extensions).
