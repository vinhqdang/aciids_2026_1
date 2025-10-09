"""
Experiment configuration system.
Provides unified, reproducible configuration for all experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_name: str = "synthetic"  # synthetic, ieee-cis, paysim, elliptic
    num_samples: Optional[int] = None  # None = use all data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True

    # Feature configuration
    num_continuous: int = 10
    num_categorical: int = 5
    categorical_vocab_sizes: List[int] = field(default_factory=lambda: [100] * 5)

    # Graph configuration
    num_nodes: int = 1000
    graph_window_size: int = 100
    use_caching: bool = True

    # Preprocessing
    scaling: str = "standard"  # standard, minmax, robust, none
    normalization: bool = True
    feature_engineering: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "stream_fraudx"  # stream_fraudx, baseline

    # STREAM-FraudX architecture
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Temporal Graph Tower
    graph_use_attention: bool = True
    graph_attention_type: str = "recency_weighted"  # mean, recency_weighted
    graph_hot_node_cache: bool = True
    graph_cache_size: int = 1000

    # Tabular Tower
    tabular_use_feature_gating: bool = True
    tabular_use_ft_transformer: bool = True
    tabular_num_blocks: int = 3

    # Fusion
    fusion_type: str = "film_modulation"  # cross_attention, film_modulation, concat
    fusion_residual: bool = True

    # Adapters
    use_adapters: bool = False
    adapter_bottleneck_dim: int = 16

    # Meta-learning
    use_meta_learning: bool = False
    meta_lr: float = 1e-4


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30

    # Loss function
    loss_type: str = "combined_focal"  # focal, asymmetric_focal, combined_focal
    use_irm: bool = False
    irm_penalty: float = 0.1

    # Sampling
    use_label_aware_sampling: bool = False
    oversample_ratio: float = 1.0

    # Learning rate scheduling
    use_warmup: bool = True
    warmup_epochs: int = 5
    use_cosine_schedule: bool = True

    # Advanced optimization
    use_amp: bool = True  # Automatic Mixed Precision
    use_ema: bool = True  # Exponential Moving Average
    ema_decay: float = 0.999
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_start_epoch: int = 20

    # Gradient management
    grad_clip_norm: float = 1.0
    use_grad_checkpointing: bool = False

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auprc"
    early_stopping_mode: str = "max"


@dataclass
class PretrainingConfig:
    """Stage-A pretraining configuration."""
    enabled: bool = False
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 256

    # Pretraining tasks
    use_contrastive: bool = True
    use_reconstruction: bool = True
    use_temporal_ordering: bool = True

    # Checkpoint
    checkpoint_path: Optional[str] = None
    freeze_backbone: bool = False
    adapter_only: bool = False


@dataclass
class StreamingConfig:
    """Stage-C streaming adaptation configuration."""
    enabled: bool = False

    # Online learning
    meta_batch_size: int = 32
    meta_lr: float = 1e-4
    adaptation_steps: int = 5

    # Drift detection
    use_drift_detection: bool = True
    drift_threshold: float = 0.1
    drift_window_size: int = 1000

    # Conformal prediction
    use_conformal: bool = True
    conformal_alpha: float = 0.1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    run_id: Optional[str] = None  # Auto-generated if None
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, auto

    # Component configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Output
    output_dir: str = "artifacts/runs"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5

    # Logging
    log_frequency: int = 1
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to file (JSON or YAML)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            if path.suffix == '.json':
                json.dump(self.to_dict(), f, indent=2)
            elif path.suffix in ['.yaml', '.yml']:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from file."""
        path = Path(path)

        with open(path, 'r') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        # Recursively create config objects
        return cls(
            experiment_name=data.get('experiment_name', 'default_experiment'),
            run_id=data.get('run_id'),
            seed=data.get('seed', 42),
            device=data.get('device', 'cuda'),
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            pretraining=PretrainingConfig(**data.get('pretraining', {})),
            streaming=StreamingConfig(**data.get('streaming', {})),
            output_dir=data.get('output_dir', 'artifacts/runs'),
            save_checkpoints=data.get('save_checkpoints', True),
            checkpoint_frequency=data.get('checkpoint_frequency', 5),
            log_frequency=data.get('log_frequency', 1),
            verbose=data.get('verbose', True)
        )

    @classmethod
    def from_args(cls, args) -> 'ExperimentConfig':
        """Create config from argparse arguments."""
        config = cls()

        # Map common arguments to config fields
        if hasattr(args, 'dataset'):
            config.data.dataset_name = args.dataset
        if hasattr(args, 'num_samples'):
            config.data.num_samples = args.num_samples
        if hasattr(args, 'batch_size'):
            config.data.batch_size = args.batch_size
        if hasattr(args, 'epochs'):
            config.training.max_epochs = args.epochs
        if hasattr(args, 'lr'):
            config.training.learning_rate = args.lr
        if hasattr(args, 'seed'):
            config.seed = args.seed

        return config
