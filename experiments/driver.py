"""
Unified Experiment Driver for STREAM-FraudX.
Consolidates main.py, run_simple_baselines.py, run_all_experiments.py into a single runner.
Supports both neural models and classical baselines with reproducible logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from datetime import datetime
from functools import partial

from .logger import ExperimentLogger
from .config import ExperimentConfig
from .utils import set_seed, get_device, count_parameters, worker_init_fn

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn
from stream_fraudx.baselines.ml_baselines import (
    RandomForestBaseline, LogisticRegressionBaseline,
    LightGBMBaseline, XGBoostBaseline, CatBoostBaseline
)
from stream_fraudx.losses.focal_losses import AsymmetricFocalLoss, CombinedFocalLoss
from stream_fraudx.utils.metrics import compute_metrics


class ExperimentDriver:
    """
    Unified driver for running STREAM-FraudX experiments.

    Features:
    - Single source of truth for training/evaluation
    - Supports neural models and ML baselines
    - Reproducible experiments with deterministic seeds
    - Structured logging to JSON/CSV
    - Checkpoint management
    - Resume support
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment driver.

        Args:
            config: Complete experiment configuration
        """
        self.config = config

        # Generate run ID if not provided
        if config.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.run_id = f"{config.experiment_name}_{timestamp}"

        # Initialize logger
        self.logger = ExperimentLogger(
            run_id=config.run_id,
            output_dir=config.output_dir
        )

        # Set seed
        seed_info = set_seed(config.seed, deterministic=True)
        self.logger.log_seeds(config.seed, seed_info)

        # Get device
        self.device = get_device(config.device)
        if config.verbose:
            print(f"Using device: {self.device}")

        # Log configuration
        self.logger.log_hyperparameters(config.to_dict())

        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def setup_data(self):
        """Setup data loaders based on configuration."""
        if self.config.verbose:
            print(f"\n[1/5] Setting up data: {self.config.data.dataset_name}")

        dataset_name = self.config.data.dataset_name

        if dataset_name == "synthetic":
            dataset = SyntheticFraudDataset(
                num_samples=self.config.data.num_samples or 20000,
                num_nodes=self.config.data.num_nodes,
                fraud_rate=0.05,
                num_continuous=self.config.data.num_continuous,
                num_categorical=self.config.data.num_categorical
            )

            # Split dataset
            train_size = int(self.config.data.train_split * len(dataset))
            val_size = int(self.config.data.val_split * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )

            # Create data loaders
            worker_fn = partial(worker_init_fn, seed=self.config.seed)

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=self.config.data.shuffle,
                num_workers=self.config.data.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=worker_fn
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=worker_fn
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=worker_fn
            )

        elif dataset_name in ["ieee-cis", "paysim", "elliptic"]:
            # TODO: Implement proper DataLoader for real datasets
            raise NotImplementedError(f"{dataset_name} data loading not yet implemented in driver. Use 'synthetic' for now.")

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if self.config.verbose:
            print(f"Train: {len(self.train_loader.dataset)}, "
                  f"Val: {len(self.val_loader.dataset)}, "
                  f"Test: {len(self.test_loader.dataset)}")

    def setup_model(self):
        """Setup model based on configuration."""
        if self.config.verbose:
            print(f"\n[2/5] Setting up model: {self.config.model.model_type}")

        model_type = self.config.model.model_type

        if model_type == "stream_fraudx":
            # Create STREAM-FraudX model
            model_config = STREAMFraudXConfig()
            model_config.hidden_dim = self.config.model.hidden_dim
            model_config.num_heads = self.config.model.num_heads
            model_config.num_layers = self.config.model.num_layers
            model_config.dropout = self.config.model.dropout

            self.model = STREAMFraudX(
                continuous_dims=list(range(self.config.data.num_continuous)),
                categorical_vocab_sizes=self.config.data.categorical_vocab_sizes,
                use_adapters=self.config.model.use_adapters,
                hidden_dim=self.config.model.hidden_dim,
                num_heads=self.config.model.num_heads,
                dropout=self.config.model.dropout
            ).to(self.device)

            # Log model info
            model_info = count_parameters(self.model)
            model_info['architecture'] = 'STREAM-FraudX'
            self.logger.log_model_info(model_info)

            if self.config.verbose:
                print(f"Model parameters: {model_info['total']:,}")

        else:
            # Baseline models don't need PyTorch setup
            self.model = None

    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        if self.model is None:
            return  # Skip for baseline models

        if self.config.verbose:
            print(f"\n[3/5] Setting up training")

        # Optimizer
        if self.config.training.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        # Scheduler
        if self.config.training.use_cosine_schedule:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs
            )
        else:
            self.scheduler = None

        # Loss function
        if self.config.training.loss_type == "focal":
            self.criterion = AsymmetricFocalLoss()
        elif self.config.training.loss_type == "combined_focal":
            self.criterion = CombinedFocalLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        if self.config.verbose:
            print(f"Optimizer: {self.config.training.optimizer}")
            print(f"Loss: {self.config.training.loss_type}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['labels'].float().unsqueeze(1))

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip_norm
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = avg_loss

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['labels'].float().unsqueeze(1))

                total_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = avg_loss

        return metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(batch)
                preds = torch.sigmoid(outputs).cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        return metrics

    def train_neural_model(self):
        """Train neural model (STREAM-FraudX)."""
        if self.config.verbose:
            print(f"\n[4/5] Training neural model")

        best_val_metric = 0.0
        patience_counter = 0

        for epoch in range(1, self.config.training.max_epochs + 1):
            self.logger.start_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log metrics
            epoch_metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            self.logger.end_epoch(epoch_metrics)

            # Print progress
            if self.config.verbose and epoch % self.config.log_frequency == 0:
                print(f"\nEpoch {epoch}/{self.config.training.max_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                      f"AUPRC: {train_metrics['auprc']:.4f}, "
                      f"ROC-AUC: {train_metrics['roc_auc']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"AUPRC: {val_metrics['auprc']:.4f}, "
                      f"ROC-AUC: {val_metrics['roc_auc']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if self.config.save_checkpoints and epoch % self.config.checkpoint_frequency == 0:
                checkpoint_path = self.logger.run_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch)
                self.logger.log_artifact(f"checkpoint_epoch_{epoch}", checkpoint_path, "checkpoint")

            # Early stopping
            val_metric = val_metrics[self.config.training.early_stopping_metric]
            if self.config.training.early_stopping_mode == "max":
                improved = val_metric > best_val_metric
            else:
                improved = val_metric < best_val_metric

            if improved:
                best_val_metric = val_metric
                patience_counter = 0
                # Save best model
                best_path = self.logger.run_dir / "best_model.pt"
                self.save_checkpoint(best_path, epoch)
                self.logger.log_artifact("best_model", best_path, "checkpoint")
            else:
                patience_counter += 1

            if patience_counter >= self.config.training.early_stopping_patience:
                if self.config.verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

        # Test
        if self.config.verbose:
            print(f"\n[5/5] Testing model")

        # Load best model
        best_path = self.logger.run_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.test()

        if self.config.verbose:
            print(f"\nTest Results:")
            print(f"  AUPRC: {test_metrics['auprc']:.4f}")
            print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
            print(f"  F1: {test_metrics['f1']:.4f}")

        return test_metrics

    def train_baseline(self, baseline_class, **kwargs):
        """Train a classical ML baseline."""
        if self.config.verbose:
            print(f"\n[4/5] Training baseline: {baseline_class.__name__}")

        # Prepare data
        X_train, y_train = [], []
        for batch in self.train_loader:
            features = np.concatenate([
                batch['continuous'].numpy(),
                batch['categorical'].numpy()
            ], axis=1)
            X_train.append(features)
            y_train.append(batch['labels'].numpy())

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        X_test, y_test = [], []
        for batch in self.test_loader:
            features = np.concatenate([
                batch['continuous'].numpy(),
                batch['categorical'].numpy()
            ], axis=1)
            X_test.append(features)
            y_test.append(batch['labels'].numpy())

        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)

        # Train
        baseline = baseline_class(**kwargs)
        start_time = time.time()
        baseline.train(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        y_pred = baseline.predict(X_test)

        # Evaluate
        test_metrics = compute_metrics(y_test, y_pred)
        test_metrics['train_time'] = train_time

        if self.config.verbose:
            print(f"\nTest Results:")
            print(f"  AUPRC: {test_metrics['auprc']:.4f}")
            print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
            print(f"  F1: {test_metrics['f1']:.4f}")
            print(f"  Training time: {train_time:.2f}s")

        return test_metrics

    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def run(self):
        """Run complete experiment."""
        try:
            # Setup
            self.setup_data()
            self.setup_model()
            self.setup_training()

            # Train
            if self.config.model.model_type == "stream_fraudx":
                test_metrics = self.train_neural_model()
            else:
                # Run baseline
                baseline_map = {
                    'random_forest': RandomForestBaseline,
                    'logistic_regression': LogisticRegressionBaseline,
                    'lightgbm': LightGBMBaseline,
                    'xgboost': XGBoostBaseline,
                    'catboost': CatBoostBaseline
                }
                baseline_class = baseline_map.get(self.config.model.model_type)
                if baseline_class is None:
                    raise ValueError(f"Unknown model type: {self.config.model.model_type}")
                test_metrics = self.train_baseline(baseline_class)

            # Finalize
            self.logger.finalize(status="completed", final_metrics=test_metrics)

            return test_metrics

        except Exception as e:
            self.logger.finalize(status="failed")
            raise e
