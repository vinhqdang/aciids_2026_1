"""
Training pipeline for STREAM-FraudX.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import time
from tqdm import tqdm
from pathlib import Path

from ..losses.focal_losses import CombinedFocalLoss
from ..losses.irm_loss import IRMLiteLoss
from ..utils.metrics import compute_metrics


class STREAMFraudXTrainer:
    """
    Trainer for STREAM-FraudX with three stages:
    1. Self-supervised pretraining (Stage A)
    2. Supervised fine-tuning (Stage B) - with optional checkpoint loading and adapter-only training
    3. Streaming adaptation (Stage C)
    """
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 pretrained_checkpoint: Optional[str] = None,
                 freeze_backbone: bool = False,
                 irm_weight: float = 0.1):
        self.model = model.to(device)
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.irm_weight = irm_weight

        # Load pretrained checkpoint if provided
        if pretrained_checkpoint:
            self.load_pretrained_encoder(pretrained_checkpoint)

        # Freeze backbone if requested (Stage B adapter-only training)
        if freeze_backbone:
            self._freeze_backbone_layers()

        # Setup optimizer with parameter groups
        self.optimizer = self._create_optimizer(learning_rate, weight_decay)

        # Losses
        self.supervised_loss = CombinedFocalLoss()
        self.irm_loss = IRMLiteLoss()

    def load_pretrained_encoder(self, checkpoint_path: str):
        """
        Load pretrained encoder weights from Stage A.

        Args:
            checkpoint_path: Path to pretrained checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load weights (strict=False to allow missing classifier head)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print(f"Loaded pretrained checkpoint from {checkpoint_path}")
        if missing:
            print(f"  Missing keys (expected for new heads): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def _freeze_backbone_layers(self):
        """
        Freeze backbone (TGT + TTT) and only train adapters and classification head.
        """
        # Freeze TGT and TTT towers
        for name, param in self.model.named_parameters():
            if 'tgt' in name or 'ttt' in name:
                if 'adapter' not in name:  # Keep adapters trainable
                    param.requires_grad = False

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Backbone frozen: {trainable_params:,}/{total_params:,} parameters trainable "
              f"({100 * trainable_params / total_params:.1f}%)")

    def _create_optimizer(self, learning_rate: float, weight_decay: float) -> optim.Optimizer:
        """
        Create optimizer with separate parameter groups for backbone and adapters.

        Args:
            learning_rate: Base learning rate
            weight_decay: Weight decay

        Returns:
            Optimizer instance
        """
        # Separate parameters into groups
        backbone_params = []
        adapter_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'adapter' in name:
                adapter_params.append(param)
            elif 'classifier' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Create parameter groups with different LRs
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': learning_rate,
                'name': 'backbone'
            })

        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': learning_rate * 2,  # Higher LR for adapters
                'name': 'adapters'
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': learning_rate * 2,  # Higher LR for head
                'name': 'head'
            })

        return optim.AdamW(param_groups, weight_decay=weight_decay)

    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_supervised_loss = 0.0
        total_irm_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward pass
            labels = batch['labels']

            # Get embeddings if IRM is needed
            use_irm = 'time_slices' in batch and self.irm_weight > 0

            if use_irm:
                logits, embeddings = self.model(batch, return_embeddings=True)
            else:
                logits = self.model(batch)

            # Compute supervised loss
            supervised_loss = self.supervised_loss(logits, labels)

            # Compute IRM penalty with gradient fix
            if use_irm:
                # IMPORTANT: Detach embeddings to avoid double backward through IRM
                # This prevents "RuntimeError: Trying to backward through the graph a second time"
                embeddings_detached = {k: v.detach().requires_grad_(True)
                                      for k, v in embeddings.items()}

                irm_penalty = self.irm_loss(
                    embeddings_detached['fused'],
                    labels,
                    batch['time_slices']
                )

                # Combine losses
                loss = supervised_loss + self.irm_weight * irm_penalty
                total_irm_loss += irm_penalty.item()
            else:
                loss = supervised_loss
                irm_penalty = torch.tensor(0.0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            total_supervised_loss += supervised_loss.item()
            num_batches += 1

            # Update progress bar
            postfix = {'loss': total_loss / num_batches}
            if use_irm:
                postfix['irm'] = total_irm_loss / num_batches
            pbar.set_postfix(postfix)

        metrics = {
            'loss': total_loss / num_batches,
            'supervised_loss': total_supervised_loss / num_batches,
        }

        if use_irm:
            metrics['irm_loss'] = total_irm_loss / num_batches

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        all_scores = []
        all_labels = []

        for batch in tqdm(val_loader, desc='Evaluating'):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            logits = self.model(batch, update_memory=False)
            scores = torch.sigmoid(logits)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

        # Compute metrics
        import numpy as np
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_scores)
        )

        return metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
