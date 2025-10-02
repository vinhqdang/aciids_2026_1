"""
Training pipeline for STREAM-FraudX.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import time
from tqdm import tqdm

from ..losses.focal_losses import CombinedFocalLoss
from ..losses.irm_loss import IRMLiteLoss
from ..utils.metrics import compute_metrics


class STREAMFraudXTrainer:
    """
    Trainer for STREAM-FraudX with three stages:
    1. Self-supervised pretraining
    2. Supervised fine-tuning
    3. Streaming adaptation
    """
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Losses
        self.supervised_loss = CombinedFocalLoss()
        self.irm_loss = IRMLiteLoss()

    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward pass (get embeddings for IRM if needed)
            labels = batch['labels']

            if 'time_slices' in batch:
                logits, embeddings = self.model(batch, return_embeddings=True)
                # Compute main loss
                loss = self.supervised_loss(logits, labels)
                # Add IRM penalty
                irm_penalty = self.irm_loss(
                    embeddings['fused'], labels, batch['time_slices']
                )
                loss = loss + 0.1 * irm_penalty
            else:
                logits = self.model(batch)
                loss = self.supervised_loss(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': total_loss / num_batches})

        return {'loss': total_loss / num_batches}

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
