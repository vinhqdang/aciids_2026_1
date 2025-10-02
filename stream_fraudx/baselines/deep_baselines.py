"""
Deep learning baselines for fraud detection.
Includes: MLP, TabTransformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


class DeepBaseline:
    """Base class for deep learning baselines."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare features from batch."""
        features = {}

        if 'continuous' in batch:
            features['continuous'] = batch['continuous']

        if 'categorical' in batch:
            features['categorical'] = batch['categorical']

        return features

    def train(self, train_loader: DataLoader, epochs: int = 10, lr: float = 1e-3):
        """Train the model."""
        raise NotImplementedError

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict probabilities."""
        raise NotImplementedError


class MLPBaseline(DeepBaseline):
    """Multi-Layer Perceptron baseline."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2):
        super().__init__("MLP")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers).to(self.device)

    def train(self, train_loader: DataLoader, epochs: int = 10, lr: float = 1e-3):
        """Train the MLP model."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                # Prepare features
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['labels'].float().to(self.device)

                # Concatenate features
                features = torch.cat([continuous, categorical], dim=1)

                # Forward pass
                logits = self.model(features).squeeze()
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1} Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict fraud probabilities."""
        self.model.eval()

        all_probs = []

        for batch in test_loader:
            continuous = batch['continuous'].to(self.device)
            categorical = batch['categorical'].to(self.device)

            features = torch.cat([continuous, categorical], dim=1)
            logits = self.model(features).squeeze()
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)


class TabTransformerBaseline(DeepBaseline):
    """
    TabTransformer baseline for tabular data.

    Uses transformer attention over categorical embeddings + continuous features.
    """

    def __init__(self,
                 continuous_dim: int,
                 categorical_vocab_sizes: List[int],
                 embed_dim: int = 32,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__("TabTransformer")

        self.continuous_dim = continuous_dim
        self.embed_dim = embed_dim

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size in categorical_vocab_sizes
        ])

        # Continuous feature projection
        self.continuous_proj = nn.Linear(continuous_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        total_features = embed_dim * (len(categorical_vocab_sizes) + 1)
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.model = self.to(self.device)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = continuous.size(0)

        # Embed categorical features
        cat_embeds = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]

        # Project continuous features
        cont_embed = self.continuous_proj(continuous).unsqueeze(1)

        # Stack embeddings (batch, num_features, embed_dim)
        all_embeds = torch.stack(cat_embeds + [cont_embed.squeeze(1)], dim=1)

        # Apply transformer
        transformed = self.transformer(all_embeds)

        # Flatten and classify
        flattened = transformed.flatten(1)
        logits = self.classifier(flattened)

        return logits.squeeze()

    def train(self, train_loader: DataLoader, epochs: int = 10, lr: float = 1e-3):
        """Train the TabTransformer model."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                continuous = batch['continuous'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['labels'].float().to(self.device)

                # Forward pass
                logits = self.forward(continuous, categorical)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1} Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict fraud probabilities."""
        self.eval()

        all_probs = []

        for batch in test_loader:
            continuous = batch['continuous'].to(self.device)
            categorical = batch['categorical'].to(self.device)

            logits = self.forward(continuous, categorical)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)
