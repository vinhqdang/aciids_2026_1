"""
Main execution script for STREAM-FraudX
Runs training, evaluation, and streaming inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import json
from pathlib import Path

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.training.trainer import STREAMFraudXTrainer
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn


def main(args):
    """Main training and evaluation pipeline."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data =====
    print("\n[1/5] Loading Data...")
    full_dataset = SyntheticFraudDataset(
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        fraud_rate=args.fraud_rate,
        num_continuous=args.num_continuous,
        num_categorical=args.num_categorical
    )

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ===== Model =====
    print("\n[2/5] Building Model...")
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(args.num_continuous))
    config.categorical_vocab_sizes = [100] * args.num_categorical

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=args.use_adapters
    )

    model_size = model.get_model_size()
    print(f"Model parameters: {model_size['total']:,}")
    print(f"  - TGT: {model_size['tgt']:,}")
    print(f"  - TTT: {model_size['ttt']:,}")
    print(f"  - Fusion: {model_size['fusion']:,}")
    print(f"  - Head: {model_size['head']:,}")
    print(f"  - Adapters: {model_size['adapters']:,}")

    # ===== Training =====
    print("\n[3/5] Training Model...")
    trainer = STREAMFraudXTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate
    )

    best_auprc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        if epoch % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"  Val AUPRC: {val_metrics['auprc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']
                best_epoch = epoch
                trainer.save_checkpoint(
                    output_dir / 'best_model.pt',
                    epoch,
                    val_metrics
                )
                print(f"  → New best model saved!")

    print(f"\nBest epoch: {best_epoch} with AUPRC: {best_auprc:.4f}")

    # ===== Evaluation =====
    print("\n[4/5] Evaluating on Test Set...")
    trainer.load_checkpoint(output_dir / 'best_model.pt')
    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Results:")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  FPR: {test_metrics['fpr']:.4f}")

    if 'precision@100' in test_metrics:
        print(f"  Precision@100: {test_metrics['precision@100']:.4f}")
    if 'precision@500' in test_metrics:
        print(f"  Precision@500: {test_metrics['precision@500']:.4f}")

    # Save results
    results = {
        'config': config.to_dict(),
        'model_size': model_size,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[5/5] Results saved to {output_dir}/results.json")
    print("\n✓ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STREAM-FraudX Training')

    # Data
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_nodes', type=int, default=1000)
    parser.add_argument('--fraud_rate', type=float, default=0.01)
    parser.add_argument('--num_continuous', type=int, default=10)
    parser.add_argument('--num_categorical', type=int, default=5)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5)

    # Model
    parser.add_argument('--use_adapters', action='store_true', default=True)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()

    main(args)
