"""
Stage A: Self-Supervised Pretraining for STREAM-FraudX
Implements MEM + InfoNCE pretraining on unlabeled transaction streams.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.losses.pretraining_losses import MaskedEdgeModelingLoss, SubgraphContrastiveLoss
from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn


def pretrain(args):
    """Run self-supervised pretraining."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage A] Self-Supervised Pretraining")
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data =====
    print("[1/4] Loading unlabeled data...")
    dataset = SyntheticFraudDataset(
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        fraud_rate=args.fraud_rate,
        num_continuous=args.num_continuous,
        num_categorical=args.num_categorical,
        seed=args.seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")

    # ===== Model =====
    print("\n[2/4] Building model...")
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(args.num_continuous))
    config.categorical_vocab_sizes = [100] * args.num_categorical

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=False  # No adapters during pretraining
    ).to(device)

    model_size = model.get_model_size()
    print(f"Model parameters: {model_size['total']:,}")

    # ===== Losses =====
    print("\n[3/4] Setting up pretraining losses...")

    # MEM Loss
    mem_loss = MaskedEdgeModelingLoss(
        mask_ratio=0.15,
        attribute_dims={'amount_bin': 50, 'mcc_bin': 20, 'device_type': 10}
    )
    # Build prediction heads
    mem_loss.build_heads(input_dim=model.tgt.node_dim * 2)
    mem_loss = mem_loss.to(device)

    # Contrastive Loss
    contrastive_loss = SubgraphContrastiveLoss(
        temperature=0.2,
        queue_size=2048,
        use_queue=True
    ).to(device)

    # Optimizer (all parameters including MEM heads)
    all_params = list(model.parameters()) + list(mem_loss.parameters())
    optimizer = optim.AdamW(all_params, lr=args.learning_rate, weight_decay=1e-5)

    # ===== Training =====
    print(f"\n[4/4] Pretraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        mem_loss.train()

        total_mem_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')

        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else
                    ({k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                    for k, v in batch.items()}

            # === MEM Loss ===
            masked_batch, mask, targets = mem_loss.mask_edges(batch)

            # Forward pass through model
            edge_reps, embeddings = model(masked_batch, update_memory=False, return_embeddings=True)

            # Compute MEM loss
            loss_mem = mem_loss(embeddings['graph'], mask, targets)

            # === Contrastive Loss ===
            # Create augmented views
            view1 = embeddings['fused']

            # Augment batch for second view
            augmented_batch = contrastive_loss._augment_events(batch)
            _, aug_embeddings = model(augmented_batch, update_memory=False, return_embeddings=True)
            view2 = aug_embeddings['fused']

            loss_contrastive = contrastive_loss(view1, view2)

            # Combined loss
            loss = args.mem_weight * loss_mem + args.contrastive_weight * loss_contrastive

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_mem_loss += loss_mem.item()
            total_contrastive_loss += loss_contrastive.item()
            num_batches += 1

            pbar.set_postfix({
                'mem': f'{total_mem_loss/num_batches:.4f}',
                'contrast': f'{total_contrastive_loss/num_batches:.4f}'
            })

        # Epoch summary
        avg_mem = total_mem_loss / num_batches
        avg_contrastive = total_contrastive_loss / num_batches
        print(f"\nEpoch {epoch} Summary:")
        print(f"  MEM Loss: {avg_mem:.4f}")
        print(f"  Contrastive Loss: {avg_contrastive:.4f}")
        print(f"  Total Loss: {avg_mem * args.mem_weight + avg_contrastive * args.contrastive_weight:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'pretrain_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mem_loss_state_dict': mem_loss.state_dict(),
                'config': config.to_dict()
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = output_dir / 'pretrained_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }, final_path)

    print(f"\n✓ Pretraining complete! Model saved to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage A: Self-Supervised Pretraining')

    # Data
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    parser.add_argument('--num_nodes', type=int, default=2000, help='Number of nodes')
    parser.add_argument('--fraud_rate', type=float, default=0.01, help='Fraud rate (unused in pretraining)')
    parser.add_argument('--num_continuous', type=int, default=10)
    parser.add_argument('--num_categorical', type=int, default=5)

    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Pretraining epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mem_weight', type=float, default=1.0, help='MEM loss weight')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='Contrastive loss weight')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain', help='Output directory')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pretrain(args)
