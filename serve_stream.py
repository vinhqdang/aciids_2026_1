"""
Stage C: Streaming Adaptation for STREAM-FraudX
Executes micro-batch streaming with active learning and drift adaptation.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.training.drift_adaptation import (
    MetaAdapterOptimizer,
    StreamingAdaptation,
    DriftDetector
)
from stream_fraudx.training.active_learning import ConformalPredictor, ActiveLearner
from stream_fraudx.data.module import StreamDataModule
from stream_fraudx.utils.metrics import compute_metrics
from stream_fraudx.losses.focal_losses import CombinedFocalLoss


def serve_stream(args):
    """Run streaming adaptation pipeline."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage C] Streaming Adaptation")
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data Module =====
    print("[1/6] Setting up data module...")
    data_module = StreamDataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        microbatch_size=args.microbatch_size,
        label_delay_batches=args.label_delay,
        synthetic_params={
            'num_samples': args.num_samples,
            'num_nodes': args.num_nodes,
            'fraud_rate': args.fraud_rate,
            'num_continuous': args.num_continuous,
            'num_categorical': args.num_categorical,
        }
    )

    data_module.setup()

    # Get model config from dataset
    categorical_vocab_sizes = data_module.get_vocab_sizes()
    num_continuous = data_module.get_continuous_dims()

    print(f"Dataset: {args.dataset}")
    print(f"Micro-batch size: {args.microbatch_size}")
    print(f"Label delay: {args.label_delay} batches")

    # ===== Model =====
    print("\n[2/6] Loading model...")
    config = STREAMFraudXConfig()
    config.continuous_dims = list(range(num_continuous))
    config.categorical_vocab_sizes = categorical_vocab_sizes

    model = STREAMFraudX(
        continuous_dims=config.continuous_dims,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        use_adapters=True
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    model_size = model.get_model_size()
    print(f"Model parameters: {model_size['total']:,}")

    # ===== Streaming Components =====
    print("\n[3/6] Setting up streaming components...")

    # Meta-adapter optimizer
    adapter_params = [p for name, p in model.named_parameters()
                     if 'adapter' in name and p.requires_grad]
    meta_optimizer = MetaAdapterOptimizer(
        adapter_params=adapter_params,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps
    )

    # Loss function
    loss_fn = CombinedFocalLoss()

    # Streaming adaptation manager
    streaming_manager = StreamingAdaptation(
        model=model,
        meta_optimizer=meta_optimizer,
        loss_fn=loss_fn,
        daily_label_budget=args.label_budget,
        microbatch_seconds=args.window_seconds,
        adaptation_frequency=args.adapt_every
    )

    # Active learner
    conformal_predictor = ConformalPredictor(
        miscoverage_rate=0.1,
        calibration_size=1000
    )
    active_learner = ActiveLearner(
        conformal_predictor=conformal_predictor,
        use_business_cost=True,
        diversity_weight=0.1
    )

    # Drift detector
    drift_detector = DriftDetector(
        window_size=100,
        threshold=0.1
    )

    print(f"Meta-LR: {args.meta_lr}, Inner-LR: {args.inner_lr}")
    print(f"Daily label budget: {args.label_budget}")
    print(f"Adaptation frequency: every {args.adapt_every} micro-batches")

    # ===== Streaming Loop =====
    print(f"\n[4/6] Starting streaming loop...")

    stream = data_module.stream_dataloader(
        split='test',
        window_seconds=args.window_seconds
    )

    # Tracking metrics
    all_scores = []
    all_labels = []
    drift_events = []
    adaptation_events = []

    model.eval()

    for microbatch_idx, microbatch in enumerate(tqdm(stream, desc='Streaming')):
        # Move to device
        microbatch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in microbatch.items()}

        # Process micro-batch
        with torch.no_grad():
            scores = torch.sigmoid(streaming_manager.process_microbatch(
                microbatch, active_learner
            ))

        # Store predictions
        if 'labels' in microbatch:
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(microbatch['labels'].cpu().numpy())

            # Update drift detector
            drift_detector.update(scores, microbatch['labels'])

            # Check for drift
            if drift_detector.check_drift():
                print(f"\n  ⚠️  Drift detected at micro-batch {microbatch_idx}")
                drift_events.append({
                    'microbatch': microbatch_idx,
                    'timestamp': microbatch.get('timestamp', 0)
                })

                # Reset adapters to meta-initialization
                meta_optimizer.reset_to_meta()
                drift_detector.reset()

            # Update conformal predictor
            conformal_predictor.calibrate(scores, microbatch['labels'])

        # Log adaptation events
        if microbatch_idx > 0 and microbatch_idx % args.adapt_every == 0:
            adaptation_events.append({
                'microbatch': microbatch_idx,
                'labels_used': streaming_manager.labels_used_today,
                'buffer_size': len(streaming_manager.labeled_buffer)
            })

        # Reset daily budget (simulate day boundary)
        if args.simulate_days and microbatch_idx > 0 and microbatch_idx % args.batches_per_day == 0:
            streaming_manager.reset_daily_budget()
            print(f"\n  📅 Day boundary - budget reset at micro-batch {microbatch_idx}")

    # ===== Evaluation =====
    print("\n[5/6] Computing final metrics...")

    import numpy as np
    final_metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_scores)
    )

    print("\nFinal Streaming Performance:")
    print(f"  AUPRC: {final_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  F1: {final_metrics['f1']:.4f}")

    if 'precision@100' in final_metrics:
        print(f"  Precision@100: {final_metrics['precision@100']:.4f}")

    print(f"\nAdaptation Summary:")
    print(f"  Drift events: {len(drift_events)}")
    print(f"  Adaptation events: {len(adaptation_events)}")
    print(f"  Total labels used: {streaming_manager.labels_used_today}")

    # ===== Save Results =====
    print(f"\n[6/6] Saving results...")

    results = {
        'config': {
            'dataset': args.dataset,
            'microbatch_size': args.microbatch_size,
            'label_delay': args.label_delay,
            'label_budget': args.label_budget,
            'meta_lr': args.meta_lr,
            'inner_lr': args.inner_lr,
            'adapt_every': args.adapt_every,
        },
        'model_size': model_size,
        'final_metrics': final_metrics,
        'drift_events': drift_events,
        'adaptation_summary': {
            'num_drift_events': len(drift_events),
            'num_adaptations': len(adaptation_events),
            'total_labels_used': streaming_manager.labels_used_today,
        }
    }

    with open(output_dir / 'streaming_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save meta-adapter state
    meta_optimizer.save_meta_state(output_dir / 'meta_adapter.pt')

    print(f"\n✓ Streaming adaptation complete!")
    print(f"  Results saved to {output_dir}/streaming_results.json")
    print(f"  Meta-adapter saved to {output_dir}/meta_adapter.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage C: Streaming Adaptation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ieee-cis', 'paysim', 'elliptic'])
    parser.add_argument('--data_dir', type=str, default='data')

    # Synthetic data (if used)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_nodes', type=int, default=1000)
    parser.add_argument('--fraud_rate', type=float, default=0.01)
    parser.add_argument('--num_continuous', type=int, default=10)
    parser.add_argument('--num_categorical', type=int, default=5)

    # Streaming
    parser.add_argument('--microbatch_size', type=int, default=100)
    parser.add_argument('--window_seconds', type=int, default=30)
    parser.add_argument('--label_delay', type=int, default=10)
    parser.add_argument('--label_budget', type=int, default=100)
    parser.add_argument('--batches_per_day', type=int, default=2880,
                       help='Micro-batches per simulated day (default: 2880 for 30s batches)')
    parser.add_argument('--simulate_days', action='store_true', default=False)

    # Meta-learning
    parser.add_argument('--meta_lr', type=float, default=0.01)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--adapt_every', type=int, default=10)

    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/streaming')

    args = parser.parse_args()

    torch.manual_seed(42)
    serve_stream(args)
