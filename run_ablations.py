"""
Ablation Study Runner for STREAM-FraudX
Tests different component combinations to measure their contribution.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from stream_fraudx.models.stream_fraudx import STREAMFraudX, STREAMFraudXConfig
from stream_fraudx.training.trainer import STREAMFraudXTrainer
from stream_fraudx.data.module import StreamDataModule
from stream_fraudx.utils.metrics import compute_metrics


def run_single_ablation(config_name: str,
                        data_module: StreamDataModule,
                        model_config: dict,
                        train_config: dict,
                        device: torch.device,
                        output_dir: Path):
    """
    Run a single ablation experiment.

    Args:
        config_name: Name of the configuration
        data_module: Data module
        model_config: Model configuration dict
        train_config: Training configuration dict
        device: Device to use
        output_dir: Output directory

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")
    print(f"Config: {model_config}")

    # Create model
    model = STREAMFraudX(**model_config).to(device)

    model_size = model.get_model_size()
    print(f"Model parameters: {model_size['total']:,}")

    # Create trainer
    trainer = STREAMFraudXTrainer(
        model=model,
        device=device,
        learning_rate=train_config['learning_rate'],
        irm_weight=train_config.get('irm_weight', 0.0)
    )

    # Get dataloaders
    train_loader = data_module.train_dataloader(batch_size=train_config['batch_size'])
    val_loader = data_module.val_dataloader(batch_size=train_config['batch_size'])
    test_loader = data_module.test_dataloader(batch_size=train_config['batch_size'])

    # Training loop
    best_auprc = 0.0
    best_epoch = 0

    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate every N epochs
        if epoch % train_config['eval_every'] == 0:
            val_metrics = trainer.evaluate(val_loader)

            print(f"Epoch {epoch}/{train_config['epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val AUPRC: {val_metrics['auprc']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']
                best_epoch = epoch

                # Save best checkpoint
                checkpoint_path = output_dir / f'{config_name}_best.pt'
                trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)

    # Load best model and evaluate on test set
    best_checkpoint = output_dir / f'{config_name}_best.pt'
    trainer.load_checkpoint(best_checkpoint)
    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTest Results for {config_name}:")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

    return {
        'config_name': config_name,
        'model_config': model_config,
        'model_size': model_size,
        'best_epoch': best_epoch,
        'best_val_auprc': best_auprc,
        'test_metrics': test_metrics
    }


def run_ablations(args):
    """Run all ablation experiments."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Ablation Study for STREAM-FraudX")
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Data Module =====
    print("[1/3] Setting up data module...")
    data_module = StreamDataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        synthetic_params={
            'num_samples': args.num_samples,
            'num_nodes': args.num_nodes,
            'fraud_rate': args.fraud_rate,
            'num_continuous': args.num_continuous,
            'num_categorical': args.num_categorical,
        }
    )
    data_module.setup()

    categorical_vocab_sizes = data_module.get_vocab_sizes()
    continuous_dims = list(range(data_module.get_continuous_dims()))

    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(data_module.train_dataset)}, "
          f"Val: {len(data_module.val_dataset)}, "
          f"Test: {len(data_module.test_dataset)}")

    # ===== Training Config =====
    train_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'eval_every': args.eval_every,
        'irm_weight': args.irm_weight
    }

    # ===== Ablation Configurations =====
    print("\n[2/3] Defining ablation configurations...")

    ablations = []

    # 1. Full model
    ablations.append({
        'name': 'full',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': True,
            'use_tabular': True,
            'use_fusion': True,
            'use_adapters': True,
        }
    })

    # 2. No adapters
    ablations.append({
        'name': 'no_adapters',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': True,
            'use_tabular': True,
            'use_fusion': True,
            'use_adapters': False,
        }
    })

    # 3. Graph only
    ablations.append({
        'name': 'graph_only',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': True,
            'use_tabular': False,
            'use_fusion': False,
            'use_adapters': True,
        }
    })

    # 4. Tabular only
    ablations.append({
        'name': 'tabular_only',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': False,
            'use_tabular': True,
            'use_fusion': False,
            'use_adapters': True,
        }
    })

    # 5. No fusion (separate towers)
    ablations.append({
        'name': 'no_fusion',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': True,
            'use_tabular': True,
            'use_fusion': False,
            'use_adapters': True,
        }
    })

    # 6. Minimal (no graph, no adapters, no fusion)
    ablations.append({
        'name': 'minimal',
        'config': {
            'continuous_dims': continuous_dims,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'use_graph': False,
            'use_tabular': True,
            'use_fusion': False,
            'use_adapters': False,
        }
    })

    print(f"Total ablations: {len(ablations)}")

    # ===== Run Ablations =====
    print("\n[3/3] Running ablation experiments...")

    all_results = []

    for ablation in ablations:
        try:
            result = run_single_ablation(
                config_name=ablation['name'],
                data_module=data_module,
                model_config=ablation['config'],
                train_config=train_config,
                device=device,
                output_dir=output_dir
            )
            all_results.append(result)

        except Exception as e:
            print(f"\n❌ Error in {ablation['name']}: {str(e)}")
            continue

    # ===== Summary =====
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    # Sort by test AUPRC
    all_results.sort(key=lambda x: x['test_metrics']['auprc'], reverse=True)

    print(f"\n{'Configuration':<20} {'Params':<12} {'AUPRC':<8} {'ROC-AUC':<8} {'F1':<8}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['config_name']:<20} "
              f"{result['model_size']['total']:>10,}  "
              f"{result['test_metrics']['auprc']:>6.4f}  "
              f"{result['test_metrics']['roc_auc']:>6.4f}  "
              f"{result['test_metrics']['f1']:>6.4f}")

    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Ablation study complete!")
    print(f"  Results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study for STREAM-FraudX')

    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ieee-cis', 'paysim', 'elliptic'])
    parser.add_argument('--data_dir', type=str, default='data')

    # Synthetic data (if used)
    parser.add_argument('--num_samples', type=int, default=20000)
    parser.add_argument('--num_nodes', type=int, default=1000)
    parser.add_argument('--fraud_rate', type=float, default=0.01)
    parser.add_argument('--num_continuous', type=int, default=10)
    parser.add_argument('--num_categorical', type=int, default=5)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--irm_weight', type=float, default=0.1)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/ablations')

    args = parser.parse_args()

    torch.manual_seed(42)
    run_ablations(args)
