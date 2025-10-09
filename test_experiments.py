#!/usr/bin/env python
"""Quick test script to run experiments and get actual results."""

from experiments.config import ExperimentConfig
from experiments.driver import ExperimentDriver
import json

def run_baseline_experiment():
    """Run Random Forest baseline."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Random Forest Baseline")
    print("="*80)

    config = ExperimentConfig(
        experiment_name="baseline_rf_test",
        seed=42,
        device="cpu"  # Use CPU for baseline
    )
    config.data.dataset_name = "synthetic"
    config.data.num_samples = 5000
    config.data.batch_size = 64
    config.model.model_type = "random_forest"

    driver = ExperimentDriver(config)
    results = driver.run()

    print("\n✅ Random Forest Results:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AUPRC: {results['auprc']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

    return results

def run_xgboost_experiment():
    """Run XGBoost baseline."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: XGBoost Baseline")
    print("="*80)

    config = ExperimentConfig(
        experiment_name="baseline_xgboost_test",
        seed=42,
        device="cpu"
    )
    config.data.dataset_name = "synthetic"
    config.data.num_samples = 5000
    config.data.batch_size = 64
    config.model.model_type = "xgboost"

    driver = ExperimentDriver(config)
    results = driver.run()

    print("\n✅ XGBoost Results:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AUPRC: {results['auprc']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

    return results

def run_streamfraudx_experiment():
    """Run STREAM-FraudX neural model."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: STREAM-FraudX Neural Model")
    print("="*80)

    config = ExperimentConfig(
        experiment_name="streamfraudx_test",
        seed=42,
        device="cuda"  # Use GPU for neural model
    )
    config.data.dataset_name = "synthetic"
    config.data.num_samples = 5000
    config.data.batch_size = 64
    config.training.max_epochs = 10  # Quick test
    config.training.learning_rate = 1e-3
    config.model.model_type = "stream_fraudx"

    driver = ExperimentDriver(config)
    results = driver.run()

    print("\n✅ STREAM-FraudX Results:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AUPRC: {results['auprc']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

    return results

if __name__ == "__main__":
    print("\n🚀 Running STREAM-FraudX Experiments")
    print("Dataset: Synthetic (5K samples)")
    print("Purpose: Get actual performance numbers\n")

    # Run experiments
    rf_results = run_baseline_experiment()
    xgb_results = run_xgboost_experiment()
    sf_results = run_streamfraudx_experiment()

    # Summary comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"\n{'Model':<25} {'ROC-AUC':<12} {'AUPRC':<12} {'F1':<12}")
    print("-" * 80)
    print(f"{'Random Forest':<25} {rf_results['roc_auc']:<12.4f} {rf_results['auprc']:<12.4f} {rf_results['f1']:<12.4f}")
    print(f"{'XGBoost':<25} {xgb_results['roc_auc']:<12.4f} {xgb_results['auprc']:<12.4f} {xgb_results['f1']:<12.4f}")
    print(f"{'STREAM-FraudX':<25} {sf_results['roc_auc']:<12.4f} {sf_results['auprc']:<12.4f} {sf_results['f1']:<12.4f}")

    # Save results
    all_results = {
        'random_forest': rf_results,
        'xgboost': xgb_results,
        'stream_fraudx': sf_results
    }

    with open('test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to test_results.json")
    print("="*80)
