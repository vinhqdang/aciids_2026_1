"""
Generate comprehensive final results report from experiment JSON output.
Compiles results into markdown format with tables, statistics, and analysis.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np


def format_metric(value, precision=4):
    """Format metric with specified precision."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def generate_report(results_file: str, output_file: str):
    """Generate comprehensive markdown report from results JSON."""

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results found in JSON file!")
        return

    # Group results by dataset
    by_dataset = {}
    for r in results:
        dataset = r.get('dataset', 'Unknown')
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(r)

    # Generate report
    report = []
    report.append("# STREAM-FraudX Experimental Results")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"- **Datasets Evaluated**: {len(by_dataset)}")
    report.append(f"- **Total Experiments**: {len(results)}")
    report.append("")

    # Find STREAM-FraudX results
    streamfraudx_results = [r for r in results if r.get('model') == 'STREAM-FraudX']
    if streamfraudx_results:
        avg_auprc = np.mean([r['auprc'] for r in streamfraudx_results])
        avg_roc = np.mean([r['roc_auc'] for r in streamfraudx_results])
        report.append(f"### STREAM-FraudX Performance")
        report.append(f"- **Average AUPRC**: {avg_auprc:.4f}")
        report.append(f"- **Average ROC-AUC**: {avg_roc:.4f}")
        report.append("")

    report.append("---")
    report.append("")

    # Results by Dataset
    for dataset_name, dataset_results in sorted(by_dataset.items()):
        report.append(f"## Dataset: {dataset_name}")
        report.append("")

        # Dataset statistics
        report.append("### Results Summary")
        report.append("")
        report.append("| Model | AUPRC | ROC-AUC | Precision | Recall | F1 | Training Time (s) |")
        report.append("|-------|-------|---------|-----------|--------|----|--------------------|")

        for r in sorted(dataset_results, key=lambda x: x.get('auprc', 0), reverse=True):
            model = r.get('model', 'Unknown')
            auprc = format_metric(r.get('auprc', 0))
            roc_auc = format_metric(r.get('roc_auc', 0))
            precision = format_metric(r.get('precision', 0))
            recall = format_metric(r.get('recall', 0))
            f1 = format_metric(r.get('f1', 0))
            train_time = format_metric(r.get('training_time', 0), precision=2)

            report.append(f"| {model} | {auprc} | {roc_auc} | {precision} | {recall} | {f1} | {train_time} |")

        report.append("")

        # Best model
        best = max(dataset_results, key=lambda x: x.get('auprc', 0))
        report.append(f"**Best Model**: {best.get('model')} (AUPRC: {best.get('auprc', 0):.4f})")
        report.append("")

        # Comparison
        if len(dataset_results) > 1:
            streamfraudx = next((r for r in dataset_results if r.get('model') == 'STREAM-FraudX'), None)
            baselines = [r for r in dataset_results if r.get('model') != 'STREAM-FraudX']

            if streamfraudx and baselines:
                best_baseline = max(baselines, key=lambda x: x.get('auprc', 0))
                improvement = streamfraudx.get('auprc', 0) - best_baseline.get('auprc', 0)

                report.append("### STREAM-FraudX vs Best Baseline")
                report.append("")
                report.append(f"- **Best Baseline**: {best_baseline.get('model')} (AUPRC: {best_baseline.get('auprc', 0):.4f})")
                report.append(f"- **STREAM-FraudX**: AUPRC: {streamfraudx.get('auprc', 0):.4f}")
                report.append(f"- **Improvement**: {improvement:+.4f} ({improvement/best_baseline.get('auprc', 1)*100:+.1f}%)")
                report.append("")

        report.append("---")
        report.append("")

    # Overall Comparison
    report.append("## Overall Model Comparison")
    report.append("")

    # Group by model across all datasets
    by_model = {}
    for r in results:
        model = r.get('model', 'Unknown')
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    report.append("### Average Performance Across All Datasets")
    report.append("")
    report.append("| Model | Avg AUPRC | Avg ROC-AUC | Avg F1 | Datasets |")
    report.append("|-------|-----------|-------------|--------|----------|")

    for model, model_results in sorted(by_model.items(), key=lambda x: np.mean([r.get('auprc', 0) for r in x[1]]), reverse=True):
        avg_auprc = np.mean([r.get('auprc', 0) for r in model_results])
        avg_roc = np.mean([r.get('roc_auc', 0) for r in model_results])
        avg_f1 = np.mean([r.get('f1', 0) for r in model_results])
        num_datasets = len(model_results)

        report.append(f"| {model} | {avg_auprc:.4f} | {avg_roc:.4f} | {avg_f1:.4f} | {num_datasets} |")

    report.append("")
    report.append("---")
    report.append("")

    # Key Findings
    report.append("## Key Findings")
    report.append("")

    # Find best performers
    if streamfraudx_results:
        wins = sum(1 for dataset in by_dataset.values()
                   if max(dataset, key=lambda x: x.get('auprc', 0)).get('model') == 'STREAM-FraudX')

        report.append(f"1. **STREAM-FraudX won {wins}/{len(by_dataset)} datasets** in terms of AUPRC")
        report.append("")

        # Calculate average improvements
        improvements = []
        for dataset_results in by_dataset.values():
            sf = next((r for r in dataset_results if r.get('model') == 'STREAM-FraudX'), None)
            baselines = [r for r in dataset_results if r.get('model') != 'STREAM-FraudX']
            if sf and baselines:
                best_baseline = max(baselines, key=lambda x: x.get('auprc', 0))
                improvement = sf.get('auprc', 0) - best_baseline.get('auprc', 0)
                improvements.append(improvement)

        if improvements:
            avg_improvement = np.mean(improvements)
            report.append(f"2. **Average improvement over baselines**: {avg_improvement:+.4f} AUPRC points")
            report.append("")

    # Training efficiency
    report.append("3. **Training Efficiency**:")
    report.append("")
    for model, model_results in sorted(by_model.items()):
        avg_time = np.mean([r.get('training_time', 0) for r in model_results])
        report.append(f"   - {model}: {avg_time:.2f}s average")
    report.append("")

    report.append("---")
    report.append("")

    # Methodology
    report.append("## Methodology")
    report.append("")
    report.append("### Models Evaluated")
    report.append("")
    report.append("1. **STREAM-FraudX**: Dual-tower architecture (Temporal Graph + Tabular Transformer)")
    report.append("   - Graph tower: TGN-style temporal graph neural network")
    report.append("   - Tabular tower: Feature tokenization + transformer")
    report.append("   - Fusion: Gated cross-attention")
    report.append("")
    report.append("2. **RandomForest**: Sklearn ensemble baseline")
    report.append("3. **LogisticRegression**: Linear baseline")
    report.append("")

    report.append("### Datasets")
    report.append("")
    for dataset_name in sorted(by_dataset.keys()):
        report.append(f"- **{dataset_name}**: ")
        if dataset_name == 'IEEE-CIS':
            report.append("  590K transactions, 3.5% fraud rate, rich categorical features")
        elif dataset_name == 'PaySim':
            report.append("  6.3M transactions, 0.13% fraud rate, temporal streaming")
        elif dataset_name == 'Elliptic':
            report.append("  203K nodes, 21% fraud rate (labeled), Bitcoin transaction graph")
        elif dataset_name == 'Synthetic':
            report.append("  5K transactions, 5% fraud rate, validation dataset")
        report.append("")

    report.append("### Metrics")
    report.append("")
    report.append("- **AUPRC**: Area Under Precision-Recall Curve (primary metric)")
    report.append("- **ROC-AUC**: Area Under ROC Curve")
    report.append("- **Precision/Recall/F1**: Classification metrics")
    report.append("")

    report.append("### Experimental Setup")
    report.append("")
    report.append("- **Split**: 70% train, 15% validation, 15% test")
    report.append("- **Batch size**: 128-256")
    report.append("- **Epochs**: 20-30")
    report.append("- **Loss**: Asymmetric Focal Loss (handling imbalance)")
    report.append("- **Optimizer**: AdamW with gradient clipping")
    report.append("")

    report.append("---")
    report.append("")

    # Conclusion
    report.append("## Conclusion")
    report.append("")

    if streamfraudx_results:
        avg_auprc = np.mean([r['auprc'] for r in streamfraudx_results])
        if avg_auprc > 0.75:
            report.append(f"✅ **STREAM-FraudX achieves strong performance** with {avg_auprc:.4f} average AUPRC across all datasets.")
        else:
            report.append(f"⚠️ **STREAM-FraudX shows moderate performance** with {avg_auprc:.4f} average AUPRC.")
        report.append("")

    report.append("### Strengths")
    report.append("")
    report.append("- Dual-tower architecture captures both graph and tabular patterns")
    report.append("- Temporal graph tower handles evolving transaction networks")
    report.append("- Transformer tower processes rich categorical features")
    report.append("")

    report.append("### Next Steps")
    report.append("")
    report.append("1. **Hyperparameter tuning**: Optimize learning rates, batch sizes, architecture dims")
    report.append("2. **Ablation studies**: Test individual components (graph tower, tabular tower, fusion)")
    report.append("3. **Pretraining**: Apply self-supervised pretraining (MEM + InfoNCE)")
    report.append("4. **Adaptation**: Enable parameter-efficient adapters for drift handling")
    report.append("5. **Active learning**: Integrate conformal prediction for label efficiency")
    report.append("")

    report.append("---")
    report.append("")
    report.append(f"*Report generated by STREAM-FraudX pipeline on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Report generated: {output_file}")
    print(f"  - {len(results)} experiments")
    print(f"  - {len(by_dataset)} datasets")
    print(f"  - {len(by_model)} models")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate final results report')
    parser.add_argument('--input', type=str, default='results_experiment.json',
                       help='Input JSON file with experiment results')
    parser.add_argument('--output', type=str, default='RESULTS_FINAL.md',
                       help='Output markdown file')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("\nRun experiments first:")
        print("  python run_all_experiments.py")
        exit(1)

    generate_report(args.input, args.output)
