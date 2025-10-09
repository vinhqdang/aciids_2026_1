#!/bin/bash
# STREAM-FraudX: Single-command execution script
# Runs complete experiment pipeline: Stage A (pretraining) -> Stage B (training) -> Stage C (streaming)

set -e  # Exit on error

echo "========================================="
echo "STREAM-FraudX Experiment Pipeline"
echo "========================================="
echo ""

# Check conda environment
if ! conda env list | grep -q "py310"; then
    echo "Creating conda environment py310..."
    conda create -n py310 python=3.10 -y
fi

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py310

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data artifacts/runs artifacts/preprocessing artifacts/reports checkpoints outputs

# Run experiments
echo ""
echo "========================================="
echo "Stage B: Supervised Training"
echo "========================================="
echo ""

# Run baseline experiments
echo "[1/3] Running baseline models..."
python -m experiments.driver \
    --experiment_name "baseline_rf" \
    --model_type "random_forest" \
    --dataset "synthetic" \
    --num_samples 10000 \
    --seed 42

python -m experiments.driver \
    --experiment_name "baseline_xgboost" \
    --model_type "xgboost" \
    --dataset "synthetic" \
    --num_samples 10000 \
    --seed 42

# Run STREAM-FraudX neural model
echo ""
echo "[2/3] Running STREAM-FraudX (enhanced architecture)..."
python -m experiments.driver \
    --experiment_name "streamfraudx_v2" \
    --model_type "stream_fraudx" \
    --dataset "synthetic" \
    --num_samples 10000 \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 42

echo ""
echo "[3/3] Generating final report..."
python generate_final_report.py

echo ""
echo "========================================="
echo "Experiment Pipeline Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - artifacts/runs/<run_id>/metrics.{json,csv}"
echo "  - artifacts/reports/"
echo ""
echo "View experiment logs:"
echo "  python -c 'from experiments.logger import ExperimentLogger; print(ExperimentLogger.list_runs())'"
echo ""
