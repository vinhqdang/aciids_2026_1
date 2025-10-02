#!/bin/bash
################################################################################
# STREAM-FraudX: Complete Experimental Pipeline
# One command to: setup -> download data -> run experiments -> generate results
################################################################################

set -e  # Exit on error

OUTPUT_FILE="RESULTS_FINAL.md"
LOG_FILE="run_all.log"

echo "================================================================================"
echo "STREAM-FraudX: Complete Experimental Pipeline"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Setup environment (install dependencies)"
echo "  2. Download real fraud detection datasets (IEEE-CIS, PaySim, Elliptic)"
echo "  3. Run experiments on all datasets"
echo "  4. Generate comprehensive results report"
echo ""
echo "Output will be written to: $OUTPUT_FILE"
echo "Logs will be written to: $LOG_FILE"
echo ""
echo "Estimated time: 2-4 hours (depending on download speed)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "[$(date)] Starting STREAM-FraudX pipeline..."
echo ""

################################################################################
# Step 1: Environment Setup
################################################################################
echo "================================================================================"
echo "[1/4] Environment Setup"
echo "================================================================================"

# Activate conda environment
echo "[$(date)] Activating py310 environment..."
conda activate py310 || { echo "Failed to activate py310"; exit 1; }

# Install dependencies
echo "[$(date)] Installing dependencies..."
pip install -q torch numpy scikit-learn pandas tqdm kaggle 2>&1 | tail -5

echo "[$(date)] ✓ Environment ready"
echo ""

################################################################################
# Step 2: Download Datasets
################################################################################
echo "================================================================================"
echo "[2/4] Downloading Real Datasets"
echo "================================================================================"

# Check Kaggle API setup
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API not configured!"
    echo ""
    echo "Please setup Kaggle credentials:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create API token (downloads kaggle.json)"
    echo "  3. Place in ~/.kaggle/kaggle.json"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "[$(date)] Downloading datasets (this may take 30-60 minutes)..."
python download_datasets.py 2>&1 | grep -E "(Downloading|Extracting|✓|✗|ERROR)"

# Check if downloads succeeded
if [ -f data/paysim/PS_20174392719_1491204439457_log.csv ]; then
    echo "[$(date)] ✓ PaySim dataset ready"
else
    echo "[$(date)] ⚠ PaySim download may have failed, will use synthetic data as fallback"
fi

echo ""

################################################################################
# Step 3: Run Experiments
################################################################################
echo "================================================================================"
echo "[3/4] Running Experiments"
echo "================================================================================"

echo "[$(date)] Running experiments on all datasets..."
echo "This will take 1-3 hours depending on hardware..."
echo ""

# Run main experiments
python run_all_experiments.py --output results_experiment.json 2>&1

echo ""
echo "[$(date)] ✓ Experiments completed"
echo ""

################################################################################
# Step 4: Generate Report
################################################################################
echo "================================================================================"
echo "[4/4] Generating Results Report"
echo "================================================================================"

echo "[$(date)] Compiling comprehensive results..."
python generate_final_report.py --output "$OUTPUT_FILE"

echo ""
echo "[$(date)] ✓ Report generated: $OUTPUT_FILE"
echo ""

################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Detailed report: $OUTPUT_FILE"
echo "  - Raw results: results_experiment.json"
echo "  - Full logs: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review $OUTPUT_FILE for comprehensive analysis"
echo "  2. Check specific metrics in results_experiment.json"
echo "  3. Run ablation studies: python run_ablations.py"
echo ""
echo "[$(date)] Pipeline finished successfully!"
