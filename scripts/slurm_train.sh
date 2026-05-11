#!/bin/bash
#SBATCH --job-name=ppg2ecg
#SBATCH --partition=engineering
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j-%x.out
#SBATCH --error=results/slurm-%j-%x.err

# Usage:
#   sbatch scripts/slurm_train.sh configs/unet.yaml
#   sbatch scripts/slurm_train.sh configs/bilstm.yaml
#   sbatch scripts/slurm_train.sh configs/transformer.yaml

set -e

CONFIG_PATH="${1:-configs/unet.yaml}"

echo "===================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Start: $(date)"
echo "===================="

# Activate conda env (assumes conda is initialized in your .bashrc)
source ~/.bashrc
conda activate ppg2ecg

# GPU check
nvidia-smi

# Train
cd $SLURM_SUBMIT_DIR
python -m src.train --config "$CONFIG_PATH"

# Evaluate immediately after training
MODEL_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['model']['name'])")
python -m src.evaluate --model "$MODEL_NAME"

echo "===================="
echo "End: $(date)"
echo "===================="
