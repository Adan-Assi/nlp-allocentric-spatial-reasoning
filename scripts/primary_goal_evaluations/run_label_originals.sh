#!/bin/bash
#SBATCH --job-name=label_originals
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/label_originals_%j.out
#SBATCH --error=logs/label_originals_%j.err

echo "Starting original instruction labeling"
echo "Date: $(date)"
echo "Node: $(hostname)"

source /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3/etc/profile.d/conda.sh
conda activate nlp_spatial

cd /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning

export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

mkdir -p logs

python scripts/primary_goal_evaluations/label_originals.py

echo "Finished at $(date)"