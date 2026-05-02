#!/bin/bash
#SBATCH --job-name=nearest_poi_baseline
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=s-004
#SBATCH --output=logs/nearest_poi_baseline_%j.out
#SBATCH --error=logs/nearest_poi_baseline_%j.err

echo "Starting Nearest-POI Baseline"
echo "Date: $(date)"
echo "Node: $(hostname)"

source /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3/etc/profile.d/conda.sh
conda activate nlp_spatial

cd /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning

export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

mkdir -p logs

python scripts/primary_goal_evaluations/nearest_poi_baseline.py

echo "Finished at $(date)"