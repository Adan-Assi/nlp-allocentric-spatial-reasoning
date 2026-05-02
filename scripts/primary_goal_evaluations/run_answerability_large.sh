#!/bin/bash
#SBATCH --job-name=rvs_cls_large
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --output=logs/answerability_large_%j.out
#SBATCH --error=logs/answerability_large_%j.err

echo "Starting Answerability Classification — FLAN-T5-large"
echo "Date: $(date)"
echo "Node: $(hostname)"

source /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3/etc/profile.d/conda.sh
conda activate nlp_spatial

cd /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning

export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

mkdir -p logs

python scripts/primary_goal_evaluations/evaluate_answerability_large.py

echo "Finished at $(date)"