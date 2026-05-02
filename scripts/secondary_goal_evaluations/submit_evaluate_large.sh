#!/bin/bash
#SBATCH --job-name=rvs_large
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mask_eval_large_%j.out
#SBATCH --error=logs/mask_eval_large_%j.err

echo "🚀 Starting LLM Masked Evaluation — FLAN-T5-large"
echo "📅 Date: $(date)"
echo "🖥️  Node: $(hostname)"

# Environment — use persistent storage path per cluster guidelines
source /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3/etc/profile.d/conda.sh
conda activate nlp_spatial

cd /vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning

# Force CPU — avoid CUDA version mismatch on studentkillable nodes
export CUDA_VISIBLE_DEVICES=""

# HuggingFace cache to persistent storage per cluster guidelines
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

mkdir -p logs

python scripts/secondary_goal_evaluations/evaluate_llm_large.py

echo "✅ Finished at $(date)"