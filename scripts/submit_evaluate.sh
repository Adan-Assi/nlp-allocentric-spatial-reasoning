#!/bin/bash
#SBATCH --job-name=rvs_mask_eval
#SBATCH --output=logs/mask_eval_%j.out
#SBATCH --error=logs/mask_eval_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

PROJECT_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning"
ANACONDA_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3"

cd $PROJECT_ROOT
mkdir -p logs

source "$ANACONDA_ROOT/bin/activate"
conda activate nlp_spatial

export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=""

echo "🚀 Starting LLM Masked Evaluation"
echo "📅 Date: $(date)"
echo "🖥️  Node: $(hostname)"

python scripts/evaluate_llm_masked.py

echo "✅ Finished at $(date)"
