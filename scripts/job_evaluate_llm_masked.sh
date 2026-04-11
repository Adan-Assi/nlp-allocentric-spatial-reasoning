#!/bin/bash

#SBATCH --job-name=rvs_mask_eval
#SBATCH --output=logs/mask_eval_%j.out 
#SBATCH --error=logs/mask_eval_%j.err 
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=240                    # Increased to 4 hours (240 mins) for 22k rows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     
#SBATCH --gpus=1                      
#SBATCH --signal=USR1@120             # TAU guideline for clean exits [cite: 56]

# 1. Setup paths
PROJECT_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning"
ANACONDA_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3"

cd $PROJECT_ROOT [cite: 64]

# Ensure the logs directory exists [cite: 60]
mkdir -p logs

# 2. Activate the Anaconda environment
source "$ANACONDA_ROOT/bin/activate"
conda activate nlp_env

# 3. Set Python Path
export PYTHONPATH=$PYTHONPATH:.

# 4. Run the MASKED evaluation
# We point to the new script we created to handle the 22k experimental variants
python scripts/evaluate_llm_masked.py