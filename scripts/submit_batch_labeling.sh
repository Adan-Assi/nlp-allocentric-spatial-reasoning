#!/bin/bash

#SBATCH --job-name=rvs_silver_labeling
#SBATCH --output=logs/labeling_%A_%a.out 
#SBATCH --error=logs/labeling_%A_%a.err 
#SBATCH --array=0-2
#SBATCH --partition=studentkillable
#SBATCH --time=1440 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 

# --- 1. Setup Paths ---
PROJECT_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/nlp-allocentric-spatial-reasoning"
ANACONDA_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/adanassi/anaconda3"

cd $PROJECT_ROOT

# Ensure logs directory exists
mkdir -p logs

# --- 2. Activate Environment ---
# Using the specific activation method from the tutorial's template
source "$ANACONDA_ROOT/bin/activate"
conda activate nlp_spatial

# Ensure local 'src' is in the python path
export PYTHONPATH=$PYTHONPATH:.

# --- 3. Define City Array ---
CITIES=("manhattan" "pittsburgh" "philadelphia")
CITY=${CITIES[$SLURM_ARRAY_TASK_ID]}

echo "🚀 Starting Labeling Pipeline for: $CITY"
echo "📅 Date: $(date)"

# --- 4. Execute ---
# The updated script runs here
python scripts/batch_labeling.py --city $CITY

echo "✅ Finished processing $CITY at $(date)"