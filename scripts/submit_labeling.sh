#!/bin/bash
#SBATCH --job-name=rvs_silver_labeling
#SBATCH --output=scripts/logs/labeling_%A_%a.out
#SBATCH --error=scripts/logs/labeling_%A_%a.err
#SBATCH --array=0-2
#SBATCH --partition=studentbatch    # As per the teacher's guide for batch jobs
#SBATCH --time=1440                 # Max 1 day (1440 minutes) for studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # Good for geometry/spatial math
#SBATCH --mem=16000                 # 16GB RAM (might adjust if files are huge)

# --- 1. Define City Array ---
CITIES=("manhattan" "pittsburgh" "philadelphia")
CITY=${CITIES[$SLURM_ARRAY_TASK_ID]}

echo "🚀 Starting Labeling Pipeline for: $CITY"
echo "📅 Date: $(date)"
echo "📍 Node: $SLURM_NODENAME"

# --- 2. Venv Environment Setup ---
# TAU Note: "Ensure your venv is on your NetApp storage, NOT your home quota!"
VENV_PATH=C:\university\NLP\project_repo\nlp-allocentric-spatial-reasoning\.venv

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated from $VENV_PATH"
else
    echo "❌ Error: venv not found at $VENV_PATH"
    exit 1
fi

# --- 3. Execute ---
# The teacher's guide recommends using 'python' directly or 'srun python'
# for array tasks. We use the --city flag we added to batch_labeling.py.
python scripts/batch_labeling.py --city $CITY

echo "✅ Finished processing $CITY at $(date)"