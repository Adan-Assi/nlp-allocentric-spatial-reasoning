#!/bin/bash

#SBATCH --job-name=rvs_eval          # Job name
#SBATCH --output=logs/eval_%j.out    # Standard output log (%j = JobID)
#SBATCH --error=logs/eval_%j.err     # Standard error log
#SBATCH --partition=gpu              # Partition name (check your cluster's names)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Total memory
#SBATCH --time=01:00:00              # Time limit (HH:MM:SS)

# 1. Load modules (Adjust these to your cluster's specific modules)
module load python/3.10
module load cuda/11.8

# 2. Create logs directory if it doesn't exist
mkdir -p logs

# 3. Activate your virtual environment
# Note: Ensure you have a linux-compatible venv or use 'pip install' here
source .venv/bin/activate

# 4. Set PYTHONPATH so it can find config.py
export PYTHONPATH=$PYTHONPATH:.

# 5. Run the evaluation
# We set limit=None to run all 7,000+ rows
# We increase batch_size to 32 because we have a GPU
python scripts/evaluate_llm.py