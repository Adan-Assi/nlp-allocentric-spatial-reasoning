# ---------- MANHATTAN ONLY ----------
#!/bin/bash
#SBATCH --job-name=rvs_label_variants
#SBATCH --output=logs/label_variants_%j.out
#SBATCH --error=logs/label_variants_%j.err
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

echo "🚀 Starting Oracle 2 labeling for manhattan"
echo "📅 Date: $(date)"
echo "🖥️  Node: $(hostname)"

python scripts/label_variants.py --city manhattan

echo "✅ Finished at $(date)"
