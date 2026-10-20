#!/bin/bash
#SBATCH --job-name=4m_grp_smooth
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_fellow_binxuwang
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/4m_grp_smooth_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/4m_grp_smooth_%A_%a.err
set -euo pipefail
source /n/home12/binxuwang/.bashrc
conda activate torch2
B=$([ "$SLURM_ARRAY_TASK_ID" = 0 ] && echo 8 || echo 11)
OUT=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results/fourmodel_group_smoothness/b${B}
cd /n/home12/binxuwang/T-SAE-Spatial
PYTHONNOUSERSITE=1 python -u /n/home12/binxuwang/NanoCodex/groups/dm-with-binxu/fourmodel_group_smoothness.py --block "$B" --n-images 200 --min-active-images 10 --outdir "$OUT"
