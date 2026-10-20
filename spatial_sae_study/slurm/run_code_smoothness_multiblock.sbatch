#!/bin/bash
#SBATCH --job-name=code_smooth
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/code_smooth_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/code_smooth_%A_%a.err
set -euo pipefail
BLOCKS=(0 2 4 6 8 10 11)
B=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
HERE=/n/home12/binxuwang/NanoCodex/groups/dm-with-binxu
REPO=/n/home12/binxuwang/T-SAE-Spatial
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/T-SAE-Spatial
RES=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export TORCH_HOME=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache
export HF_HOME=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_datasets_cache
export PYTHONPATH=$REPO/temporal-saes/dictionary_learning:$REPO/scripts:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
if [ "$B" -eq 2 ]; then
  SP=$STORE/dino_b2_postLN_spatial_s0/ae_latest.pt
  BL=$STORE/dino_b2_postLN_baseline_s0/ae_latest.pt
else
  SP=$STORE/spatial_block$B/ae_latest.pt
  BL=$STORE/spatial_block${B}_baseline/ae_latest.pt
fi
$PY -u $HERE/code_smoothness_same_vs_across.py \
  --ckpt spatial:$SP --ckpt baseline:$BL --dino_block $B \
  --n_images 200 --null_reps 10 --n_boot 5000 --n_perm 100000 --device cuda \
  --outdir $RES/code_smoothness_b${B}_s0
