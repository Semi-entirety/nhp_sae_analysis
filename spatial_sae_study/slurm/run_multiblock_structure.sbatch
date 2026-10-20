#!/bin/bash
#SBATCH --job-name=multiblock_struct
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/multiblock_struct_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/multiblock_struct_%A_%a.err
set -euo pipefail
ROOT=/n/home12/binxuwang/T-SAE-Spatial
HERE=/n/home12/binxuwang/NanoCodex/groups/dm-with-binxu
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang
RES=$STORE/Projects/nhp_sae_analysis/results
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export TORCH_HOME=$STORE/torch_cache
export HF_HOME=$STORE/hf_datasets_cache
export PYTHONPATH=$ROOT/temporal-saes/dictionary_learning:$ROOT/scripts:$HERE:${PYTHONPATH:-}
CONDS='dino_b0_postLN dino_b2_postLN dino_b4_postLN dino_b6_postLN dino_b8_postLN dino_b10_postLN dino_b11_postLN'
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
 $PY -u $HERE/shortcut_analysis_multiblock.py --conditions $CONDS --n_images 200 --outdir $RES/shortcut_multiblock_s0
else
 $PY -u $HERE/pca_curve_multiblock.py --conds $CONDS --n_images 150 --n_pcs 256 --ks 5 10 20 30 40 50 64 100 128 200 256 --outdir $RES/pca_curve_multiblock_s0
fi
