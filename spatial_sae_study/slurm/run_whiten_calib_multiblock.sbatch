#!/bin/bash
#SBATCH --job-name=whcal_multi
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/whcal_multi_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/whcal_multi_%A_%a.err
set -euo pipefail
BLOCKS=(2 4 6 8 10 11); B=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
ROOT=/n/home12/binxuwang/T-SAE-Spatial
HERE=/n/home12/binxuwang/NanoCodex/groups/dm-with-binxu
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export TORCH_HOME=$STORE/torch_cache
export HF_HOME=$STORE/hf_datasets_cache
export PYTHONPATH=$ROOT/temporal-saes/dictionary_learning:$ROOT/scripts:$HERE:${PYTHONPATH:-}
$PY -u $HERE/whiten_calib_multiblock.py --dino_block $B --n_fit_images 1024 --n_val_images 200 \
 --outdir $STORE/Projects/nhp_sae_analysis/results/whiten_calib_b$B
