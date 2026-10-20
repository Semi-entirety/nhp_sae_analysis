#!/bin/bash
#SBATCH --job-name=wheval_multi
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/wheval_multi_%j.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/wheval_multi_%j.err
set -euo pipefail
ROOT=/n/home12/binxuwang/T-SAE-Spatial
HERE=/n/home12/binxuwang/NanoCodex/groups/dm-with-binxu
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export TORCH_HOME=$STORE/torch_cache
export HF_HOME=$STORE/hf_datasets_cache
export PYTHONPATH=$ROOT/temporal-saes/dictionary_learning:$ROOT/scripts:$HERE:${PYTHONPATH:-}
CONDS='dino_b0_whiten dino_b2_whiten dino_b4_whiten dino_b6_whiten dino_b8_whiten dino_b10_whiten dino_b11_whiten'
$PY -u $HERE/shortcut_analysis_whiten_multiblock.py --conditions $CONDS --n_images 200 \
 --outdir $STORE/Projects/nhp_sae_analysis/results/whiten_shortcut_multiblock_s0
