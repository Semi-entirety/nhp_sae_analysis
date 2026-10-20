#!/bin/bash
#SBATCH --job-name=fvu_multi
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/fvu_multi_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/fvu_multi_%A_%a.err
set -euo pipefail
B=(0 2 4 6 8 10 11); BLOCK=${B[$SLURM_ARRAY_TASK_ID]}
ROOT=/n/home12/binxuwang/T-SAE-Spatial; HERE=/n/home12/binxuwang/NanoCodex/groups/dm-with-binxu
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export TORCH_HOME=$STORE/torch_cache; export HF_HOME=$STORE/hf_datasets_cache
export PYTHONPATH=$ROOT/scripts:$ROOT/temporal-saes/dictionary_learning:${PYTHONPATH:-}
$PY -u $HERE/eval_fullcode_fvu_multiblock.py --block $BLOCK --n_images 200 --outdir $STORE/Projects/nhp_sae_analysis/results/fullcode_fvu_multiblock_s0
