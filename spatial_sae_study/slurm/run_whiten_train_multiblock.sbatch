#!/bin/bash
#SBATCH --job-name=whtrain_multi
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=/n/home12/binxuwang/T-SAE-Spatial/logs/whtrain_multi_%A_%a.out
#SBATCH --error=/n/home12/binxuwang/T-SAE-Spatial/logs/whtrain_multi_%A_%a.err
set -euo pipefail
BLOCKS=(2 4 6 8 10 11); B=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
ROOT=/n/home12/binxuwang/T-SAE-Spatial
STORE=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang
PROJ=$STORE/Projects/T-SAE-Spatial
RES=$STORE/Projects/nhp_sae_analysis/results
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
DATA=$PROJ/data/imagenet_data/train-00000-of-00001-18bc3231d015f1e8.parquet
DINO=$STORE/torch_cache/hub/facebookresearch_dinov2_main
STATS=$RES/whiten_calib_b$B/whiten_stats.npz
CSV=$RES/whiten_calib_b$B/whiten_calib.csv
export TORCH_HOME=$STORE/torch_cache
export PYTHONPATH=$ROOT/temporal-saes/dictionary_learning:$ROOT/scripts:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd $ROOT
test -f "$STATS"; test -f "$CSV"
SCALE=$($PY -c "import pandas as pd; d=pd.read_csv('$CSV'); print(float(d.loc[d.variant=='diag','input_scale_to_ref'].iloc[0]))")
COMMON="--data_path $DATA --dino_repo_path $DINO --dino_block $B --dict_size 16384 --k 32 --group_fractions 0.2 0.8 --temp_alpha 100 --steps 50000 --seed 0 --save_every 25000 --log_every 500 --channel_transform diag --channel_stats $STATS --input_scale $SCALE"
echo "block=$B scale=$SCALE GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
for ARM in spatial baseline; do
 CW=1.0; [ "$ARM" = baseline ] && CW=0.0
 OUT=$PROJ/dino_b${B}_whiten_${ARM}_s0
 mkdir -p "$OUT"
 $PY -u scripts/new_spatial_train_triplet.py $COMMON --contrastive_weight $CW --save_dir "$OUT"
done
