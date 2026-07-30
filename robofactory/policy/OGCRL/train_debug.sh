#!/usr/bin/env bash
set -euo pipefail

export JAX_PLATFORMS=cuda
ROOT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory"
DATA_ZARR="/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr"
OUTPUT_DIR="/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/debug_ogcrl"
GOAL_PATH="${OUTPUT_DIR}/debug_goals.pkl"

if [ ! -e "$DATA_ZARR" ]; then
  echo "[ERROR] Training data not found: $DATA_ZARR" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[WARN] Output dir does not exist. Creating: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

cd "$ROOT_DIR"

python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=LiftBarrier-rf \
  task.dataset.zarr_path="$DATA_ZARR" \
  training.debug=True \
  training.seed=100 \
  training.device=cuda:0 \
  exp_name=DEBUG_OGCRL \
  logging.mode=offline \
  env_name=robofactory \
  agent=debug \
  train_steps=1 \
  log_interval=1 \
  save_interval=1 \
  agent.batch_size=16 \
  agent.encoder=impala_small \
  observation=visual \
  save_dir="$OUTPUT_DIR" \
  save_goal=True \
  save_goal_path="$GOAL_PATH"