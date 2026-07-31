#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory"
DATA_ZARR="/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/"
# 用于后续拼接各个智能体的数据集路径
OUTPUT_DIR="/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/IHIQL_LiftBarrier"
GOAL_PATH="${OUTPUT_DIR}/LiftBarrier_goals.pkl"

if [ ! -e "$DATA_ZARR" ]; then
  echo "[ERROR] Training data not found: $DATA_ZARR" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[WARN] Output dir does not exist. Creating: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

cd "$ROOT_DIR"

#IHIQL LiftBarrier train and eval
python policy/OGCRL/train.py \
    --config-name=robot_gc_test_train.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=LiftBarrier-rf-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=hiql \
    agent.high_alpha=3.0 \
    agent.low_alpha=3.0 \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.low_actor_rep_grad=True \
    agent.p_aug=0.5 \
    agent.subgoal_steps=10 \
    observation=visual \
    save_dir=expacp 

python policy/OGCRL/train.py \
    --config-name=robot_gc_test_train.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=LiftBarrier-rf-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=hiql \
    agent.high_alpha=3.0 \
    agent.low_alpha=3.0 \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.low_actor_rep_grad=True \
    agent.p_aug=0.5 \
    agent.subgoal_steps=10 \
    observation=visual \
    save_dir=expacp 

bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml expacp/MANGOBench/hiql 15000 policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql
