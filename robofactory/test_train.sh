#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=5
ROOT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory"
DATA_ZARR="/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/"
# 用于后续拼接各个智能体的数据集路径
OUTPUT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/expacp/MANGOBench/hiql"
# OUTPUT_DIR="/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/IHIQL_LiftBarrier"



if [ ! -e "$DATA_ZARR" ]; then
  echo "[ERROR] Training data not found: $DATA_ZARR" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[WARN] Output dir does not exist. Creating: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

cd "$ROOT_DIR"


# 循环
# 循环 01 ~ 03
for EXP_ID in 01 01 02 02 03 03; do
  echo "====== 开始处理实验 ID: $EXP_ID ======"
  # EXP_ID="${EXP_ID:-01}
  SAVE_DIR="${OUTPUT_DIR}${EXP_ID}"
  GOAL_DIR="policy/OGCRL/ogcrl/goal${EXP_ID}"
  # 06测试修改后动作归一化器是否正常保存和加载
  # IHIQL LiftBarrier train and eval
  if [ ! -d "$SAVE_DIR" ]; then
    echo "[WARN] Save dir does not exist. Creating: $SAVE_DIR"
    mkdir -p "$SAVE_DIR"
  fi

  if [ ! -d "$GOAL_DIR" ]; then
    echo "[WARN] Goal dir does not exist. Creating: $GOAL_DIR"
    mkdir -p "$GOAL_DIR"
  fi

  # python policy/OGCRL/train.py \
  #     --config-name=robot_gc_test_train.yaml \
  #     task.name=LiftBarrier-rf \
  #     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
  #     task.dataset.normalize_actions=true \
  #     training.debug=False \
  #     training.seed=100 \
  #     training.device=cuda:3 \
  #     exp_name=LiftBarrier-rf-robot_gc-train \
  #     logging.mode=online \
  #     env_name=robofactory \
  #     agent=hiql \
  #     agent.high_alpha=3.0 \
  #     agent.low_alpha=3.0 \
  #     train_steps=15000 \
  #     log_interval=1000 \
  #     save_interval=15000 \
  #     agent.batch_size=256 \
  #     agent.encoder=impala_small \
  #     agent.low_actor_rep_grad=True \
  #     agent.p_aug=0.5 \
  #     agent.subgoal_steps=10 \
  #     observation=visual \
  #     save_dir="$SAVE_DIR" \
  #     save_goal=True \
  #     save_goal_path="${GOAL_DIR}/LiftBarrier-rf_Agent0_150_Temperal.pkl"


  # python policy/OGCRL/train.py \
  #     --config-name=robot_gc_test_train.yaml \
  #     task.name=LiftBarrier-rf \
  #     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
  #     task.dataset.normalize_actions=true \
  #     training.debug=False \
  #     training.seed=100 \
  #     training.device=cuda:3 \
  #     exp_name=LiftBarrier-rf-robot_gc-train \
  #     logging.mode=online \
  #     env_name=robofactory \
  #     agent=hiql \
  #     agent.high_alpha=3.0 \
  #     agent.low_alpha=3.0 \
  #     train_steps=15000 \
  #     log_interval=1000 \
  #     save_interval=15000 \
  #     agent.batch_size=256 \
  #     agent.encoder=impala_small \
  #     agent.low_actor_rep_grad=True \
  #     agent.p_aug=0.5 \
  #     agent.subgoal_steps=10 \
  #     observation=visual \
  #     save_dir="$SAVE_DIR" \
  #     save_goal=True \
  #     save_goal_path="${GOAL_DIR}/LiftBarrier-rf_Agent1_150_Temperal.pkl"

  echo "====== 完成实验 ID: $EXP_ID ======"
  bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR/MANGOBench/hiql" 15000 "$GOAL_DIR" visual 3.0 3.0  impala_small True 0.5 10 None hiql "$EXP_ID"

done
# 结束
# cd ..
# cd ..
# python run.py --size 16 --gpu 4
