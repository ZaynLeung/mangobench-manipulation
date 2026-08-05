#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
ROOT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory"
DATA_ZARR="/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/"
# 用于后续拼接各个智能体的数据集路径
OUTPUT_DIR="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/expacp/MANGOBench/hiql"
# OUTPUT_DIR="/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/IHIQL_LiftBarrier"

# GOAL_PATH="policy/OGCRL/ogcrl/goal"
EXP_ID="${EXP_ID:-06}"
SAVE_DIR="${OUTPUT_DIR}${EXP_ID}"
GOAL_DIR="policy/OGCRL/ogcrl/goal${EXP_ID}"

if [ ! -e "$DATA_ZARR" ]; then
  echo "[ERROR] Training data not found: $DATA_ZARR" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[WARN] Output dir does not exist. Creating: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

cd "$ROOT_DIR"

SAVE_DIR_01="${OUTPUT_DIR}01"
SAVE_DIR_02="${OUTPUT_DIR}02"
SAVE_DIR_03="${OUTPUT_DIR}03"
SAVE_DIR_04="${OUTPUT_DIR}04"
SAVE_DIR_05="${OUTPUT_DIR}05"
SAVE_DIR_06="${OUTPUT_DIR}06"

mkdir -p "$SAVE_DIR_01" "$SAVE_DIR_02" "$SAVE_DIR_03" "$SAVE_DIR_04" "$SAVE_DIR_05" "$SAVE_DIR_06"

# 01
# IHIQL LiftBarrier train and eval
# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_01" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent0_150_Temperal.pkl"


# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_01" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent1_150_Temperal.pkl"

# bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR_01/MANGOBench/hiql" 15000 /home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql
# 提前终止
# exit 0
# 02
#IHIQL LiftBarrier train and eval
# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_02" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent0_150_Temperal.pkl"


# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_02" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent1_150_Temperal.pkl"

# bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR_02/MANGOBench/hiql" 15000 /home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql

# 03
#IHIQL LiftBarrier train and eval
# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_03" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent0_150_Temperal.pkl"


# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_03" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal/LiftBarrier-rf_Agent1_150_Temperal.pkl"

# bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR_03/MANGOBench/hiql" 15000 policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql


# 04
# # IHIQL LiftBarrier train and eval
# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_04" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal04/LiftBarrier-rf_Agent0_150_Temperal.pkl"


# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:4 \
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
#     save_dir="$SAVE_DIR_04" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal04/LiftBarrier-rf_Agent1_150_Temperal.pkl"

# bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR_04/MANGOBench/hiql" 15000 policy/OGCRL/ogcrl/goal04/ visual 3.0 3.0  impala_small True 0.5 10 None hiql

# # 05
# # IHIQL LiftBarrier train and eval
# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:6 \
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
#     save_dir="$SAVE_DIR_05" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal05/LiftBarrier-rf_Agent0_150_Temperal.pkl"


# python policy/OGCRL/train.py \
#     --config-name=robot_gc_test_train.yaml \
#     task.name=LiftBarrier-rf \
#     task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
#     training.debug=False \
#     training.seed=100 \
#     training.device=cuda:6 \
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
#     save_dir="$SAVE_DIR_05" \
#     save_goal=True \
#     save_goal_path="/home/wangyi/Liangziyan/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/goal05/LiftBarrier-rf_Agent1_150_Temperal.pkl"

# bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR_05/MANGOBench/hiql" 15000 policy/OGCRL/ogcrl/goal05/ visual 3.0 3.0  impala_small True 0.5 10 None hiql

# 06测试修改后动作归一化器是否正常保存和加载
# IHIQL LiftBarrier train and eval
python policy/OGCRL/train.py \
    --config-name=robot_gc_test_train.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent0_150.zarr" \
    +task.dataset.normalize_actions=true \
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
    save_dir="$SAVE_DIR" \
    save_goal=True \
    save_goal_path="${GOAL_DIR}/LiftBarrier-rf_Agent0_150_Temperal.pkl"


python policy/OGCRL/train.py \
    --config-name=robot_gc_test_train.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path="${DATA_ZARR}LiftBarrier-rf_Agent1_150.zarr" \
    +task.dataset.normalize_actions=true \
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
    save_dir="$SAVE_DIR" \
    save_goal=True \
    save_goal_path="${GOAL_DIR}/LiftBarrier-rf_Agent1_150_Temperal.pkl"

bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml "$SAVE_DIR/MANGOBench/hiql" 15000 "$GOAL_DIR" visual 3.0 3.0  impala_small True 0.5 10 None hiql "$EXP_ID"

# 结束
cd ..
cd ..
python run.py --size 16 --gpu 0
