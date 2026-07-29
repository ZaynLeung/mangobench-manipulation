#!/usr/bin/env bash

python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=DEBUG_TASK \
  task.dataset.zarr_path=PATH_TO_DEBUG_ZARR \
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
  save_dir=PATH_TO_DEBUG_SAVE \
  save_goal=True \
  save_goal_path=PATH_TO_DEBUG_GOAL