# MangoBench-Manipulation 项目概览

## 项目简介

MangoBench 是一个多智能体目标驱动离线强化学习基准（CVPR 2026），覆盖 47 个任务。本仓库为 **manipulation（双臂操作）** 部分，基于 RoboFactory 扩展，实现多机器人协作的双臂操作任务。

- 仓库地址: https://github.com/SYSU-SAIL/mangobench-manipulation.git
- 运动学部分: https://github.com/WendyeeWang/mangobench-locomotion

## 技术栈

- Python >= 3.9, PyTorch 2.6, JAX, Flax
- ManiSkill 3.0 (mani_skill==3.0.0b12) + SAPIEN (物理仿真)
- Hydra (配置管理), wandb (实验跟踪)
- zarr (数据集存储), diffusers (扩散策略)

## 目录结构

```
mangobench-manipulation/
├── setup.py                        # 包安装（robofactory）
├── robofactory/
│   ├── __init__.py                 # 导出 tasks + planner
│   ├── configs/                    # 任务配置文件（YAML）
│   │   ├── robocasa/               # RoboCasa 场景配置
│   │   └── table/                  # 桌面场景配置
│   ├── tasks/                      # 任务环境定义（11个任务）
│   │   ├── lift_barrier.py         # LiftBarrierEnv
│   │   ├── place_food.py           # PlaceFoodEnv
│   │   ├── stack_cube.py           # StackCubeEnv
│   │   ├── two_robots_stack_cube.py
│   │   ├── three_robots_stack_cube.py
│   │   ├── pass_shoe.py, pick_meat.py, strike_cube.py
│   │   ├── camera_alignment.py, take_photo.py
│   │   └── long_pipeline_delivery.py
│   ├── planner/                    # 运动规划器（数据采集）
│   │   ├── run.py                  # 数据采集主入口
│   │   ├── motionplanner.py        # 运动规划核心
│   │   ├── utils.py                # 规划工具
│   │   └── solutions/              # 各任务的规划求解器（11个）
│   ├── policy/                     # 策略训练与评估
│   │   ├── OGCRL/                  # 目标条件 RL 策略（主要使用）
│   │   │   ├── train.py            # 训练入口（Hydra）
│   │   │   ├── train_eval_acp.sh   # 训练+评估批量脚本
│   │   │   ├── eval_multi_dp.py    # 多智能体评估
│   │   │   └── ogcrl/              # 核心算法包
│   │   │       ├── agents/         # RL 算法：hiql, crl, gcbc, gciql, gcivl
│   │   │       ├── common/         # 通用模块：replay_buffer, sampler
│   │   │       ├── config/         # Hydra 配置（agent/task）
│   │   │       ├── dataset/        # 数据集加载与转换
│   │   │       ├── utils/          # 工具：multiagent_manager, encoders, networks
│   │   │       └── workspace/      # 训练工作流：robotworkspace.py
│   │   └── Diffusion-Policy/       # 扩散策略（对比方法）
│   │       ├── train.py            # 训练入口
│   │       ├── diffusion_policy/   # 扩散策略核心包
│   │       └── eval_multi_dp.py    # 多智能体评估
│   ├── script/                     # 数据处理脚本
│   │   ├── run_task.py             # 运行任务可视化
│   │   ├── generate_data.py        # 生成演示数据
│   │   ├── parse_h5_to_pkl_multi.py
│   │   ├── parse_pkl_to_zarr_dp.py # 转换为 zarr 格式
│   │   └── download_assets.py
│   └── utils/                      # 底层工具
│       ├── envs/                   # SAPIEN 环境封装
│       ├── building/               # URDF 加载
│       ├── scenes/                 # 场景构建（robocasa/table）
│       └── wrappers/               # 录制包装器
```

## 核心代码关系

```
┌──────────────────────────────────────────────────────────┐
│              policy/OGCRL/train.py                         │
│        （Hydra 配置驱动，训练主入口）                         │
└──────┬────────────────────────────────────────────────────┘
       │ hydra.main → RobotWorkspace.run()
       ▼
┌──────────────────────────────────────────────────────────┐
│         ogcrl/workspace/robotworkspace.py                  │
│  1. 加载 zarr 数据 → convert_replaybuffer_to_rl_dataset   │
│  2. 创建 GCDataset / HGCDataset                          │
│  3. 初始化 MultiAgentManager（多个独立 agent）             │
│  4. 训练循环 + 日志 + 保存                                │
└──────┬──────────────┬────────────────┬───────────────────┘
       │              │                │
       ▼              ▼                ▼
┌────────────┐ ┌───────────────┐ ┌──────────────────────┐
│  agents/   │ │ dataset/      │ │ utils/               │
│ hiql       │ │ robot_gc_     │ │ multiagent_manager   │
│ crl        │ │ dataset.py    │ │ encoders (impala)    │
│ gcbc       │ │ robot_image_  │ │ networks (Flax)      │
│ gciql      │ │ dataset.py    │ │ flax_utils           │
│ gcivl      │ │               │ │ log_utils (wandb)    │
└────────────┘ └───────────────┘ └──────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              planner/run.py                                │
│      （运动规划数据采集，Motion Planning Solutions）         │
└──────┬───────────────────────────────────────────────────┘
       │ MP_SOLUTIONS[env_id] → 各任务求解器
       ▼
┌──────────────────┐    ┌───────────────────────────────┐
│  tasks/          │    │  utils/envs/sapien_env.py     │
│  (ManiSkill Env) │◄───│  utils/scenes/ (场景构建)      │
│  11 个协作任务    │    │  utils/building/ (URDF 加载)   │
└──────────────────┘    └───────────────────────────────┘
```

## 关键概念

### 多智能体训练模式
- 每个 agent 独立训练各自的策略（CTDE: 集中训练、分散执行）
- 数据按 agent 分割：`TaskName_Agent0_150.zarr`, `TaskName_Agent1_150.zarr`
- `MultiAgentManager` 管理多个独立的 JAX agent 实例
- 评估时多个 agent 联合推理

### 策略算法（ogcrl/agents/）
| 算法 | 类 | 说明 |
|------|------|------|
| hiql | HIQLAgent | Hierarchical IQL（主基线） |
| crl | CRLAgent | Contrastive RL |
| gcbc | GCBCAgent | Goal-Conditioned Behavior Cloning |
| gciql | GCIQLAgent | Goal-Conditioned IQL |
| gcivl | GCIVLAgent | Goal-Conditioned IVL |

### 任务环境（tasks/）
- 基于 ManiSkill + SAPIEN 构建
- 支持 2-3 个机器人协作
- 配置文件在 `configs/table/` 或 `configs/robocasa/`
- 典型任务: LiftBarrier-rf, PlaceFood, StackCube, PassShoe 等

### 数据流水线
```
planner/run.py → 生成 .h5 演示数据
    → script/parse_h5_to_pkl_multi.py → .pkl
    → script/parse_pkl_to_zarr_dp.py → .zarr（训练用）
```

## 运行方式

```bash
# 安装
pip install -e .
cd robofactory && pip install -r requirements.txt
pip install -r requirements_ogbench.txt

# 数据采集（运动规划）
python -m robofactory.planner.run -e LiftBarrier-rf -c configs/table/lift_barrier.yaml -n 100

# 数据处理
python script/parse_h5_to_pkl_multi.py
python script/parse_pkl_to_zarr_dp.py

# 训练 OGCRL（从 robofactory/ 目录运行）
python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    agent=hiql observation=visual save_dir=expacp

# 批量训练+评估
bash ./policy/OGCRL/train_eval_acp.sh

# 可视化任务
python script/run_task.py configs/table/lift_barrier.yaml
```

## 修改代码时的注意事项

- 新增算法需在 `policy/OGCRL/ogcrl/agents/` 下添加文件并在 `__init__.py` 的 `agents` dict 中注册
- 新增任务需在 `tasks/` 下定义环境类，在 `tasks/__init__.py` 中导出，并在 `planner/solutions/` 中添加求解器
- 配置管理使用 Hydra，主配置在 `ogcrl/config/robot_gc.yaml`，agent 配置在 `ogcrl/config/agent/`
- 训练脚本需从 `robofactory/` 目录运行（Hydra config_path 相对路径）
- 多 agent 数据需按 Agent0/Agent1 分别生成 zarr 文件
- 评估使用 `eval_multi.sh`，需要先保存 goal（首次训练加 `save_goal=True`）
