# MangoBench-Locomotion 项目概览

## 项目简介

MangoBench 是一个多智能体目标驱动离线强化学习基准（CVPR 2026），覆盖 47 个任务。本仓库为 **locomotion（运动学）** 部分，基于 OGBench 扩展，将单智能体环境改造为多智能体协作场景。

- 仓库地址: https://github.com/SYSU-SAIL/mangobench-locomotion.git
- 操作空间部分: https://github.com/WendyeeWang/mangobench-manipulation

## 技术栈

- Python 3.10, JAX, Flax, Gymnasium
- MuJoCo (mujoco >= 3.1.6, dm_control >= 1.0.20)
- wandb (实验跟踪), absl-py (命令行参数)

## 目录结构

```
mangobench-locomotion/
├── ogbench/                    # 核心环境包（基于 OGBench 扩展）
│   ├── __init__.py             # 导出 locomaze/manipspace/powderworld + 工具函数
│   ├── utils.py                # 数据集下载/加载/环境创建的核心工具
│   ├── locomaze/               # 运动迷宫环境（主要使用）
│   │   ├── __init__.py         # Gymnasium 环境注册（antmaze/pointmaze/humanoidmaze/antsoccer）
│   │   ├── maze.py             # make_maze_env 入口，迷宫环境实现
│   │   ├── ant.py              # Ant 智能体定义
│   │   ├── humanoid.py         # Humanoid 智能体定义
│   │   ├── point.py            # Point 智能体定义
│   │   └── assets/             # MuJoCo XML 模型文件
│   ├── online_locomotion/      # 在线运动环境（ant_ball, ant, humanoid）
│   ├── manipspace/             # 操作空间环境
│   └── powderworld/            # 粉末世界环境
├── impls/                      # 算法实现与训练入口
│   ├── main.py                 # 训练主入口（多智能体训练循环）
│   ├── MujocoMulti.py          # 多智能体观察/动作空间分割（batch_partition）
│   ├── hyperparameters_multi.sh # 全部实验运行命令（超参配置）
│   ├── requirements.txt        # Python 依赖
│   ├── agents/                 # 离线 RL 算法实现
│   │   ├── __init__.py         # 算法注册表
│   │   ├── gcbc.py             # Goal-Conditioned Behavior Cloning
│   │   ├── crl.py              # Contrastive RL
│   │   ├── hiql.py             # Hierarchical IQL
│   │   └── sac.py              # Soft Actor-Critic（用于数据生成）
│   └── utils/                  # 训练工具模块
│       ├── datasets.py         # Dataset/GCDataset/HGCDataset 数据集类
│       ├── encoders.py         # 观察编码器
│       ├── networks.py         # 神经网络定义（Flax）
│       ├── env_utils.py        # 环境创建工具
│       ├── evaluation.py       # 评估逻辑
│       ├── multiagent_manager.py # MultiAgentManager（多智能体管理器）
│       ├── flax_utils.py       # 模型保存/恢复
│       └── log_utils.py        # 日志/wandb 工具
├── data_gen_scripts/           # 数据集生成脚本
│   ├── commands.sh             # 数据生成命令
│   ├── generate_locomaze.py    # 生成 locomaze 数据
│   ├── generate_antsoccer.py   # 生成 antsoccer 数据
│   ├── generate_manipspace.py  # 生成 manipspace 数据
│   ├── generate_powderworld.py # 生成 powderworld 数据
│   ├── main_sac.py             # SAC 在线训练（采集数据用）
│   ├── online_env_utils.py     # 在线环境工具
│   └── viz_utils.py            # 可视化工具
└── pyproject.toml              # 包元信息（ogbench v1.0.1）
```

## 核心代码关系

```
┌─────────────────────────────────────────────────────┐
│                  impls/main.py                       │
│          （训练主循环，命令行参数解析）                  │
└───────┬──────────────┬───────────────┬──────────────┘
        │              │               │
        ▼              ▼               ▼
┌──────────────┐ ┌───────────┐ ┌────────────────────┐
│ MujocoMulti  │ │  agents/  │ │ utils/             │
│ .py          │ │ gcbc/crl/ │ │ multiagent_manager │
│              │ │ hiql/sac  │ │ datasets/eval/...  │
│ batch_       │ └─────┬─────┘ └────────┬───────────┘
│ partition()  │       │                │
└──────┬───────┘       │                │
       │               ▼                ▼
       │    MultiAgentManager     ogbench/utils.py
       │    (为每个 agent 创建     (make_env_and_datasets
       │     独立实例并分别训练)     load_dataset, download)
       │                                │
       ▼                                ▼
  观察/动作空间按              ogbench/locomaze/
  agent_conf 分割              (Gymnasium 环境注册)
  (2x4, 2x4d, 4x2)            maze.py → ant/humanoid/point
```

## 关键概念

### 多智能体分割 (MujocoMulti.py)
- `batch_partition(batch, agent_conf, scenario)`: 将单体 Ant 的观察和动作按关节分割给多个 agent
- `agent_conf` 格式: `NxM` 表示 N 个 agent 各控制 M 个关节，`d` 后缀表示 decoupled
- 支持 scenario: `manyagent_ant`, `manyagent_antsoccer`

### MultiAgentManager (impls/utils/multiagent_manager.py)
- 管理多个独立的 agent 实例（每个 agent 有自己的网络参数）
- 提供统一的 `update()`, `evaluate()`, `save()`, `restore()` 接口

### 环境注册 (ogbench/locomaze/__init__.py)
- 通过 Gymnasium register 注册环境 ID
- 入口函数: `ogbench.locomaze.maze:make_maze_env`
- 环境类型: pointmaze / antmaze / humanoidmaze / antsoccer
- 迷宫类型: medium / large / giant / teleport / arena

### 数据集格式
- `.npz` 文件，包含 `observations`, `actions`, `terminals`
- 通过 `ogbench.utils.load_dataset()` 加载，支持 compact/regular 两种模式
- 数据集从 `https://rail.eecs.berkeley.edu/datasets/ogbench` 自动下载

## 运行方式

```bash
# 安装
conda create -n mangobench python=3.10 && conda activate mangobench
cd impls && pip install -r requirements.txt

# 训练（示例）
python main.py --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gcbc.py \
  --agent_conf=2x4 --scenario=manyagent_ant --n_agents=2

# 批量运行所有实验
bash ./impls/hyperparameters_multi.sh
```

## 修改代码时的注意事项

- 新增算法需在 `impls/agents/` 下添加文件并在 `__init__.py` 中注册
- 多智能体分割逻辑在 `MujocoMulti.py`，修改关节分配需同步修改 obs_labels/action_labels
- 环境相关修改在 `ogbench/locomaze/` 下，注意 Gymnasium 注册机制
- 训练超参配置集中在 `hyperparameters_multi.sh`
