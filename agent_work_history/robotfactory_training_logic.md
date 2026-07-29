# RobotFactory 训练逻辑与主调用关系

记录时间: 2026-07-28

## 结论

这个仓库里真正负责训练控制的代码，不在 `train.py` 本身，而是在各自的 `workspace/robotworkspace.py`。
`train.py` 只负责 Hydra 启动、读取配置、实例化 workspace，然后把控制权交给 `workspace.run()`。

当前仓库里有两套训练实现：

1. `policy/OGCRL/`：主训练链路，面向 goal-conditioned / multi-agent RL。
2. `policy/Diffusion-Policy/`：扩散策略的并行实现，训练方式更偏监督式行为克隆。

## OGCRL 主训练链路

```mermaid
flowchart TD
    A[policy/OGCRL/train.py] --> B[Hydra 读取 config]
    B --> C[hydra.utils.get_class(cfg._target_)]
    C --> D[ogcrl.workspace.robotworkspace.RobotWorkspace]
    D --> E[RobotWorkspace.run()]
    E --> F[instantiate(cfg.task.dataset)]
    F --> G[BaseImageDataset / replay_buffer]
    G --> H[convert_replaybuffer_to_rl_dataset]
    H --> I[Dataset.create]
    I --> J[GCDataset / HGCDataset]
    J --> K[agents[config.agent_name]]
    K --> L[agent.create(...)]
    L --> M[训练循环: sample -> update]
    M --> N[save_agent / CsvLogger]
```

### 关键文件关系

- [policy/OGCRL/train.py](../robofactory/policy/OGCRL/train.py)
  - 训练入口。
  - 负责注册 `eval` resolver，加载 `ogcrl/config`，然后实例化 `_target_` 指定的 workspace。

- [policy/OGCRL/ogcrl/config/robot_gc.yaml](../robofactory/policy/OGCRL/ogcrl/config/robot_gc.yaml)
  - 总配置入口。
  - `_target_` 指向 `ogcrl.workspace.robotworkspace.RobotWorkspace`。
  - 决定 `task.dataset`、`agent`、`train_steps`、`save_interval` 等训练参数。

- [policy/OGCRL/ogcrl/workspace/robotworkspace.py](../robofactory/policy/OGCRL/ogcrl/workspace/robotworkspace.py)
  - 真正的训练控制层。
  - 先构建数据集，再做 replay buffer 到 goal-conditioned 数据集转换。
  - 用 `agents` 注册表创建具体算法实例，最后进入迭代更新和保存。

- [policy/OGCRL/ogcrl/workspace/base_workspace.py](../robofactory/policy/OGCRL/ogcrl/workspace/base_workspace.py)
  - 提供 checkpoint 保存 / 读取 / output_dir 统一管理。
  - `RobotWorkspace` 继承它，训练状态会走这里的 `save_checkpoint`、`load_checkpoint` 体系。

- [policy/OGCRL/ogcrl/dataset/robot_gc_dataset.py](../robofactory/policy/OGCRL/ogcrl/dataset/robot_gc_dataset.py)
  - 把 `replay_buffer` 转成 RL 训练格式。
  - `convert_replaybuffer_to_rl_dataset(...)` 负责终止位、goal 保存、图像缩放、state/visual 观测切换。
  - `Dataset`、`GCDataset`、`HGCDataset` 负责采样与 goal-conditioned batch 构造。

- [policy/OGCRL/ogcrl/agents/__init__.py](../robofactory/policy/OGCRL/ogcrl/agents/__init__.py)
  - 算法注册表。
  - `config.agent.agent_name` 决定最终创建 `hiql`、`crl`、`gcbc`、`gciql` 或 `gcivl`。

### 运行时流程

1. `train.py` 被 Hydra 启动。
2. Hydra 读取 `robot_gc.yaml`，确定 workspace 类和训练参数。
3. `RobotWorkspace.__init__()` 初始化随机种子和训练状态。
4. `RobotWorkspace.run()` 实例化 `cfg.task.dataset`，拿到 `replay_buffer`。
5. `convert_replaybuffer_to_rl_dataset()` 把原始轨迹整理成 goal-conditioned 可训练数据。
6. `Dataset.create()`、`GCDataset/HGCDataset` 完成 batch 采样包装。
7. 通过 `agents[...]` 创建具体算法实例，调用 `agent.update(batch)` 进入训练循环。
8. 按 `log_interval` 记录日志，按 `save_interval` 保存模型。

## Diffusion-Policy 链路

```mermaid
flowchart TD
    A[policy/Diffusion-Policy/train.py] --> B[Hydra 读取 config]
    B --> C[hydra.utils.get_class(cfg._target_)]
    C --> D[diffusion_policy.workspace.robotworkspace.RobotWorkspace]
    D --> E[RobotWorkspace.__init__]
    E --> F[instantiate(cfg.policy)]
    E --> G[instantiate(cfg.optimizer)]
    D --> H[RobotWorkspace.run()]
    H --> I[instantiate(cfg.task.dataset)]
    I --> J[create_dataloader]
    J --> K[model.compute_loss(batch)]
    K --> L[optimizer.step / ema / checkpoint]
```

### 关键文件关系

- [policy/Diffusion-Policy/train.py](../robofactory/policy/Diffusion-Policy/train.py)
  - 训练入口。
  - 和 OGCRL 一样，只负责 Hydra 启动和 workspace 实例化。

- [policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py](../robofactory/policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py)
  - 负责模型、优化器、数据加载、EMA、checkpoint 和训练循环。
  - 训练主循环是 `compute_loss -> backward -> optimizer.step -> validation -> checkpoint`。

## 一句话总结

如果要找训练主逻辑，优先看 `policy/*/train.py` 之后跳到对应的 `workspace/robotworkspace.py`。
真正决定训练行为的是 workspace，而不是入口脚本本身。