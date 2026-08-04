# 动作归一化器（ActionNormalizer）检查结果

## 结论摘要

- 仓库里确实实现了动作归一化器 `ActionNormalizer`，其归一化/反归一化是把动作线性映射到 `[-1, 1]` 再映射回原尺度（不包含 clip/clamp）。
- 训练阶段是否对动作做归一化，完全由数据集 `RobotImageDataset(normalize_actions=...)` 的开关控制；默认配置未显式开启，因此默认不会归一化。
- HIQL 的损失实现本身不强制动作必须在 `[-1,1]`，它只要求“数据 actions 的尺度”与“actor 分布学习到的尺度”一致。
- OGCRL 的评测脚本会尝试加载 `action_normalizer.pkl`，加载成功才会在评测时对策略输出动作做反归一化；加载失败就直接把原始策略输出送入环境/规划。

## 1. 动作归一化器实现与参数

- 位置：[action_normalizer.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/utils/action_normalizer.py)
- 关键参数：
  - `action_min` / `action_max`：由 `fit(actions)` 统计得到（每个动作维度的 min/max）。
  - `range_eps`：防止某些维度动作跨度过小导致除零；会对跨度小的维度使用 1.0 替代。
  - `fitted`：是否已 `fit` 的标记（未 fit 调用 normalize/unnormalize 会报错）。
- 归一化/反归一化定义（线性映射，不包含裁剪）：
  - `normalize`: `(a - min) / (max - min) * 2 - 1`
  - `unnormalize`: `(a + 1) / 2 * (max - min) + min`

## 2. 训练阶段：哪些地方需要动作归一化

### 2.1 数据集侧（是否执行归一化的唯一入口）

- 位置：[robot_image_dataset.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/dataset/robot_image_dataset.py#L13-L53)
- `RobotImageDataset.__init__(..., normalize_actions=False)` 默认是 `False`。
- 当 `normalize_actions=True` 时会发生两件事：
  - 对 replay buffer 的 `action` 做 `fit` 并归一化到 `[-1,1]`
  - 将归一化后的动作“原地写回”到 `replay_buffer.root["data"]["action"]`
  - 同时把 `self.action_normalizer` 暴露给 workspace（用于保存）

### 2.2 训练入口是否把开关传进来

- dataset 创建位置：[robotworkspace.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/workspace/robotworkspace.py#L78-L82)
  - 代码为 `hydra.utils.instantiate(cfg.task.dataset)`，所以只有当配置里带了 `task.dataset.normalize_actions: true`，才会启用动作归一化。
- 默认 task 配置未显式开启该字段：[default_task.yaml](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/config/task/default_task.yaml#L3-L11)
  - 因此默认训练不会归一化（除非运行时通过命令行 override）。

### 2.3 HIQL 训练/损失是否“必须动作归一化”

- HIQL 使用 `dist.log_prob(batch['actions'])` 训练 actor（没有额外的 `clip(-1,1)` 或 tanh-squash 变换）：
  - 低层 actor loss：[hiql.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/agents/hiql.py#L68-L109)
- 采样后 `clip(-1,1)` 被注释掉：
  - [hiql.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/agents/hiql.py#L197-L221)
- actor 分布是无界高斯（默认 `tanh_squash=False`）：
  - [networks.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/utils/networks.py#L143-L216)
- 结论：
  - 训练代码本身不强制动作必须在 `[-1,1]`；
  - 但如果你决定在数据端做了 `normalize_actions=True`，则训练时 actor 学到的动作尺度也会相应变成“数据中的 [-1,1] 尺度”；这要求评测/部署阶段必须做一致的反归一化，否则动作尺度会错。

### 2.4 训练时是否会保存 action_normalizer.pkl

- 保存位置：[robotworkspace.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/workspace/robotworkspace.py#L95-L99)
- 条件：仅当 dataset 上存在 `action_normalizer` 属性（即你启用了 `normalize_actions=True`）才会保存到 `cfg.save_dir/action_normalizer.pkl`。

## 3. 评测阶段：是否需要反归一化、在哪里做

### 3.1 OGCRL 路径（robofactory/policy/OGCRL/eval_multi_dp.py）

- 脚本会尝试加载每个 agent 的 `action_normalizer.pkl`，若加载成功则对 `now_action` 执行 `unnormalize`：
  - 加载逻辑：[eval_multi_dp.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/eval_multi_dp.py#L214-L233)
  - 反归一化逻辑：[eval_multi_dp.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/eval_multi_dp.py#L320-L328)
- 若加载失败：评测就会直接使用策略输出动作进入环境/规划（同样不做 clip/clamp）。

### 3.2 Diffusion-Policy 路径（robofactory/policy/Diffusion-Policy）

- 反归一化在 policy 内部完成，不依赖 `action_normalizer.pkl`：
  - `predict_action()` 内部会执行 `self.normalizer['action'].unnormalize(...)`：
    - [diffusion_unet_image_policy.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/Diffusion-Policy/diffusion_policy/policy/diffusion_unet_image_policy.py#L123-L186)
- `LinearNormalizer` 的 `unnormalize()` 也是线性变换，不包含裁剪：
  - [normalizer.py](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/Diffusion-Policy/diffusion_policy/model/common/normalizer.py#L264-L280)

## 4. 针对“成功率波动很大”的最小化排查清单（不改代码）

1. 确认你当前训练是否真的启用了 `task.dataset.normalize_actions=true`（默认没有）。
2. 若训练启用了动作归一化：
   - 确认训练输出目录里确实生成了 `action_normalizer.pkl`（保存逻辑见 [robotworkspace.py:L95-L99](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/ogcrl/workspace/robotworkspace.py#L95-L99)）。
   - 确认评测脚本能加载到同一份 `action_normalizer.pkl` 并执行了反归一化（见 [eval_multi_dp.py:L214-L233](file:///d:/2026/code/mangobench-manipulation/robofactory/policy/OGCRL/eval_multi_dp.py#L214-L233)）。
3. 若训练未启用动作归一化：
   - 评测也不应依赖 `action_normalizer.pkl`；否则训练/评测尺度不一致会导致动作幅度错误，从而成功率大幅波动。
4. 数据一致性（不读大日志的前提下）：
   - 每次训练是否使用相同的 `task.dataset.zarr_path`（不同数据子集会带来性能波动）。
5. 目标/评测随机性：
   - 若评测目标存在随机采样（或从文件挑选子集），且 seed/采样不固定，成功率也会出现明显波动（需要进一步确认评测侧 goal 选择逻辑）。

