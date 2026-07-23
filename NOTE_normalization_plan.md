# MangoBench-Manipulation 动作归一化方案分析

## 1. 关节上下限调查结论

### 1.1 已有的关节限位信息

经过代码排查，**关节位置上下限存在于 URDF 模型中**，通过 `mplib` 的 `pinocchio_model` 解析获取：

```python
# robofactory/utils/mplib_utils.py (FlexiblePlanner.__init__)
self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())
# 返回形状为 (n_joints, 2) 的数组，每行为 [lower, upper]
```

此外还有关节速度/加速度限制（用于轨迹规划的时间参数化）：
```python
self.joint_vel_limits = joint_vel_limits  # 默认为 np.ones(7) * 0.9
self.joint_acc_limits = joint_acc_limits  # 默认为 np.ones(7) * 0.9
```

### 1.2 但训练数据中没有使用这些限位

关键发现：
- `joint_limits` 仅在 **规划器**（`FlexiblePlanner`）中用于路径规划和 IK 约束检查
- **训练侧**（`policy/OGCRL/`）的数据加载流程 (`robot_gc_dataset.py`) **没有任何归一化处理**
- `data["action"]` 和 `data["state"]` 直接作为原始值送入模型训练
- OGCRL 的 `GCDataset` / `HGCDataset` 中没有 normalizer 逻辑

### 1.3 对比：Diffusion-Policy 侧已有归一化

`policy/Diffusion-Policy/` 侧有完善的归一化框架：
- `diffusion_policy/model/common/normalizer.py` — `LinearNormalizer` / `SingleFieldLinearNormalizer`
- `diffusion_policy/common/normalize_util.py` — `get_range_normalizer_from_stat()` 等工具
- `diffusion_policy/dataset/robot_image_dataset.py` — `get_normalizer()` 方法
- 训练时 `workspace` 调用 `dataset.get_normalizer()` → `model.set_normalizer(normalizer)`

**但 OGCRL 侧完全没有类似机制。**

---

## 2. 问题分析

对于 OGCRL 策略（hiql, crl, gcbc 等），action 数据的维度结构为：
- Panda 7-DoF 关节位置 + 1-DoF 夹爪 = 8 维（每个 agent）
- 各关节的量纲和范围差异大（如关节 1 范围 [-2.89, 2.89]，夹爪 [0, 0.04]）

没有归一化会导致：
1. 不同维度梯度量级不一致，训练不稳定
2. 网络输出需要覆盖差异极大的输出范围
3. 模型对小范围维度（如夹爪）不敏感

---

## 3. 归一化方案设计

### 3.1 方案选择：基于数据统计的 Min-Max 归一化（推荐）

**理由**：
- URDF 中的 `joint_limits` 是理论极限，实际数据可能只覆盖其中一个子集
- 使用数据统计的 min/max 更贴合实际分布，归一化后利用率更高
- 同时保留 URDF limits 作为 fallback（clip 安全边界）

归一化公式：
```
normalized = (x - min) / (max - min) * 2 - 1   →  映射到 [-1, 1]
unnormalized = (normalized + 1) / 2 * (max - min) + min
```

### 3.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│   数据加载阶段（robotworkspace.py）                           │
│                                                             │
│   zarr 数据 → convert_replaybuffer_to_rl_dataset()          │
│        ↓                                                    │
│   ActionNormalizer.fit(dataset["actions"])                   │
│   dataset["actions"] = normalizer.normalize(actions)        │
│        ↓                                                    │
│   保存 normalizer 到 save_dir/action_normalizer.pkl         │
│        ↓                                                    │
│   GCDataset / HGCDataset (归一化后的 actions)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│   训练阶段                                                   │
│                                                             │
│   agent 网络输出 → 输出范围 [-1, 1]（tanh 激活）             │
│   loss 计算使用归一化后的 actions                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│   推理/评估阶段（eval_multi_dp.py）                          │
│                                                             │
│   加载 normalizer = load(save_dir/action_normalizer.pkl)    │
│   model_output ([-1, 1]) → normalizer.unnormalize()         │
│        ↓                                                    │
│   真实 action → env.step(action)                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 具体实现位置

| 文件 | 修改内容 |
|------|---------|
| `ogcrl/dataset/robot_gc_dataset.py` | 在 `convert_replaybuffer_to_rl_dataset()` 中 fit normalizer 并归一化 actions |
| `ogcrl/workspace/robotworkspace.py` | 保存 normalizer；传递给评估代码 |
| `ogcrl/utils/action_normalizer.py` | **新建**：ActionNormalizer 类 |
| `policy/OGCRL/eval_multi_dp.py` | 加载 normalizer，推理后 unnormalize |
| `ogcrl/config/robot_gc.yaml` | 添加 `normalize_actions: true` 配置项 |

### 3.4 ActionNormalizer 类设计

```python
# ogcrl/utils/action_normalizer.py
import numpy as np
import pickle

class ActionNormalizer:
    """动作空间归一化器，将 action 映射到 [-1, 1]"""
    
    def __init__(self):
        self.fitted = False
        self.action_min = None  # (action_dim,)
        self.action_max = None  # (action_dim,)
        self.range_eps = 1e-7   # 防止除零
    
    def fit(self, actions: np.ndarray):
        """从数据中计算 min/max 统计量
        Args:
            actions: shape (N, action_dim)
        """
        self.action_min = actions.min(axis=0)
        self.action_max = actions.max(axis=0)
        # 对范围过小的维度做保护
        small_range = (self.action_max - self.action_min) < self.range_eps
        self.action_max[small_range] = self.action_min[small_range] + 1.0
        self.fitted = True
    
    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """归一化到 [-1, 1]"""
        assert self.fitted
        return (actions - self.action_min) / (self.action_max - self.action_min) * 2 - 1
    
    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        """从 [-1, 1] 还原到原始空间"""
        assert self.fitted
        return (actions + 1) / 2 * (self.action_max - self.action_min) + self.action_min
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'min': self.action_min, 'max': self.action_max}, f)
    
    @classmethod
    def load(cls, path: str):
        normalizer = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        normalizer.action_min = data['min']
        normalizer.action_max = data['max']
        normalizer.fitted = True
        return normalizer
```

### 3.5 数据加载侧修改要点

在 `convert_replaybuffer_to_rl_dataset()` 返回前：

```python
# 在 robotworkspace.py 中调用
normalizer = ActionNormalizer()
normalizer.fit(dataset["actions"])
dataset["actions"] = normalizer.normalize(dataset["actions"])
normalizer.save(os.path.join(save_dir, "action_normalizer.pkl"))
```

### 3.6 评估侧修改要点

在 `eval_multi_dp.py` 的推理循环中：

```python
normalizer = ActionNormalizer.load(os.path.join(checkpoint_dir, "action_normalizer.pkl"))

# 推理时
raw_action = agent.sample_actions(...)  # 输出 [-1, 1]
action = normalizer.unnormalize(raw_action)  # 还原到真实值
obs, reward, done, info = env.step(action)
```

---

## 4. 注意事项

1. **state 观测是否也要归一化？** — 当 `observation=visual` 时，观测是图像（已经是 uint8 [0,255]），不需要额外归一化。当 `observation=state` 时建议也做类似处理。

2. **归一化器与多 Agent 的关系** — 每个 Agent 的 zarr 数据独立，因此每个 Agent 会有独立的 normalizer。保存时按 `AgentX_action_normalizer.pkl` 区分。

3. **向后兼容** — 添加配置项 `normalize_actions: true/false`，默认 false 以兼容旧实验。

4. **是否使用 URDF limits 替代数据统计？** — 不推荐作为主方案，因为 Panda 的理论关节范围很大，但实际数据只覆盖一小部分，使用理论范围会导致归一化后的数据集中在很窄的区间，浪费网络表达能力。可以把 URDF limits 作为 clip 的安全边界。

5. **Diffusion-Policy 侧** — 已有完善的归一化框架，无需重复实现。本方案仅针对 OGCRL 侧。
