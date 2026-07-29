# MangoBench-Manipulation 快速发烟测试

这个文档用于在**最短时间内确认代码和入口脚本能跑通**，不追求完整训练结果。建议先做“无界面检查”，再做“最小仿真检查”。

## 1. 目标

确认下面几件事：

- `robofactory` 包可以被正常导入
- 关键 CLI 入口可以正常解析参数
- 至少一个最小任务可以启动并完成 1 条轨迹/1 次运行
- 如果后续要训练，`OGCRL` 的 Hydra 入口也能正常启动

## 2. 前置条件

在项目根目录 `mangobench-manipulation/` 下执行。

建议先完成基础安装：

```bash
pip install -e .
cd robofactory
pip install -r requirements.txt
pip install -r requirements_ogbench.txt
```

如果你已经配置过 RoboFactory / ManiSkill / SAPIEN 环境，可以直接跳到下面的测试命令。

## 3. 最快的无界面检查

这一步只验证“代码能被 Python 找到、入口能启动参数解析”，通常几秒钟内结束。

```bash
cd /home/wangyi/Liangziyan/mangobench-manipulation

# 1) 包导入检查
python -c "import robofactory; print('CONFIG_DIR =', robofactory.CONFIG_DIR)"

# 2) 规划器入口检查（只看 help，不真正跑任务）
python -m robofactory.planner.run -h

# 3) 数据生成脚本参数检查
cd robofactory
python script/generate_data.py -h
python policy/OGCRL/train.py --help
```

如果上面任何一步报错，优先检查：

- Python 环境是否激活
- `pip install -e .` 是否执行成功
- 是否在正确的目录下运行（尤其是 `robofactory/` 里的训练入口）

## 4. 最小仿真检查

这一步用于确认“任务真的能启动并跑一小段”，建议先选一个最轻量的任务：`LiftBarrier-rf`。

### 4.1 直接跑一个最小任务

`run_task.py` 会帮你调用规划器，并固定使用 1 条轨迹、CPU 和可视化模式：

```bash
cd /home/wangyi/Liangziyan/mangobench-manipulation/robofactory
python script/run_task.py configs/table/lift_barrier.yaml
```

如果你的机器没有图形界面，这个命令可能因为 `--render-mode=human` / `--vis` 失败。那就改用下面的无头数据生成命令。

### 4.2 无头生成 1 条演示数据

这是更适合“快速验证任务流程”的方式：

```bash
cd /home/wangyi/Liangziyan/mangobench-manipulation/robofactory
python script/generate_data.py \
  --scene table \
  --task LiftBarrier-rf \
  --num 100 \
  --record-dir data/demos_smoke
```

运行成功后，说明至少满足：

- 场景配置可以找到
- 任务可以实例化
- 规划/采样流程可以启动
- 结果能写入本地目录

## 5. 如果你想顺手检查训练入口

训练脚本依赖 Hydra 配置，最轻量的检查方式是先看帮助信息：

```bash
cd /home/wangyi/Liangziyan/mangobench-manipulation/robofactory
python policy/OGCRL/train.py --help
```

如果你已经有可用的 zarr 数据，再做一次最小训练启动：

```bash
python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=LiftBarrier-rf \
  task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
  agent=hiql \
  observation=visual \
  save_dir=exp_smoke
```

> 提示：首次训练如果需要评估目标，仓库 README 里提到要加 `save_goal=True` 和 `save_goal_path=policy/OGCRL/ogcrl/goal`。

## 6. 推荐的快速排查顺序

如果你只想尽快定位问题，按这个顺序执行就行：

1. `python -c "import robofactory"`
2. `python -m robofactory.planner.run -h`
3. `python script/generate_data.py --scene table --task LiftBarrier-rf --num 1`
4. `python policy/OGCRL/train.py --help`

## 7. 这套 smoke test 的判断标准

满足以下任意一种，基本就能说明“代码能跑通”：

- 只做无界面检查：所有 help / import 命令都正常退出
- 做最小仿真检查：`LiftBarrier-rf` 能成功生成 1 条轨迹，且输出目录中出现结果文件

如果你希望，我也可以继续帮你补一份“更严格的完整冒烟测试清单”，把数据处理、训练、评估三段都串起来。