# 主旨
本文件用于指导用于debug训练文件的代码生成。
# 前置规则
首先，不要修改任何原始代码逻辑。
# 共用组件debug
暂时不需要debug

# OGCRL debug
## 各项路径、实验数据路径和保存路径
### 代码本体路径
即代码存放的位置
/home/wangyi/Liangziyan/RoboFactory/robofactory/policy/OGCRL

### 实验数据路径
训练数据目录：/media/disk/wangyi/RoboFactory/robofactory/data/
其中包含：
h5_data
pkl_data
zarr_data

调试使用的数据文件：
/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr
/media/disk/wangyi/RoboFactory/robofactory/data/zarr_data/LiftBarrier-rf_Agent1_150.zarr

### 日志及输出保存路径
输出目录：/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/debug_ogcrl
目标文件保存路径：/media/data01/wangyi/liangziyan/mangobench_manipulation/logs/debug_ogcrl/debug_goals.pkl

若输出目录不存在，脚本会自动创建并提示。
## 用于测试的简化算法文件
只通过添加一个算法文件来进行debug，即类比OCGRL的hiql.py文件，添加一个debugalgor.py文件。输入接口和输出接口保持一致，但内部网络简化。调用OGCRL文件夹中的其他功能组件测试。
## 缩短时间debug的实验配置
然后也新建一个实验配置文件用于debug，debug的实验配置也可以简化或减少训练轮次，仅验证单轮训练，梯度更新和模型保存，以及推理成功即可。

参考配置：
```bash
python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
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
    save_dir=expacp \
    save_goal=True \
    save_goal_path=policy/OGCRL/ogcrl/goal
```
其中有些参数需要修改以符合快速debug的要求。

## 输出关键值
例如归一化前后的数据的范围，归一化前后的输出范围，数据加载的路径，数据加载的数量，模型保存路径。或者你认为有什么需要补充的关键值。

# DP dubug
暂时不需要debug