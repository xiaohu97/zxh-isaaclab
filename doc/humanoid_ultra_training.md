# Humanoid Ultra 训练说明

## 已集成内容

参考文件已按本项目结构放置：

- 机器人配置：`source/unitree_rl_lab/unitree_rl_lab/assets/robots/humanoid_ultra.py`
- URDF、MJCF 和网格：`source/unitree_rl_lab/unitree_rl_lab/assets/robots/humanoid_ultra_description`
- 可训练任务：`source/unitree_rl_lab/unitree_rl_lab/tasks/humanoid_ultra/base`
- 未改动的原始训练资料：`reference/humanoid_ultra/original`

正式 Isaac Lab 任务使用 `meshes_orign` 中经过检查的 STL。`meshes`
目录内部分简化网格不是有效的标准 STL，因此不用于训练。

当前注册了四个 RSL-RL 任务：

```text
USTC-Humanoid-Ultra-12dof-Flat
USTC-Humanoid-Ultra-12dof-Rough
USTC-Humanoid-Ultra-27dof-Flat
USTC-Humanoid-Ultra-27dof-Rough
```

## 环境检查

先进入安装了 Isaac Lab 2.2 / Isaac Sim 5.0 的环境：

```bash
conda activate ustc_isaaclab
./unitree_rl_lab.sh -i
./unitree_rl_lab.sh -l
```

如果环境名称不同，替换第一条命令即可。列表中应出现上面的四个任务。

## 开始训练

先用少量环境验证模型和奖励，再扩大并行环境数量：

```bash
# 12 自由度平地冒烟训练
./unitree_rl_lab.sh -t \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --num_envs 64 \
  --max_iterations 10

# 12 自由度正式平地训练
./unitree_rl_lab.sh -t \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --num_envs 4096

# 27 自由度平地训练
./unitree_rl_lab.sh -t \
  --task USTC-Humanoid-Ultra-27dof-Flat \
  --num_envs 4096

# 平地策略稳定后训练复杂地形
./unitree_rl_lab.sh -t \
  --task USTC-Humanoid-Ultra-27dof-Rough \
  --num_envs 4096
```

环境数量需要根据显存调整。显存不足时依次尝试 `2048`、`1024`、`512`。
默认训练迭代数来自对应的 `agents/*_agent_cfg.py`，也可用
`--max_iterations` 临时覆盖。

训练日志分别写入：

```text
logs/rsl_rl/humanoidultra12dof_flat
logs/rsl_rl/humanoidultra12dof_rough
logs/rsl_rl/humanoidultra27dof_flat
logs/rsl_rl/humanoidultra27dof_rough
```

查看曲线：

```bash
tensorboard --logdir logs/rsl_rl
```

## 播放策略

```bash
./unitree_rl_lab.sh -p \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --checkpoint "/home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/训练目录/model_1000.pt"
```

导出供 MuJoCo 使用的 TorchScript 策略时，向 `--checkpoint` 传入绝对路径：

```bash
./unitree_rl_lab.sh -p \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --checkpoint "/home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/训练目录/model_1000.pt" \
  --headless \
  --export_only
```

输出位于 checkpoint 同目录下的 `exported/policy.pt` 和
`exported/policy.onnx`。Humanoid Ultra 的独立 MuJoCo 测试器位于
`/home/zxh/ustc_humanoid/unitree_mujoco/sim2sim/humanoid_ultra`。

先在 Isaac Lab 中确认站立、速度跟踪、关节限位和足端接触正常，再进行
MuJoCo sim2sim。当前集成没有 Humanoid Ultra 的 Unitree SDK
关节映射和控制器，因此不会生成可直接用于实机的 `deploy.yaml`。

## 推荐训练顺序

1. 检查 URDF 的单位、质量、惯量、关节轴、限位和碰撞体。
2. 在 `humanoid_ultra.py` 中校准初始姿态、力矩/速度限制、刚度和阻尼。
3. 先训练 12-DOF Flat，使下肢站立和低速行走稳定。
4. 再训练 27-DOF Flat，逐步开放上肢动作并控制躯干稳定。
5. Flat 收敛后训练 Rough，逐步增加地形课程和随机扰动强度。
6. 做多随机种子评估、sim2sim，然后单独实现部署关节映射和安全状态机。

## 新机器人接入方法

接入另一个机器人时，按下面的边界修改，不要直接把外部工程整体塞进任务目录：

1. 将 URDF/MJCF/mesh 放入 `assets/robots/<robot>_description`。
2. 在 `assets/robots/<robot>.py` 定义 `ArticulationCfg`、初始关节角和执行器参数。
3. 在 `tasks/locomotion/robots` 或独立任务包中定义环境配置和奖励。
4. 在任务包 `__init__.py` 中注册唯一的 Gym task ID。
5. 在 `agents` 中定义 RSL-RL 网络、PPO 参数和实验名。
6. 依次执行任务列表检查、1 环境 1 迭代测试、小规模训练和大规模训练。

观测历史长度、镜像索引、动作维度和关节名称必须随自由度重新计算。直接复用
其他机器人的这些索引通常不会报配置错误，但会让对称增强和策略学习失效。

## 原始资料限制

- `reference/humanoid_ultra/original/policy.pt` 和 `gamepad_reader.py`
  的文件格式不是当前项目可直接加载的标准 PyTorch/文本格式，未参与训练。
- 原始 AMP 任务依赖未提供的运动数据、RoboLab 数学工具和定制 runner，
  因此本次只保留为参考，没有注册 AMP task。
- 仓库默认忽略 `*.pt`，所以参考策略已复制到工作目录，但不会默认进入 Git。
