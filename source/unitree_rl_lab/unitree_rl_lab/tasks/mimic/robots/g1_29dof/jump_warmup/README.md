# G1 29DoF Jump Warmup

这个目录是 `jump` 动作的第一阶段训练环境，用来先让策略学会完整跳跃和落地，再进入更严格的第二阶段训练。

## 任务入口

```bash
Unitree-G1-29dof-Mimic-Jump-Warmup
```

配置文件：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump_warmup/tracking_env_cfg.py
```

动作文件沿用原始 jump：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump/jump.npz
```

## 为什么需要 warmup

跳跃动作比普通舞蹈更难收敛，原因是：

- 起跳和落地阶段速度、角速度变化大。
- 双脚短时间离地，接触状态变化明显。
- 如果一开始使用严格终止条件，策略还没学到后半段就会频繁 reset。
- 如果一开始加入外力、质心、摩擦等 domain randomization，学习难度会过高。

所以 warmup 的目标不是最终部署鲁棒性，而是让策略先完整看完动作并学会基本存活。

## 相比原始 mimic 的调整

### 1. 减少随机干扰

原始配置来自 `dance_102`，包含较强 domain randomization：

```text
static_friction_range: 0.3 - 1.6
dynamic_friction_range: 0.3 - 1.2
restitution_range: 0.0 - 0.5
add_joint_default_pos: enabled
base_com: enabled
push_robot: enabled
```

warmup 改为：

```text
static_friction_range: 0.8 - 1.2
dynamic_friction_range: 0.8 - 1.2
restitution_range: 0.0 - 0.1
add_joint_default_pos: None
base_com: None
push_robot: None
```

作用：先去掉外界干扰，让策略专注学习跳跃动作本身。

### 2. 放宽终止条件

原始 dance 基类：

```text
anchor_pos threshold: 0.25
anchor_ori threshold: 0.8
ee_body_pos threshold: 0.25
ee_body_pos bodies: left/right ankle + left/right wrist
```

warmup 当前配置：

```text
anchor_pos threshold: 0.8
anchor_ori threshold: 1.5
ee_body_pos threshold: 1.2
ee_body_pos bodies: left/right ankle only
```

作用：避免策略在跳跃中段因为误差较大过早 reset，使 rollout 能覆盖完整起跳和落地过程。

### 3. 降低动作变化惩罚

原始配置：

```text
action_rate_l2 weight: -1e-1
```

warmup 当前配置：

```text
action_rate_l2 weight: -5e-2
```

作用：跳跃需要更快的关节动作，减小该惩罚可以避免策略过度保守。

## 当前训练结果

最新有效 run：

```text
logs/rsl_rl/unitree_g1_29dof_mimic_jump/2026-05-26_16-58-58_warmup
```

当前已保存 checkpoint：

```text
model_37000.pt
```

最近指标：

```text
Train/mean_reward: 76.99
Train/mean_episode_length: 1500.00
Episode_Termination/time_out: 0.9976
Episode_Termination/anchor_pos: 0.0015
Episode_Termination/anchor_ori: 0.0007
Episode_Termination/ee_body_pos: 0.0002
Metrics/motion/error_anchor_pos: 0.4359
Metrics/motion/error_body_pos: 0.0884
Metrics/motion/error_joint_pos: 1.0613
```

结论：

- warmup 已经基本满 episode 存活。
- 姿态和身体局部跟踪已经明显改善。
- `error_anchor_pos` 仍然偏大，说明整体位置或落点还不够准。

## 训练命令

从当前 warmup checkpoint 继续：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-Jump-Warmup \
  --headless \
  --resume \
  --load_run 2026-05-26_16-58-58_warmup \
  --checkpoint model_37000.pt
```

可视化检查：

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-Jump-Warmup \
  --load_run 2026-05-26_16-58-58_warmup \
  --checkpoint model_37000.pt
```

## 第二阶段建议

warmup 成功后，不建议直接恢复到最严格参数。建议逐步收紧：

```text
anchor_pos threshold: 0.8 -> 0.65
anchor_ori threshold: 1.5 -> 1.3
ee_body_pos threshold: 1.2 -> 1.0
```

如果仍然稳定，再继续收紧：

```text
anchor_pos threshold: 0.65 -> 0.5
anchor_ori threshold: 1.3 -> 1.1
ee_body_pos threshold: 1.0 -> 0.8
```

第二阶段目标：

```text
Mean episode length > 1200
time_out > 0.8
anchor_pos termination < 0.05
anchor_ori termination < 0.05
ee_body_pos termination < 0.05
error_anchor_pos 持续下降
```

等第二阶段稳定后，再逐步恢复 domain randomization，例如先恢复摩擦范围，再恢复质心随机，最后恢复 push。
