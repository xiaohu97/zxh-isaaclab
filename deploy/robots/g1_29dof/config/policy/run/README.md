# Run 策略部署目录

对应训练任务 `Unitree-G1-29dof-Run` / `Unitree-G1-29dof-RunWithId`，控制器状态 `Run`（`State_Run`）。

## 目录

```
run/
└── <版本>/                 例如 0908；控制器取按名字排序的最后一个含 exported/ 的子目录
    ├── exported/policy.onnx     纯 actor(normalizer(x))，不含动作裁剪
    └── params/
        ├── deploy.yaml          观测/动作/增益 + commands.base_velocity.gait（训练侧参数）+ actions.*.raw_clip
        ├── source.txt           来自哪个 checkpoint
        └── reference.json       Isaac Lab 参考轨迹，供离线对拍（可删）
```

## 生成

```bash
conda activate ustc_isaaclab
python scripts/rsl_rl/export_deploy.py --task Unitree-G1-29dof-Run \
    --load_run <run目录名> --checkpoint model_xxxxx.pt \
    --out deploy/robots/g1_29dof/config/policy/run/<版本> --reference_steps 40
```

不要拿训练日志目录里的 `params/deploy.yaml` 直接用：那份是训练启动时导出的，`stiffness/damping` 是
env 0 抽到的随机增益（±10%），也没有 `gait` 和 `raw_clip` 段。

## 离线对拍（不需要机器人）

```bash
cd deploy/robots/g1_29dof/test && mkdir -p build && cd build && cmake .. && make
./test_run_policy ../../config/policy/run/<版本>          # 观测拼装 + ONNX 推理 vs Isaac Lab，逐位对比
./test_gait_command ../../config/policy/run/<版本>/params/deploy.yaml   # 手柄 -> 指令的行为测试
```

`test_run_policy` 通过的标准：观测最大差 < 1e-4，动作最大差 < 2e-3。任何观测项顺序、历史堆叠、
归一化的错误都会直接表现为 O(1) 的差异。

## 手柄

| 操作 | 作用 |
|---|---|
| FixStand / Velocity 下 RB + A | 进入 Run |
| 左摇杆前后 / 左右 | 前向 / 侧向速度 |
| 右摇杆左右 | 转向 |
| 按住 RT | 允许跑步：解除走路限速，支撑相可低于 0.5（腾空） |
| 松开所有摇杆 | 策略自己减速、支撑相滑到 1.0 站定 |
| RB + X | 回 Velocity，**只在站定后放行**（指令速度 <0.1、支撑相 >0.95、关节静止） |
| LT + B | 急停到 Passive，不设门槛；高速下切阻尼必摔 |

步态参数（步频、支撑相、摆高、躯干高、俯仰）按指令速度从 `config.yaml` 的 `FSM.Run.gait.presets`
插值，不需要手动调；指令经过与训练相同的斜率限制。

## 首次上机

1. `config.yaml` 里 `FSM.Run.gait.max_lin_vel_x` 保持 1.5，不按 RT，先当走路策略用，确认与 Velocity 表现一致。
2. 松开摇杆确认能站定；站定后 RB + X 能切回 Velocity。
3. 再逐步把 `max_lin_vel_x` 放到 3.0，按住 RT 试跑。
4. 真机比原厂模型重 12%，上机优先用 `RunWithId` 训出的策略（同样的导出流程，`--task Unitree-G1-29dof-RunWithId`）。
