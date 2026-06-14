# Humanoid Ultra MuJoCo Sim2Sim 使用说明

该独立运行程序用于在 MuJoCo 中测试通过 `unitree_rl_lab` 训练的
Humanoid Ultra 12 自由度和 27 自由度运动策略，不依赖 Unitree DDS 通信桥。

## 1. 导出 Isaac Lab 策略

进入 `/home/zxh/unitree_rl_lab`：

```bash
conda activate ustc_isaaclab

./unitree_rl_lab.sh -p \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --checkpoint "/home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/model_1000.pt" \
  --headless \
  --export_only
```

导出的 TorchScript 策略位于：

```text
logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/exported/policy.pt
```

导出 27 自由度策略时，需要使用对应的 27 自由度任务名称和实验目录。

## 2. 在 MuJoCo 中运行

本机 `gmr` Conda 环境已经安装了 `torch` 和 `mujoco`：

```bash
cd /home/zxh/ustc_humanoid/unitree_mujoco
conda activate gmr

python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 12 \
  --policy /home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/exported/policy.pt \
  --vx 0.3
```

运行 27 自由度策略：

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --vx 0.3
```

### 部署 Stand 策略

`USTC-Humanoid-Ultra-27dof-Stand` 的最终策略已经导出到：

```text
/home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra27dof_stand/2026-06-13_22-58-22/exported/policy.pt
```

在 MuJoCo 中运行：

```bash
cd /home/zxh/ustc_humanoid/unitree_mujoco
conda activate gmr

python sim2sim/humanoid_ultra/sim2sim.py \
  --mode stand \
  --dof 27 \
  --policy /home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra27dof_stand/2026-06-13_22-58-22/exported/policy.pt
```

Stand 模式默认关闭弹力带，与训练环境保持一致。需要吊带保护时显式添加
`--elastic-band`。

Stand 策略中的三维命令不是行走速度：

```text
height_command > 0    下蹲，最大 1.0
height_command < 0    站高，最小 -1.0
roll_command          躯干横滚，范围 [-0.5, 0.5]
pitch_command         躯干俯仰，范围 [-0.5, 0.5]
```

也可以从命令行指定初始姿态命令：

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --mode stand \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --height-command 0.4 \
  --roll-command 0.0 \
  --pitch-command 0.0
```

## 3. 键盘控制

首先点击 MuJoCo 可视化窗口，使其获得键盘焦点。

```text
W/S 或 上/下方向键       增加/减小前进速度
A/D                    增加/减小横向速度
Q/E 或 左/右方向键       增加/减小转向角速度
X/空格                  所有速度指令清零
7/8                    缩短/加长弹力带
9 或 B                  释放/重新连接弹力带
R                      重置机器人并恢复默认弹力带状态
```

使用 `--mode stand` 时，运动按键改为：

```text
W/上方向键              增大高度命令，即下蹲
S/下方向键              减小高度命令，即站高
A/D                    调节躯干横滚
Q/E 或 左/右方向键       调节躯干俯仰
X/空格                  三个姿态命令清零
```

每次按键会改变 `0.1` 的对应速度指令，并在终端输出当前的
`vx`、`vy`、`yaw` 和弹力带状态。

## 4. 弹力带

弹力带默认开启，但机器人双脚保持接触地面。两根肩带默认承担约 `30%` 的机器人
重量，主要用于防止策略失稳后直接摔倒。按 `9` 或 `B` 可以完全释放弹力带；
再次按下会重新连接弹力带。

弹力带采用左右肩部双吊点：

- 左吊点：`left_shoulder_pitch_link` 肩关节中心
- 右吊点：`right_shoulder_pitch_link` 肩关节中心
- 受力刚体：`trunk_link`
- 两个世界坐标系锚点分别位于左右肩关节中心正上方

弹力通过肩关节中心施加到躯干，因此不会直接拉动左右手臂关节。默认情况下两根
弹力带合计承担约 `30%` 的机器人重量，并会对躯干倾斜产生一定的恢复力矩。

不使用弹力带启动：

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --no-elastic-band
```

弹力带可调参数：

- `--band-lift`：机器人初始抬升高度，默认为 `0.0 m`
- `--band-anchor-height`：弹力带世界坐标系锚点高度，默认为 `3.0 m`
- `--band-stiffness`：弹力带刚度，默认为 `500.0`
- `--band-damping`：弹力带阻尼，默认为 `100.0`
- `--band-support-ratio`：肩带承担的整机重量比例，默认为 `0.3`

不建议在行走策略测试时将 `--band-lift` 设置得过高或将
`--band-support-ratio` 设置为 `1.0`。机器人完全离地后没有足底接触，行走策略
可能持续摆腿寻找支撑。

## 5. 无界面验证

不打开 MuJoCo 窗口进行策略测试：

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 12 \
  --policy /absolute/path/to/exported/policy.pt \
  --vx 0.3 \
  --headless \
  --duration 10
```

## 6. 策略接口参数

- MuJoCo 物理仿真步长：`0.005 s`
- 策略控制周期：`0.02 s`
- 动作缩放系数：`0.25`
- Actor 历史观测长度：10 帧
- 12 自由度策略输入/输出维度：450 / 12
- 27 自由度策略输入/输出维度：900 / 27

程序会显式处理 Isaac Lab 与 MuJoCo 的关节排列差异。策略观测和策略输出使用
Isaac Lab 运行时关节顺序，PD 控制输出再映射到 MuJoCo actuator 顺序。程序同时
使用训练 URDF 的关节位置限位和速度上限，避免因仿真器模型约束不同导致策略发散。

默认不执行固定姿态预热，即 `--stand-seconds 0.0`。Isaac Lab 在 reset 后也是立即
进入策略控制。将固定姿态保持较长时间可能使机器人在策略接管前已经开始倾倒。

必须使用自由度数量匹配的导出策略。不能将 `model_1000.pt` 等原始
RSL-RL checkpoint 直接传给该程序，必须先导出为 `exported/policy.pt`。

## 7. 辨识数据采集

`collect_identification_data.py` 按辨识交付规范采集严格等间隔的 `1 kHz` 数据。
采集时 MuJoCo 以 `1 ms` 积分，PD 力矩保持 `200 Hz` 更新，策略保持 `50 Hz`
更新。默认关闭肩部弹力带，避免未建模外力污染惯性参数辨识。

固定速度或可视化键盘采集：

```bash
python sim2sim/humanoid_ultra/collect_identification_data.py \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --duration 20 \
  --vx 0.3
```

自动多频速度指令采集，并在开头和结尾保留静止段：

```bash
python sim2sim/humanoid_ultra/collect_identification_data.py \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --duration 30 \
  --profile identification \
  --static-seconds 3 \
  --headless
```

每次采集会创建独立目录，并输出：

- `*_robot_low_q.dat`
- `*_robot_dq.dat`
- `*_robot_ddq.dat`
- `*_robot_tau.dat`
- `*_robot_contact.dat`
- `*_robot_ee_force.dat`
- `*_raw.csv`
- `*_metadata.json`
- `*_joint_mapping.csv`

DAT 中的关节顺序严格使用 URDF/Pinocchio 驱动关节顺序。CSV 带秒单位时间戳，
同时保留基座、IMU、关节、接触、足底六维力和速度指令。脚底力为环境施加给机器人的
作用力，使用世界轴对齐坐标系，力矩参考点为 `left_foot`/`right_foot` site。

`metadata.json` 会记录时钟来源、单位、坐标轴、IMU 外参、关节零偏、模型质量、
弹力带状态和自动质量检查结果。当前 `identification` profile 只生成多频
`vx/vy/yaw` 指令；它适合验证采集链路和腿部行走数据，不等同于完整的全身辨识激励。
若需要高质量辨识手臂、腰部惯量和摩擦，应使用专门的全身激励策略，并分别采集互不
重叠的训练轨迹与验证轨迹。
