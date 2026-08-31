# 平滑踢腿顶点的参考:`ustc1_rightstand_stand_transition_smoothapex.npz`

原参考 `ustc1_rightstand_stand_transition.npz` 在踢腿顶点有一段重定向噪声:
左脚在 9.56 s 到 0.542 m 峰值后掉到 0.422 m,再回抬到 0.487 m 才落地——机器人
上看到的"抬两次腿"。同一段(8.9–11.9 s)也占 tightroll V1 训练全部失败的 72%。

本文件只把这一段换成平滑轨迹,其余部分逐位不变,用于和原参考做 A/B。

## 改了什么

只有 **477–523 帧(9.54–10.46 s,49 帧 / 941 帧)** 的数据不同。窗口内用
最小二乘三次多项式重新拟合根位姿和 27 个关节角,并用 smoothstep 与原轨迹
交叉淡入淡出(ramp 10 帧),因此窗口边界一阶连续。

| 指标(左脚) | 原参考 | 平滑后 |
| --- | ---: | ---: |
| 抬腿峰值 | 0.542 m | 0.543 m |
| 峰值时刻 | 9.56 s | 9.56 s |
| 顶点回抬量(rebound) | **0.067 m** | **0.000 m** |
| 顶点区局部极大个数 | 5 | 0 |
| 顶点区方向反转次数 | 10 | 0 |
| 落地时刻 | 10.52 s | 10.52 s |
| 窗口内 max\|joint_vel\| | 2.93 rad/s | 2.86 rad/s |

抬腿高度、起跳和落地时刻都保住了,只有顶点从"上-下-上"变成单峰后平滑下落。
根位姿必须一起平滑:只平滑关节的话,骨盆自身的抖动仍会让脚高出现 4 个局部
极大(工具的 `--keep-root` 可复现这一对照)。

## 一致性保证

* 窗口外(477–523 帧之外)每个数组都与原文件**逐位相同**,已校验。
* `joint_vel` 在原文件中就是 `joint_pos` 的中心差分(误差 0.0000),根的
  `body_lin_vel_w[:,0]` / `body_ang_vel_w[:,0]` 分别是根位置的中心差分和
  SO(3) 中心差分(误差 3e-6 / 1.1e-5),新文件按同样定义重新生成。
* 各刚体位姿/速度用 `scene_27dof_identified.xml` 的 MuJoCo 正运动学重算;该
  正运动学能把原文件的 `body_pos_w` / `body_quat_w` 复现到 1e-6 m / 2e-7,
  新文件自身的正运动学一致性同样是 1e-6 m。
* 所有 27 个关节仍在部署关节限位内。

## 生成命令

```bash
cd /home/zxh/unitree_rl_lab
conda run -n gmr python scripts/mimic/smooth_motion_apex_humanoid_ultra.py \
  --input  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand_stand_transition.npz \
  --output source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand_stand_transition_smoothapex.npz
```

默认参数即 `--f0 476 --f1 524 --ramp 10 --degree 3`。

## MuJoCo 冒烟测试

用 **在原参考上训练的** V1 tightroll 55500 策略跑新参考(headless,941 帧):
没有跌倒,末态 base_z 1.005 m(原参考 1.004 m),最低 base_z 0.985 m
(原参考 0.977 m),脚高回抬量 0.007 m → 0.002 m,顶点变成单峰。

这只说明文件能被现有 pipeline 正常播放、且策略不会立刻崩;它不是对平滑参考
的公平评测——要看效果得在新参考上重训。

## 怎么用

`tracking_env_cfg.py` 里的 `motion_file` 是写死的文件名(第 358 行),新文件
不会被任何现有 task 自动加载。做 A/B 时新注册一个 task,把该 task 的
`motion_file` 指到 `ustc1_rightstand_stand_transition_smoothapex.npz`,其余
配置与对照 task 保持一致。
