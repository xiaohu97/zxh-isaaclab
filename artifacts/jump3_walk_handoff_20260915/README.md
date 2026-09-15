# Jump3 → Walk 自动切换分析（2026-09-15）

> 本文保存修复前的分析快照。随后完成的第一优先修复与测试见 [WALK_HANDOFF.md](../../deploy/robots/g1_29dof/test/WALK_HANDOFF.md)，测试日志见 [fix_validation.log](fix_validation.log)。

## 结论与证据边界

用户现象：jump3 真机执行成功，动作结束自动切到 walk 的瞬间跑飞。

当前证据最支持：**参考动作在落地回弹中结束，FSM 随即让 walk 冷启动接手；姿态、速度和动作历史没有完成交接。** 同时存在旧目标可能被发布的线程时序缺陷，以及 walk 模型与 YAML 不完全配套的问题。

本分析检查本地源代码、实际配置、模型 SHA-256，并执行了参考状态下的单步 ONNX 推理。没有实机切换日志，也没有运行闭环物理仿真，因此不能把参考状态当成实机状态，不能认定哪个缺陷在这次实机失败中先触发。

部署源码、YAML、ONNX 均未修改。本目录只保存分析脚本、数据和图。

## 1. 实际加载的内容

- FSM `Mimic_Jump3`：`config/policy/mimic/jump3`，参考 `jump3_waist15.csv`，120 fps。
- jump3 根目录 ONNX 与 `exported/0915_28000/policy.onnx` 的 SHA-256 相同。
- FSM `Velocity`：`config/policy/velocity`。
- walk 根目录 ONNX 与 `0914_yawfix/exported/policy.onnx` 的 SHA-256 相同。
- 两者关节映射、PD 增益、20 ms 策略周期一致；默认姿态最大差仅 0.0098 rad。
- 动作 scale 不同：walk 全部为 0.25，jump 按关节为 0.0745～0.548。这本身正常，但上一动作不能直接跨策略复制。
- walk ONNX 输入 480 维（96 维 × 5 帧）；jump 输入 154 维，不应直接复制观测向量。

## 2. 动作末尾不是稳态

`State_Mimic.cpp:83` 只检查 `episode_length * step_dt > motion_loader->duration`，直接返回 Velocity，没有接触、倾角、速度或保持时间条件。

CSV 共 214 帧，最后时间戳 1.775 s；loader 用 N/fps 得到 1.7833 s，策略计时通常在 1.80 s 满足退出条件。这是策略步数计时，实际墙钟时刻还受推理和调度影响。

末帧参考状态如下，速度来自参考差分，非实机测量：

| 项目 | 数值 |
| --- | ---: |
| 骨盆 pitch | 23.60° |
| 躯干相对竖直倾角 | 38.79° |
| 腰 pitch | 15.00° |
| 根部水平速度大小 | 0.516 m/s |
| 根部竖直速度 | +0.403 m/s |
| 腿部最大关节速度 | 3.571 rad/s |
| 腿部与 walk 默认姿态最大差 | 0.702 rad |

训练 NPZ 末帧也有约 0.515 m/s 的水平速度和 +0.407 m/s 的竖直速度，支持该现象源于动作片段本身，而不只是 CSV 插值误差。

![参考末段状态](reference_end.png)

两套策略的 default pose 接近，不能推导出动作结束姿态接近。真正需要对齐的是交接时的实际姿态、速度、支撑状态和下发目标。

## 3. Walk 接手路径的具体问题

### 3.1 首次推理之前可能发布旧目标

`deploy/include/FSM/State_RLBase.h:15` 的 `enter()` 启动后台线程，reset 和第一次推理都在线程内。FSM 主循环每 1 ms 执行 `run()`，立即读取 `processed_actions()`。

`deploy/include/isaaclab/envs/mdp/actions/joint_actions.h:78` 的 reset 只清 `_raw_actions`，不清 `_processed_actions`。Velocity 状态对象从初始化后一直复用，因此 jump 返回 walk 时，首次推理完成前读到的可能是起跳前退出 walk 时留下的目标。程序首次进入该状态还可能读到构造时的零目标。

目标数组跨线程读写缺少同步；`policy_thread_running` 和 Mimic 的 episode 计数也有跨线程访问问题。修正时应采用有效帧标志、完整目标快照、mutex/双缓冲以及 atomic 状态，而不是只延迟若干毫秒。

### 3.2 动作与观测历史冷启动

`ManagerBasedRLEnv::reset()` 清空 action；`ObservationTermCfg::reset()` 用当前一帧重复填满历史。常规 episode reset 采用首帧填充并非独有错误，[Isaac Lab CircularBuffer 源码](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/utils/buffers/circular_buffer.html)也采用这一机制。这里的问题是把高速落地中途当成 episode 起点。

5 帧间隔 20 ms，首尾跨度约 80 ms。接手时 walk 没有这段真实运动历史，last_action 却全部是 0；重力/关节位置/速度和动作历史不对应实际控制过程。

可以候选验证的交接办法：在 jump/recovery 仍控制时，按 walk 定义维护独立历史缓冲；用实际已发布的物理目标（先按 motor id 映射）计算 walk 表示下的上一动作：

`a_prev_walk = (q_cmd_previous - offset_walk) / scale_walk`

不能复制 jump 的原始 action。也不能把后台 walk 未执行的推理输出冒充真实已执行动作。上述映射只是控制语义更连贯的候选，仍须通过闭环验证，不能保证网络在跳跃状态下稳定。

### 3.3 单步离线推理已显示大幅纠姿

用 CSV 末帧 q、差分 dq、四元数和差分角速度近似传感器状态，摇杆指令设为零，按实际 YAML 的观测缩放、项顺序和 5 帧 reset 方式构造输入，用 ONNX ReferenceEvaluator 推理：

| 输入场景 | 腰 pitch 输出目标 | 腿部目标与输入姿态最大差 |
| --- | ---: | ---: |
| walk 默认姿态、静止、上一动作 0 | −3.56° | 0.235 rad |
| jump 末帧、上一动作 0 | −44.85° | 0.631 rad |
| jump 末帧、用参考关节位置反算上一动作 | −58.00° | 0.364 rad |

末帧腰参考为 +15°。第一种末帧输入下，网络目标相对输入腰姿态相差约 −59.85°。这不是“jump 最后命令到 walk 第一命令”的实测差值，因为缺少 jump 的实际下发目标；它展示的是 walk 对此类接手状态可能给出的激烈输出。

第三行还说明：**只补 last_action 不足以解决，某些关节输出反而更大。** 不能据此把单步误差小等同于闭环更稳定。

### 3.4 Walk 缺少训练时的 raw_clip

根目录 YAML 与 `0914_yawfix/params/deploy.yaml` 语义比较，差异是漏了 `actions.JointPositionAction.raw_clip: [-10, 10]`。训练来源 `0914_yawfix/params/source.txt` 明确记录 `clip_actions: 10.0`。

应配套使用该模型的完整 YAML。当前 C++ ActionManager 已实现 raw_clip，且裁剪结果会进入 last_action。

不过上述末帧试验最大 raw action 仅 3.13，低于 10，因此补回 raw_clip 不会改变该例的 −44.85° 腰目标。raw_clip 限的是网络数值，不等于物理关节目标限位。物理目标/力矩/变化率保护需要独立设计，并在仿真及后续训练中保持一致，不能仅复制一个位置 clip 后宣称切换已修好。

## 4. 推荐解决顺序

### 第一层：修复交接实现，消除确定的配置与时序缺陷

1. 给当前 walk 配套完整 `0914_yawfix` YAML。
2. 切换时保存最后实际下发的 29 关节目标；新策略第一帧有效结果未就绪前保持该目标，避免读取旧 walk 缓存。用完整快照跨线程发布，保持 1 kHz 控制循环运行。
3. 增加可观测的 walk-entry 阶段：起始命令为零，结果就绪且通过有效性检查后，在已经稳定的状态下尝试 0.2～0.4 s 目标平滑过渡；随后再渐进放开摇杆。该时间是仿真扫描范围，不是已验证参数。
4. 从 walk 观测定义构造交接历史；对“零历史、真实历史、等效上一动作”的候选做闭环对比。当前增益相同，优先解决物理 q_target 连续性；原始 action 不适合直接混合。

### 第二层：补齐 Jump 的恢复收尾——本例最关键

建议流程：`Jump → Landing/Recovery → WalkEntry → Walk`。

当前 CSV/NPZ 没有站稳结尾。需要在原轨迹后构造可实现的恢复段：脚保持合理支撑、骨盆和躯干回正、关节和根部速度趋零，结束姿态接近同一负载下 walk 零指令的稳态范围。先在仿真尝试 0.6～1.0 s 恢复，加 0.4～0.6 s 保持；时间长短应以闭环稳定结果为准。

从已成功的 jump checkpoint 微调，重点训练末段，同时混合原起跳/腾空片段，检查跳跃性能是否保留。参考末段的所有状态字段（q、dq、body pose、linear/angular velocity）要一致；只插值关节角可能造成脚滑、穿地或改变支撑。

当前任务未覆盖 `motion_end_behavior`，共享默认是 `resample`：到末帧会重新采样并写入仿真机器人状态（`tasks/mimic/mdp/commands.py:420`）。这种回放不证明机器人会自然收稳。应在训练/评估中覆盖恢复段和末尾保持，并用同一状态连续运行到 walk，不能在切换点重置机器人。

**不能只增加 timeout：** C++ MotionLoader 的末帧 dq 复制前一差分，并不是零。冻结最后索引仍会持续输入约 3.57 rad/s 的末帧参考速度。延长等待需要一个已验证的恢复控制器或重新训练的静止尾段；只改 fps 会同时改变整个跳跃的时序。

### 第三层：按实测状态允许交接

自动结束先发出“请求退出”，满足落地稳定条件后才接手；手动退出到 walk 也走同一条件。急停优先级应独立保留。

可作为仿真起点的条件：可靠的双脚支撑、骨盆/躯干倾角回到 walk 可恢复范围、机身角速度下降、腿部速度下降，连续满足 0.1～0.2 s。举例可先扫描倾角 0.15～0.25 rad、角速度 0.3～0.5 rad/s、最大腿关节速度 0.5～1.0 rad/s；这些不是已校准上机阈值。若有可靠速度估计，再约束水平/竖直速度。零摇杆不是速度已归零的证据。

当前 `unitree_articulation.h` 只更新关节、IMU 姿态和角速度，没有足接触或机身线速度估计。不能直接用未更新字段作为上述门槛。没有可靠接触估计时，可先评估姿态与腿部运动的持续条件，但不能把它当成真正的落地检测。

等待超时应进入已验证的恢复/故障处理路径，不能超时后再次强切 walk，也不能无限冻结当前动态末帧。

### 若仍需在明显运动中快速切换

给 walk 增加来自 jump 实际落地 rollout 的初始状态/历史分布，训练从前倾、回弹、单/双脚过渡和同一手持负载下恢复。姿态、速度、支撑和历史需要一致采样。单纯加大随机推力或降低 walk 速度指令，未必覆盖这类接手状态。

Jump 新版使用 0914 球体惯量模型，walk yawfix 使用 0907 负载模型；如果真机仍携带对应球体，应同步核对球的质量、质心和惯量。这是次级分布差异，不能替代前面的交接修复。

## 5. 怎样判定解决了

记录切换前 0.5 s 和后至少 1 s：FSM/切换原因、实际 q/dq、骨盆与躯干姿态、角速度、摇杆/实际速度指令、jump 最后 q_cmd、walk 第一 raw_action/q_cmd、裁剪和平滑后的实际下发目标、推理就绪时间及延迟。

这些记录可以区分：首次推理前旧缓存被发布、首次推理本身激烈纠姿、或者落地状态已在接手前失稳。当前日志选项主要打印状态变化，没有上述连续信号。

先修配置与线程交接，再做同一机器人状态连续的 jump→recovery→walk 仿真对比；评估落地成功率、切换后 2 s 的站稳率、目标/估计 PD 力矩突变、倾角峰值以及原跳跃性能。只有上述闭环结果改善，才有依据评估真机复测。

## 复现

在仓库根目录运行：

```bash
/home/ustczxh/miniconda3/envs/ustc_isaaclab/bin/python artifacts/jump3_walk_handoff_20260915/analyze.py
```

输出：`metrics.json`（包含完整单步输出、模型哈希）与 `reference_end.png`。脚本不会连接机器人或修改部署文件。
