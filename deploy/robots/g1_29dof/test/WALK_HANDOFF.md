# Jump3 → Walk：交接修复、诊断与验证

**2026-09-15 最新状态：实机交接尚未验证稳定。** 后续物理闭环补测仍存在失稳；首帧连续不能证明落地状态可恢复。新候选控制参数未启用，最新对照结果与解决步骤见 [恢复实验报告](../../../../artifacts/jump3_walk_handoff_20260915/recovery_v2/README.md)。

## 实现

`FSM.Velocity` 改用 G1 专用 `State_Walk`，状态名称、控制按键和模型路径沿用现有配置。

1. **配置配套**：补上 `velocity/params/deploy.yaml` 的 `raw_clip: [-10, 10]`。该 YAML 与当前模型对应的 `0914_yawfix/params/deploy.yaml` 语义一致。
2. **首帧交接**：保存退出 jump 时最后实际发送的 motor-order 关节目标。新线程未产生有效结果前，控制循环继续发送这一目标；新结果首次实际接入时插值权重为零。重新进入 walk 会清除上一轮候选目标和所有过渡计时。
3. **目标与指令过渡**：1 kHz 控制循环用五次 smoothstep 从保存的目标过渡到实时 walk 候选目标；过渡计时从首次结果被控制循环采用开始，避免推理延迟跳过过渡。速度指令先归零，再逐步恢复手柄值。
4. **历史衔接**：G1 主控制线程在每次 `pre_run` 后采集完整快照：q/dq、IMU、手柄、上一条已发送的关节目标。保留最近 256 个控制样本；walk 进入时按 20 ms 间隔选取 5 帧，并按原始导出配置构建 480 维观测。缓存不足时重复最早可用帧。
5. **上一动作的语义**：根据实际下发目标按 walk 的关节映射、offset、scale 反算 `last_action`，再施加模型的 raw_clip。过渡期间每一步同样使用已发送的混合目标，后台未执行的候选输出不会被当成实际上一动作。
6. **线程交接**：传感器历史、完整目标和 ActionManager 读写均有 mutex；Mimic 的运行标志、推理就绪标志、异常状态与完成时间改为 atomic，倾角检查读取工作线程发布的结果。Mimic 躯干姿态计算使用同一次传感器快照中的腰角。
7. **退出与异常**：退出唤醒等待线程，并调用 ONNX Runtime 的 RunOptions 取消接口，再 join；重新进入前清除取消标志。walk 推理/样本超时或异常走 Passive 检查。取消是 ONNX Runtime 的协作式取消，不构成独立硬实时看门狗。
8. **动作缓存与输入检查**：Action reset 同时重建物理目标，原始 action 的错误维度/非有限值会在写缓存前被拒绝；ONNX 输入维度不匹配会在创建输入张量前报错。

## 当前参数

参数位于 `config/config.yaml → FSM.Velocity.walk_entry`：

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| `blend_time_s` | 0.30 | 从最后发送目标过渡到 walk 候选目标 |
| `zero_command_time_s` | 0.40 | 首次接入后保持零速度指令的时间 |
| `command_ramp_time_s` | 0.50 | 随后渐进恢复手柄指令的时间 |
| `inference_timeout_s` | 0.20 | 工作线程结果/采样等待上限，触发故障检查 |

这些是首次修复的初始参数。后续参数与物理闭环扫描发现失稳，未把候选参数写入部署配置。`zero_command_time_s` 必须不小于 `blend_time_s`。

模型文件未修改，SHA-256：

- Jump：`d48f4cb3117873af1bb77f21ce12d88f79193852b556767cbc22328479636c62`
- Walk：`dac6e8f5bafa8f6b1b46a32502e7f5ed67489f565d66531ffedb99ab909462ad`

## 编译和测试

在仓库根目录执行：

```bash
cmake -S deploy/robots/g1_29dof -B deploy/robots/g1_29dof/build
cmake --build deploy/robots/g1_29dof/build -j 3

cmake -S deploy/robots/g1_29dof/test -B deploy/robots/g1_29dof/test/build
cmake --build deploy/robots/g1_29dof/test/build -j 2
ctest --test-dir deploy/robots/g1_29dof/test/build --output-on-failure
```

首次修复结果：**控制器编译成功；当时默认 CTest 4/4 通过。** 生成控制器为 `deploy/robots/g1_29dof/build/g1_ctrl`，未启动机器人控制或自动重启部署进程。

| 测试 | 验证内容 |
| --- | --- |
| `walk_handoff` | 延迟 80 ms 才得到首帧时保持原目标；首帧发送差为零；目标过渡、零命令窗口、命令恢复、重复进入、超时和异常；目标及状态快照并发压力测试 |
| `walk_policy` | 使用真实 ONNX；480 个观测元素逐位对拍；motor/policy 关节映射；raw_clip/reset；实际下发动作反馈；Jump3 CSV 末段代理输入；取消推理与再次使用；ActionManager 并发压力测试 |
| `gait_command` | 原有 Run 步态命令的站立、加减速、跑步档和轴方向测试 |
| `state_walk` | 使用生产 `State_Walk.cpp` 和实际 ONNX，但 Types.h 替换成没有网络能力的传感器/发布器；连续三次进入/运行/退出，验证首次发布目标与进入前最后发送目标完全相同 |

Jump3 末帧代理输入试验中，未经混合的 walk 候选目标与参考目标最大差 **1.28796 rad**；新流程首帧下发差为 **0 rad**。把这一个候选固定后，在 0.30 s 内过渡，最大相邻 1 ms 目标变化为 **0.00804955 rad**。这是针对固定候选的控制路径测试，不是动态候选的严格变化率上限，也不代表关节实际运动速度。

额外检查：

- 纯交接组件的 AddressSanitizer + UndefinedBehaviorSanitizer 测试通过。此环境下 LeakSanitizer 与 ptrace 不兼容，因此该次通过使用 `ASAN_OPTIONS=detect_leaks=0`，不包含泄漏扫描。
- ThreadSanitizer 在当前环境启动失败（`unexpected memory mapping`，含非 PIE 构建也相同），不能声称通过线程消毒器检查；常规并发压力测试已通过。
- 额外运行了原有 `test_run_policy`。当前 Run 模型/参考的对拍失败：最大观测差 `3.57628e-07`、动作差 `580.755`。从 Git HEAD 提取修改前部署头文件独立编译后，得到完全相同的误差与失败结果。这是已有的独立 Run 对拍问题；本次未修改 Run 模型或参考。其 CTest 注册需显式打开 `-DG1_TEST_RUN_REFERENCE=ON`，原有手动运行方式继续可用。

完整测试日志见仓库 `artifacts/jump3_walk_handoff_20260915/fix_validation.log`；修改前 Run 对拍日志见同目录 `run_reference_baseline.log`。

## 结果边界与下一步

首次修复只验证了部署交接机制；后续已经补做 MuJoCo 连续物理仿真，详见上方最新报告。助手没有进行实机试跳。现有动作仍按原始动作时长请求退出，尚未增加接触/稳定门槛，也没有修改或训练落地恢复尾段。

末帧仍处于前倾、回弹状态，walk 候选输出仍可能很大。目标插值及历史衔接不能证明该动态状态可恢复，单步候选也不能用于推断真实控制稳定性。下一步应验证同一机器人状态连续执行的 jump→恢复→walk，记录实际 q/dq/姿态、最后 jump 目标、首个 walk 候选及最终发送目标，再评估第二优先的恢复尾段。

## 第二优先（2026-09-15）：jump 落地站稳尾段 + walk 落地态抗扰

实机日志（`log/log.txt` 16:43）确认：切换瞬间骨盆倾角 0.592 rad、机身角速度 2.19 rad/s、腰俯仰实测 +17.5°
对应参考末帧的 +15°——机器人在正确跟踪一个"落地回弹中途结束"的参考（NPZ 末帧根部 vz=+0.41 m/s、
水平 0.52 m/s、躯干倾角 38.8°）。交接实现不是根因，混合参数不是杠杆。训练侧新增两个任务：

| 任务 | 位置 | 作用 |
| --- | --- | --- |
| `Unitree-G1-29dof-Mimic-Jump1-1mWithIdRecover` | `source/.../mimic/robots/g1_29dof/jump1_1mwithid_recover/` | 参考追加 0.16 s 刹车 + 0.64 s 收姿到 walk 站姿 + 1.0 s 保持；`motion_end_behavior="hold"`；30% reset 落在落地/恢复段。从 0915_28000 热启动 |
| `Unitree-G1-29dof-VelocityWithIdYawRobust` | `source/.../locomotion/robots/g1/velocitywithid_yaw_robust/` | 只改 events：初始倾角/角速度/关节速度随机、推力加角速度分量；观测/动作不变，ONNX 可直接替换。从 0914_yawfix 热启动 |

参考尾段由 `scripts/mimic/add_recovery_tail_g1.py` 生成（根部由双脚位姿 + 关节角正运动学反推，脚不滑不穿地），
再用 `scripts/mimic/csv_to_npz.py --no_ground` 重算刚体状态；命令和验收标准见 recover 任务目录的 README.md。
部署顺序：先换新 jump 策略 + `jump3_waist15_recover.csv`（旧策略不能配新 CSV），验证站稳后再考虑把
State_Mimic 的时长退出改成状态门槛（倾角/角速度/腿速连续 100~200 ms 达标）。


## 2026-09-15：实机失败后的诊断补充

本次合入只增加诊断；`walk_entry` 参数、策略 ONNX、上一动作编码、姿态阈值与超时阈值沿用原配置。候选限速、短混合、历史预热没有启用。

`FSM.Velocity.walk_trace` 默认启用，`directory: log/walk_handoff` 相对于 G1 部署目录解析，`duration_s: 3.0`。每次进入 Velocity 开始一个独立 CSV，以约 100 Hz 记录：

- q/dq（29 个电机）、IMU 四元数/角速度、手柄命令；
- previous_target（上一控制目标）、candidate（walk 最新候选）、applied（本轮控制目标）；
- 推理结果年龄、控制快照年龄、推理耗时、命令恢复比例、两次采样间最大单次控制目标步长；
- event：0 样本、1 进入、2 结束、3 故障；reason_mask 为故障位或组合。

控制线程只向固定 512 行的队列投递，后台线程写盘。队列满丢记录，退出状态只投递结束记录；进程析构时后台线程排空。文件创建/写入失败会单独报错。`applied` 不代表硬件回执或实测力矩，`frame_age` 不等于 DDS 网络包年龄。

| reason / mask | 含义 |
| --- | --- |
| inference_failed / 1 | 推理工作线程异常，关联前一条异常日志 |
| first_result_timeout / 2 | 首个结果超时 |
| result_timeout / 4 | 已有结果过期 |
| no_frame / 8 | 没有控制快照 |
| invalid_frame / 16 | 非有限输入、无效四元数等 |
| frame_timeout / 32 | 控制快照过期 |
| orientation / 64 | 骨盆倾角超过原有 1.0 rad 阈值 |
| lowstate_timeout / 128 | 原有 lowstate 看门狗超时 |

手柄配置触发 Passive 单独记录 `reason=configured_transition condition=...`。自动故障可以同时含多个原因；原因不是互斥的。

新增 `walk_trace` 测试验证完整 CSV、故障记录及退出排空；`state_walk` 补充真实状态类的倾角故障原因断言。测试使用无网络的假传感器/发布器和实际 ONNX。
