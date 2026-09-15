# jump3 → walk 闭环检查（2026-09-15）

## 结论

**发现并修复了一个确定的 C++ 观测计算缺陷；当前仍不能认定现场的切换跑飞已经解决。**

修正该缺陷后，12 组 MuJoCo 参数/扰动测试、3 组完整 walk→jump→walk 流程都恢复站立，没有触发倾倒保护。但交接时姿态尚未稳定，walk 候选目标与当前状态差异很大，混合阶段之后仍有目标台阶。它们值得继续处理，不能仅凭本轮有限成功样本宣称稳定。

本次没有取得用户实际失败时的状态/目标日志，因此无法确认现场失稳的根因。仓库 `deploy/robots/g1_29dof/log` 当前不存在。新控制器已经加入交接日志，下一次仿真复现可直接记录。

## 已写入部署代码的修改

1. `deploy/robots/g1_29dof/src/State_Mimic.cpp:51`：

   ```cpp
   // 原代码：auto 保存惰性转置表达式，底层临时矩阵已经销毁
   auto rot = rot_.toRotationMatrix().transpose();
   // 修复：持有计算后的矩阵
   const Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();
   ```

   最小复现的 AddressSanitizer 明确报告 `stack-use-after-scope`；修正形式通过。在未修正的 Release 仿真程序中，jump 首帧关节目标曾达到约 4.3e3 / 1.1e33 rad，尚未到切换就失效；只替换这一行后，同样的物理模型与策略能完成 jump 和交接。这是确定的代码缺陷，但不是对用户描述的“jump 成功后切 walk 跑飞”的直接复现。现有部署构建未启用优化，未定义行为可能有不同表现。

2. `State_Walk.cpp` 新增进入时的骨盆倾角、陀螺仪、最大腿部关节速度、腿部目标误差、腰俯仰实测/上一目标，以及首帧候选目标差异和最差电机编号。`max_kp_delta` 是候选目标差乘 Kp，**不是实际施加的扭矩**。

`build/g1_ctrl` 已重新编译。未更换 ONNX，未调整部署中的 0.30 s 混合参数。0.10 s 只用于仿真对照。

## 测试方法和边界

- MuJoCo 3.10.0 CPU 物理仿真，1 ms 物理/主控制周期；使用生产 `State_Mimic`、`State_Walk`、观察/动作管理器与 C++ ONNX Runtime 1.22.0。
- 测试驱动负责物理推进、状态检查和自动交接；传感器/命令用测试适配器连接 MuJoCo，替代 DDS。**没有启动实机控制器，没有机器人网络 I/O。**
- 实际部署模型：jump3 `0915_28000`，walk `0914_yawfix`，完整 SHA256 见 `manifest.json`。
- 每次交接保留物理位置、速度和控制历史，不重置机器人，不吊挂，不使用弹力带。
- 两种私有模型基于现有 `g1_29dof_identified0907.xml`，将左手惯量替换为训练使用的 `identified0914_ball.urdf`：`sdk` 保留原模型被动关节参数和力矩上限；`matched` 对齐训练的力矩上限及 armature，并将被动阻尼/摩擦设为 0。`matched` 仍不是与 IsaacLab 完全一致的物理环境。
- 额外完整流程直接加载用户 `simulate/config.yaml` 指向的原始 `scene_identified0907.xml`，没有替换左手惯量。
- 起始参考姿态测试先使用 CSV 第一帧，初始速度为零；完整流程则从 walk 默认站姿开始，先运行 walk 2 s，再触发 jump。
- 用户 GUI 仿真使用的本地引擎库是 MuJoCo 3.3.6；本轮没有复现 GUI/DDS 的线程竞争、通信延迟、弹力带状态或真实电机动力学。这些差异限制结论。
- 实时推理线程使用墙上时间；成功测试最大主循环滞后约 4–10 ms。结果不是严格确定性的重复实验。

## 结果

| 测试 | 结果 |
|---|---|
| walk 单独运行 6 s | 正常站立 |
| 修正后 jump 参考初态，sdk / matched，各 6 s | 2/2 完成交接 |
| 2 种模型 × 0.30 / 0.10 s 混合 × 无扰动 / 正向 / 反向扰动，各 6 s | 12/12 恢复站立 |
| 先 walk 2 s → jump → walk，总时长 8 s，sdk / matched / 用户原始场景 | 3/3 恢复站立 |
| 现有 CTest 回归 | 4/4 通过 |
| 矩阵临时对象生命周期 ASan 最小复现 | 原写法失败，修正写法通过 |

扰动在切换时施加：沿身体前向线速度增加 ±0.25 m/s，同时身体俯仰角速度增加 ±0.75 rad/s。它是指定的仿真假设，未声称来自实机记录。通过标准包含无 Passive 触发、无倒地，并检查最终高度和躯干倾角；不是成功率统计或稳定性证明。

完整流程的主要数值：

| 模型 | 切换躯干倾角 | 切换后最低根高度 | 最终躯干倾角 | 最大单次目标变化 |
|---|---:|---:|---:|---:|
| matched | 37.24° | 0.726 m | 2.94° | 0.453 rad |
| sdk（0914 手惯量） | 36.57° | 0.721 m | 2.86° | 0.665 rad |
| 用户原始场景（0907） | 36.99° | 0.723 m | 2.75° | 0.305 rad |

“单次目标变化”是 1 ms 相邻控制记录的 **q_target** 差，不是实测关节在 1 ms 内运动了相同角度，也不是切换第一帧的跳变。

例如 sdk 完整流程中：进入 walk 时腰俯仰实测 +0.118 rad（+6.8°），上一目标 −0.173 rad，walk 首帧候选 −0.737 rad（−42.2°）。候选值先经过混合后才下发。当前混合只让权重连续，候选目标仍以 50 Hz 更新，因而不保证下发目标的变化率连续。

参考动作结束仍包含前倾和运动，当前自动切换只看时长。下一步应依据现场日志区分：

- 进入 walk 前已经有过大的倾角/角速度/关节速度；
- walk 首帧候选过激，及后续下发目标台阶引起额外冲击；
- 推理/状态时延或运行的二进制、策略、物理模型与本测试不同。

不能在缺少证据时单纯加长混合或冻结 jump 末帧：这也会延后 walk 的反馈调整。本轮 0.10 s 对照恢复前倾较快，但不足以据此替换实机参数。

## 资料与复现

- [曲线](handoff_diagnostics.png)、[仿真视频](jump3_walk_nominal.mp4)：视频为无扰动参考初态测试的物理轨迹回放，不是实机视频。
- `metrics.json`：所有测试数值；`traces.tar.gz`：完整 CSV 和日志。
- `sim.cpp`、`State_Mimic_fixed.cpp`、`State_Walk_diagnostic.cpp`、`CMakeLists.txt`：仿真驱动和本次测试的状态源码快照。测试适配器与公共库读取现有仓库。
- `make_model.py`：生成私有模型；`run_matrix.py`：12 组试验；`analyze.py`、`render.py`：分析和回放。
- `regression_results.log`、`eigen_lifetime*.log`、`manifest.json`：验证记录与版本校验。

在本目录中：

```bash
PY=/home/ustczxh/miniconda3/envs/ustc_identification/bin/python
$PY make_model.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
./build/sim scene_sdk.xml sequence.csv sequence 0.30 8
$PY run_matrix.py
```

仿真侧复现现场问题时，重启新编译的控制器并开启日志：

```bash
cd /home/ustczxh/humanoid/zxh-isaaclab/deploy/robots/g1_29dof
./build/g1_ctrl --network lo --log
```

日志写入该项目的 `log/log.txt`。关注跳跃之后的 `Velocity entry measured`、`Velocity first inference ready` 和任何 fault/Passive 信息。下一步需要这一段实际失败日志来对齐；本轮尚未取得。
