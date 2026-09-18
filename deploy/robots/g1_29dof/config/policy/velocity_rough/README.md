# velocity_rough：感知版 walk（高程图进 actor）

对应训练任务 `Unitree-G1-29dof-PerceptiveHeightScan`。观测 = 线上 walk 的六项 × 5 帧 + 187 维高程图 × 1 帧，
动作、步频、限速、切换逻辑与 `Velocity` 状态完全相同（都是 `State_Walk`）。

## 1. 导出策略到这里

```bash
python scripts/rsl_rl/export_deploy.py --task Unitree-G1-29dof-PerceptiveHeightScan \
    --load_run <run 目录> --checkpoint model_15000.pt \
    --out deploy/robots/g1_29dof/config/policy/velocity_rough
```

产物 `exported/policy.onnx`（输入 667 维）和 `params/deploy.yaml`（第 7 项观测 `height_scan`，`history_length: 1`，
`clip: [-1, 1]`，`params.offset: 0.5`）。`State_Walk` 启动时会校验这个布局，并据此要求 `config.yaml` 里配好 `height_map`。

## 2. 打开 FSM 状态

`config/config.yaml`：

1. 把 `FSM._` 里注释掉的 `Velocity_Rough: {id: 6, type: Walk}` 放开。
2. 把 `FixStand` / `Velocity` 里注释掉的 `Velocity_Rough: RB + B.on_pressed` 放开。
3. `FSM.Velocity_Rough.height_map` 已经写好，`grid` 必须和训练的 `HEIGHT_SCANNER_CFG` 一致（17×11 @ 0.1 m）。

进入：FixStand 或 Velocity 下 `RB + B`；退出：`RB + X` 回 Velocity，`LT + B` 急停。

## 3. 高程图消息约定（建图节点要发什么）

控制器订阅 Unitree DDS 话题 `rt/perceptive/height_map`，消息类型复用 SDK 自带的
`unitree_go::msg::dds_::HeightMap_`（Python：`unitree_sdk2py.idl.unitree_go.msg.dds_.HeightMap_`），不需要新 IDL。

| 字段 | 值 | 说明 |
|---|---|---|
| `resolution` | `0.1` | 格距 [m] |
| `width` | `17` | x 方向格数（前后 −0.8 … +0.8 m） |
| `height` | `11` | y 方向格数（左右 −0.5 … +0.5 m） |
| `origin` | `[-0.8, -0.5]` | 格 (0, 0) 的中心在躯干 yaw 系里的 (x, y) |
| `data[iy * width + ix]` | 地面高度 − 躯干高度 [m] | x 变化最快（Isaac Lab GridPattern "xy" 序）；平地约 −0.78；未知格填 NaN |
| `stamp` / `frame_id` | 任意 | 不检查；新鲜度按控制器收到的时刻算 |

坐标系：原点在 `torso_link` 投影到地面的位置，x 朝机器人前方、y 朝左，只跟随 yaw、不跟随 roll/pitch
（训练里 `ray_alignment="yaw"`）。躯干高度 = `torso_link` 原点的 z。格 (ix, iy) 的中心 = `origin + (ix, iy) * resolution`。

控制器侧（`include/height_map_gate.h`）的处理：

* 没收到 / 超过 `timeout_s`（0.3 s）没有新图 / 尺寸、分辨率、原点不一致：整张图按平地填充
  （每格 = −`nominal_torso_height`），只告警一次，**不切 Passive**（楼梯中途切阻尼比"以为是平地"更危险）。
* 单个格子 NaN 或 |值| > `max_abs_height`：该格按平地填充。
* 观测 = −value − 0.5，再 clip 到 [−1, 1]，与训练一致。

建图节点建议：Mid-360 + DLIO/FAST-LIO 定位 + elevation_mapping_cupy 出机器人中心高程图，
再按上表采样 17×11 网格发布，≥ 5 Hz（10 Hz 更好）。相对高度用建图坐标系里的 `torso_link` z 减即可，
不依赖绝对高度。发布前最好把小面积的未知格做一次最近邻填补，减少平地填充造成的"假坑/假台阶"跳变。

## 4. 离线验证

`deploy/robots/g1_29dof/test/` 里三个测试覆盖这条通路（合成策略夹具，不需要真策略）：

```bash
cd deploy/robots/g1_29dof/test && mkdir -p build && cd build && cmake .. && make && ctest
```

* `height_map_gate`：门的 fallback / live / stale / invalid 和逐格清洗。
* `walk_policy_rough`：667 维观测布局、前 480 维与线上盲走策略逐项一致、高程图 clip、缺图拒绝。
* `state_walk_rough`：真 `State_Walk` + 假 DDS，配置校验、状态切换日志、有图 / 无图两种进入。

换成真策略后跑一遍 `test_walk_policy_rough <本目录> config/policy/velocity` 看布局是否仍然一致。
