# 感知 walk 的实机高程图链路（D435i → rt/perceptive/height_map）

`Velocity_Rough`（感知版 walk，`State_Walk` + `velocity_rough` 策略）需要控制器在
`rt/perceptive/height_map` 上收到 17×11 的高程图。**机器人出厂不发这个话题**，这里的两个节点把它补上。

## 为什么是这套架构（2026-09-18 排查结论）

- **Mid-360 看不到近场**：实测点云俯仰只到向下 7°，装在头部离地 ~1.2 m，要 ~9.7 m 外才够到地面；
  策略要的身周 1.6×1.0 m 完全无回波。机器人自带的 `rt/gridmap`/`rt/ele_clouds` 当时也全空，
  且同源于这颗雷达。所以近场地形只能靠 **D435i**（确认在 Jetson 上，USB3，深度 30 Hz 正常）。
- **两台机器能力互补**：Jetson（192.168.123.164）有 ROS Noetic + realsense 驱动，但无 DDS python、无外网
  （cyclonedds 无 aarch64 轮子难装）；dev 机（跑 g1_ctrl 那台）有 cyclonedds、在 DDS 网上。
  所以：**Jetson 只取深度并 TCP 发出（哑节点），dev 机做所有位姿几何 + 发 DDS**。

```
Jetson .164                                   dev 机 .222（跑 g1_ctrl）
realsense2_camera(ROS 深度 640x480@30)         cyclonedds 收 rt/dog_odom(48Hz)
  → jetson_depth_sender.py                TCP    → height_map_mapper.py
     深度→光学系点云→降采样→发     ───5601──→       静态外参→重力调平→yaw对齐
     (rospy+numpy+socket,无需装)                    →odom系中位数累积记忆→采样17×11
                                                    → 发 rt/perceptive/height_map
```

## 文件

| 文件 | 跑在哪 | 作用 |
|---|---|---|
| `run_camera.sh` | Jetson | 启动 realsense 深度流（仅深度，省带宽） |
| `jetson_depth_sender.py` | Jetson | 深度→光学系 3D 点→TCP 发 dev 机。哑节点，不含位姿几何 |
| `height_map_mapper.py` | dev 机 | 收点 + 订 odom → 建图 → 发 `rt/perceptive/height_map` |
| `dds_idl.py` | dev 机 | 内联 DDS 类型（HeightMap_/Odometry_），不依赖 unitree_sdk2py |

## 一次性:把两个 py + dds_idl 拷到 Jetson

dev 机没外网到 Jetson 之外，用 scp（Jetson 用户 unitree）：
```bash
scp jetson_depth_sender.py unitree@192.168.123.164:~/
# dds_idl.py 只有 dev 机用；Jetson 不需要
```

## 启动（每次）

**Jetson 上开两个终端：**
```bash
# 终端 1：相机
bash ~/run_camera.sh            # 或见 run_camera.sh 里的 roslaunch 原样

# 终端 2：深度发送
source /opt/ros/noetic/setup.bash
python3 ~/jetson_depth_sender.py --dev-ip 192.168.123.222 --port 5601
```

**dev 机上（跑 g1_ctrl 那台）：**
```bash
cd deploy/robots/g1_29dof/tools/perceptive_mapping
python3 height_map_mapper.py --iface enp5s0 --print
# --print 每秒打印 ASCII 高程图；--iface 是连机器人的网卡
```
`--iface` 会自动写 `CYCLONEDDS_URI` 绑到该网卡。也可 `--out-topic rt/perceptive/height_map_test`
先发到测试话题，不碰真控制话题。

## 标定（第一次上机必做）

机器人 **FixStand 站在已知平地上**，两边节点都起好：
```bash
# dev 机，收几秒数据后 Ctrl-C，它会给出 torso-z 偏置
python3 height_map_mapper.py --iface enp5s0 --calibrate-flat
```
它打印 `→ 把 --torso-z-over-body 设为 X.XXX`，之后每次带上 `--torso-z-over-body X.XXX`，
使平地 value ≈ −0.78（控制器和训练的口径）。若报告里 p10/p90 相差 > 0.05，说明相机俯仰外参可能有偏，
前后格高度不一致，需要微调 `MOUNT_PITCH`（见 height_map_mapper.py 顶部常量）。

> 默认 `torso_z_over_body=0.111` 只是占位（实测 odom robot_center z≈0.669，躯干高 ~0.78 → 差 ~0.111），
> 必须用 `--calibrate-flat` 在你的机器人上重新标定。

## 验证（切 Velocity_Rough 之前）

1. dev 机 `--print`：平地应处处 ≈ −0.78；面前放个箱子/台阶，对应格应抬高约其真实高度。
2. 用 echo 工具确认话题内容：
   ```bash
   python3 ../height_map_echo.py --interface enp5s0 --topic rt/perceptive/height_map
   ```
   频率接近 50 Hz、17×11、origin(-0.8,-0.5)、平地 ≈ −0.78。
3. g1_ctrl 终端正常应打印 `Height map live`；出现 `none/stale/invalid` 就是没接上/超时/网格不符，
   **此时不要切 Velocity_Rough**。

## ★安全

- **没有本链路或它停了，控制器收不到图 → 按平地填充 → 感知策略摔倒率 21.5%**（见
  `../../config/policy/velocity_rough/README.md` 实测）。喂恒定假平地比盲走更糟。
- 建图节点没跑起来、或 `--print` 看到的图明显不对之前，实机继续用 `Velocity`（盲走）。
- D435i 近场盲区约 0.29 m，脚正下方看不到，靠 odom 累积记忆（走过才有）；刚站定/原地时脚下格可能是 NaN，
  控制器会把这些格按平地填，是预期行为。
- 首次上机建议先在平地 + 单个矮台阶验证，确认 `--print` 的图与实际地形吻合、切换后步态正常，再上复杂地形。

## 已验证 / 待验证

已在 dev 机用合成深度端到端验证：TCP 协议、几何（光学→躯干→世界→yaw 网格）、中位数累积（台阶不被蚀平）、
DDS 发布网格与控制器 `include/height_map_gate.h` 约定逐位一致（17×11 @0.1，origin(-0.8,-0.5)，
data[iy*17+ix]，未知 NaN）。**未在真机跑过**：真实外参精度、odom 与躯干的杆臂、走动时的累积漂移、
端到端时延都要上机用 `--print` + `--calibrate-flat` 校准确认。

## 调参（height_map_mapper.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--torso-z-over-body` | 0.111 | 高度偏置，**必须标定** |
| `--memory-s` | 10 | 累积格超时未更新→视为未知(NaN)。走得快可调小 |
| `--ewma` | 0.5 | 每格世界 z 的时间平滑；大=跟得快噪声大 |
| `--rate` | 50 | 发布频率，对齐策略 50 Hz |
| `--acc-span` | 40 | odom 系累积网格边长(m)，超出范围只用当前帧 |
Jetson 端 `--stride`（默认 4→160×120）控带宽/密度，`--zmax`（默认 4 m）截远场噪声。
