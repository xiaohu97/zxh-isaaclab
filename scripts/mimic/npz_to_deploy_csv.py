"""把 mimic 训练用的 motion .npz 转成部署侧 State_Mimic 读的 .csv。

csv 每行 = pelvis 位置(3) + pelvis 四元数 (x,y,z,w)(4) + 29 个关节角(**电机序**)，按 --fps 重采样
（关节/位置三次样条，四元数 slerp）。关节序用 deploy.yaml 的 joint_ids_map 从训练序映射到电机序。

注意：线上 jump3.csv 不是由 jump1_1m.npz 生成的——用原 npz 重生成后 root 位姿一致(<6e-3)，但落地冲击
那几行关节角最大差 0.085 rad，且任何标准插值都复现不了，应是来自更高帧率的源数据。策略训练看到的
参考是 npz，所以部署侧用本脚本从同一个 npz 生成的 csv 反而与训练更一致。

用法:
  python scripts/mimic/npz_to_deploy_csv.py <motion.npz> <deploy.yaml> <out.csv> --fps 120
"""
import argparse
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

p = argparse.ArgumentParser()
p.add_argument("npz"); p.add_argument("deploy_yaml"); p.add_argument("out")
p.add_argument("--fps", type=float, default=120.0)
a = p.parse_args()

d = np.load(a.npz); src_fps = float(np.asarray(d["fps"]).reshape(-1)[0])
jp = d["joint_pos"]; pos = d["body_pos_w"][:, 0]; quat_wxyz = d["body_quat_w"][:, 0]
T = jp.shape[0]; t_src = np.arange(T) / src_fps
ids = yaml.safe_load(open(a.deploy_yaml))["joint_ids_map"]          # 训练序 i -> 电机序 ids[i]
motor_order = np.argsort(ids)                                          # 电机序 m -> 训练序
n_out = int(np.floor(t_src[-1] * a.fps)) + 1
t_out = np.arange(n_out) / a.fps
jp_o = CubicSpline(t_src, jp, axis=0)(t_out)[:, motor_order]
pos_o = CubicSpline(t_src, pos, axis=0)(t_out)
rots = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])                         # scipy 用 (x,y,z,w)
quat_o = Slerp(t_src, rots)(t_out).as_quat()                           # (x,y,z,w)
rows = np.concatenate([pos_o, quat_o, jp_o], axis=1)
np.savetxt(a.out, rows, delimiter=",", fmt="%.9f")
print(f"{a.out}: {rows.shape[0]} 行 x {rows.shape[1]} 列, {src_fps:.0f}->{a.fps:.0f} fps, 时长 {t_out[-1]:.3f}s")
