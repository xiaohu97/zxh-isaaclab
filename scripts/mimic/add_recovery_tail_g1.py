#!/usr/bin/env python3
"""给 G1 mimic 参考动作（NPZ）追加"落地站稳"尾段，输出 csv_to_npz.py 能吃的 CSV。

背景：jump1_1m_waist15.npz 的末帧还在落地回弹中（根部 vz=+0.41 m/s、水平 0.52 m/s、
骨盆前倾 23.6°、腿部关节速度 3.6 rad/s），训练又用默认的 motion_end_behavior="resample"，
策略从没学过把身体停下来；部署按时长切 walk，walk 接手的就是这个回弹状态。

本工具生成的序列::

    原动作(不改一帧) -> 刹车段 -> 收姿段 -> 站立保持

* 刹车段 ``--brake-seconds``：每个关节按五次 Hermite 从末帧位置/速度减速到静止
  （位移 = v0*T/2），双脚固定在末帧位姿。保证关节速度连续、又不会因为末帧 3.6 rad/s
  的速度把关节甩过头。
* 收姿段 ``--settle-seconds``：关节按五次 smoothstep 过渡到目标站姿（默认 walk 的
  default_joint_pos，腰俯仰 0）；双脚位姿同步从末帧位姿过渡到"原地放平"（x/y/yaw 不变，
  roll/pitch 归零，z=脚底离地高度）。
* 站立保持 ``--hold-seconds``：重复最后一帧。

根部位姿不是插值出来的，而是由"双脚位姿 + 关节角"经 MuJoCo 正运动学反推
（root = T_foot · FK(foot←root)⁻¹，左右腿各算一次取平均），所以参考里的脚不会滑、
不会穿地、也不会因为末帧 +0.41 m/s 的竖直速度再蹦一下。左右腿反推的根部不一致
（目标站姿是对称的、实际落地脚不对称）会体现为参考脚位的轻微漂移，summary 里会
报出每只脚的漂移量，超过 --max-foot-drift 直接报错。

用法::

    python scripts/mimic/add_recovery_tail_g1.py \
        --input-npz  .../jump1_1mwithid/jump1_1m_waist15.npz \
        --output-csv .../jump1_1mwithid_recover/jump1_1m_waist15_recover.csv \
        --summary-json .../jump1_1mwithid_recover/jump1_1m_waist15_recover.summary.json

    然后用 Isaac Lab 重算全部刚体位姿/速度（不要自己拼 body 数组）::

    python scripts/mimic/csv_to_npz.py -f <output.csv> --input_fps 50 --output_fps 50 --headless --no_ground

CSV 每行 = root 位置(3) + root 四元数 x,y,z,w (4) + 29 关节角（**SDK/电机序**），
与 csv_to_npz.py 的 joint_sdk_names 一致。csv_to_npz 用 (N-1)*dt 当时长，会丢最后一帧，
这里多写一帧保持帧补偿。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 mujoco python 包做正运动学（例如 ustc_identification 环境）: {exc}")
from scipy.spatial.transform import Rotation as R, Slerp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_XML = "/home/ustczxh/humanoid/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
DEFAULT_IDS_YAML = os.path.join(REPO, "deploy/robots/g1_29dof/config/policy/mimic/jump3/params/deploy.yaml")
DEFAULT_POSE_YAML = os.path.join(REPO, "deploy/robots/g1_29dof/config/policy/velocity/params/deploy.yaml")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def quintic_hermite(p0, v0, p1, v1, T, t):
    """p(t) on [0,T] with p(0)=p0, p'(0)=v0, p''(0)=0, p(T)=p1, p'(T)=v1, p''(T)=0. Arrays broadcast over t."""
    s = np.asarray(t, dtype=float)[:, None] / T
    p0, v0, p1, v1 = (np.asarray(x, dtype=float)[None, :] for x in (p0, v0, p1, v1))
    h = p1 - p0
    a3 = (10 * h - (6 * v0 + 4 * v1) * T) / T**3
    a4 = (-15 * h + (8 * v0 + 7 * v1) * T) / T**4
    a5 = (6 * h - 3 * (v0 + v1) * T) / T**5
    tt = s * T
    return p0 + v0 * tt + a3 * tt**3 + a4 * tt**4 + a5 * tt**5


def smoothstep5(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


class Fk:
    def __init__(self, xml: str):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.sdk_names = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        if len(self.sdk_names) != 29 or self.m.nq != 36:
            raise RuntimeError(f"期望 29 关节浮动基 G1 模型，得到 nq={self.m.nq}")
        self.feet = [self.m.body(n).id for n in FEET]

    def foot_poses(self, root_pos, root_quat_wxyz, q_sdk):
        """世界系双脚位姿 -> [(pos, Rotation), ...]"""
        self.d.qpos[:3] = root_pos
        self.d.qpos[3:7] = root_quat_wxyz
        self.d.qpos[7:] = q_sdk
        mujoco.mj_kinematics(self.m, self.d)
        return [(self.d.xpos[b].copy(), R.from_matrix(self.d.xmat[b].reshape(3, 3).copy())) for b in self.feet]

    def root_from_feet(self, feet_world, q_sdk):
        """已知双脚世界位姿和关节角，反推根部位姿（左右腿各一解，取平均）。"""
        rel = self.foot_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q_sdk)  # foot in root frame
        sols = []
        for (pw, Rw), (pr, Rr) in zip(feet_world, rel):
            R_root = Rw * Rr.inv()
            p_root = pw - R_root.apply(pr)
            sols.append((p_root, R_root))
        p = 0.5 * (sols[0][0] + sols[1][0])
        q = R.from_quat(np.stack([sols[0][1].as_quat(), sols[1][1].as_quat()]))
        # slerp 0.5 via mean of two rotations (scipy mean = chordal, fine for近似一致的两解)
        Rm = q.mean()
        drift = [np.linalg.norm(sols[i][0] - p) for i in range(2)]
        return p, Rm, drift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-npz", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--mujoco-xml", default=DEFAULT_XML)
    ap.add_argument("--joint-ids-map-yaml", default=DEFAULT_IDS_YAML, help="含 joint_ids_map 的 deploy.yaml（Isaac 序 -> 电机序）")
    ap.add_argument("--target-pose-yaml", default=DEFAULT_POSE_YAML, help="含 default_joint_pos（Isaac 序）的 deploy.yaml，默认 walk")
    ap.add_argument("--waist-pitch-target", type=float, default=None, help="覆盖目标腰俯仰 [rad]")
    ap.add_argument("--brake-seconds", type=float, default=0.16)
    ap.add_argument("--settle-seconds", type=float, default=0.64)
    ap.add_argument("--hold-seconds", type=float, default=1.0)
    ap.add_argument("--sole-offset", type=float, default=None, help="脚放平时 ankle_roll 原点离地高度；默认取首帧两脚 z 的最小值")
    ap.add_argument("--max-foot-drift", type=float, default=0.06, help="参考里单脚允许的位置漂移 [m]")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.output_csv) and not a.overwrite:
        sys.exit(f"{a.output_csv} 已存在，加 --overwrite")

    d = np.load(a.input_npz)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    jp_isaac, jv_isaac = d["joint_pos"], d["joint_vel"]
    root_pos, root_quat_wxyz = d["body_pos_w"][:, 0], d["body_quat_w"][:, 0]
    n_prefix = jp_isaac.shape[0]

    ids = yaml.safe_load(open(a.joint_ids_map_yaml))["joint_ids_map"]
    motor_order = np.argsort(ids)  # 电机 m -> Isaac 序
    fk = Fk(a.mujoco_xml)

    target_isaac = np.asarray(yaml.safe_load(open(a.target_pose_yaml))["default_joint_pos"], dtype=float)
    target_sdk = target_isaac[motor_order]
    if a.waist_pitch_target is not None:
        target_sdk[fk.sdk_names.index("waist_pitch_joint")] = a.waist_pitch_target

    q_sdk = jp_isaac[:, motor_order]
    v_sdk = jv_isaac[:, motor_order]

    # ---- 末帧与首帧的脚 ----
    feet_last = fk.foot_poses(root_pos[-1], root_quat_wxyz[-1], q_sdk[-1])
    feet_first = fk.foot_poses(root_pos[0], root_quat_wxyz[0], q_sdk[0])
    sole = a.sole_offset if a.sole_offset is not None else float(min(p[2] for p, _ in feet_first))

    # ---- 刹车段：关节减速到静止，脚不动 ----
    nb = int(round(a.brake_seconds * fps))
    ns = int(round(a.settle_seconds * fps))
    nh = int(round(a.hold_seconds * fps)) + 1  # +1 补偿 csv_to_npz 丢末帧
    tb = np.arange(1, nb + 1) * dt
    q_brake = quintic_hermite(q_sdk[-1], v_sdk[-1], q_sdk[-1] + v_sdk[-1] * a.brake_seconds / 2, np.zeros(29), a.brake_seconds, tb)
    q_stop = q_brake[-1]

    # ---- 收姿段：关节 smoothstep 到目标，脚 smoothstep 到放平 ----
    ts = np.arange(1, ns + 1) * dt
    w = smoothstep5(ts / a.settle_seconds)
    q_settle = q_stop[None, :] + w[:, None] * (target_sdk - q_stop)[None, :]
    feet_flat = []
    for p, Rw in feet_last:
        yaw = Rw.as_euler("zyx")[0]
        feet_flat.append((np.array([p[0], p[1], sole]), R.from_euler("z", yaw)))

    # 末帧 NPZ 根部与"双脚反推根部"之间的偏差（双脚不完全一致时非零），在刹车+收姿段内平滑淡出，
    # 保证接缝处根部位姿连续，尾段结束时完全由脚反推。
    p_last_fk, R_last_fk, _ = fk.root_from_feet(feet_last, q_sdk[-1])
    off_pos = root_pos[-1] - p_last_fk
    off_rot = R.from_quat(root_quat_wxyz[-1, [1, 2, 3, 0]]) * R_last_fk.inv()
    n_fade = nb + ns

    rows, drifts, root_track = [], [], []
    def push(q, feet):
        k = len(rows) + 1
        fade = 1.0 - smoothstep5(k / n_fade) if n_fade > 0 else 0.0
        p, Rm, drift = fk.root_from_feet(feet, q)
        p = p + fade * off_pos
        Rm = R.from_rotvec(fade * off_rot.as_rotvec()) * Rm
        rows.append(np.concatenate([p, Rm.as_quat(), q]))  # scipy quat = x,y,z,w
        drifts.append(drift)
        root_track.append(p)

    for q in q_brake:
        push(q, feet_last)
    for k, q in enumerate(q_settle):
        feet = []
        for (p0, R0), (p1, R1) in zip(feet_last, feet_flat):
            key = R.from_quat(np.stack([R0.as_quat(), R1.as_quat()]))
            Rk = Slerp([0, 1], key)([w[k]])[0]
            feet.append((p0 + w[k] * (p1 - p0), Rk))
        push(q, feet)
    for _ in range(nh):
        push(target_sdk, feet_flat)

    tail = np.stack(rows)
    prefix = np.concatenate([root_pos, root_quat_wxyz[:, [1, 2, 3, 0]], q_sdk], axis=1)
    out = np.concatenate([prefix, tail], axis=0)

    # ---- 一致性检查 ----
    drifts = np.asarray(drifts)
    # 根部平滑度：接缝处根部位置/姿态跳变
    # 接缝处根部一帧的位移/转角：脚固定、关节按末帧速度继续动，根部自然跟着动；换算成速度后应与
    # NPZ 末帧根部速度同量级（是速度延续，不是跳变）。
    seam_pos = float(np.linalg.norm(tail[0, :3] - root_pos[-1]))
    seam_rot = float((R.from_quat(tail[0, 3:7]) * R.from_quat(root_quat_wxyz[-1, [1, 2, 3, 0]]).inv()).magnitude())
    tail_dq = np.abs(np.diff(np.concatenate([q_sdk[-1:], tail[:, 7:]]), axis=0)) * fps
    feet_z = []
    for r in tail:
        feet_z.append([p[2] for p, _ in fk.foot_poses(r[:3], r[[6, 3, 4, 5]], r[7:])])
    feet_z = np.asarray(feet_z)
    final_root = tail[-1]
    tilt = float(np.degrees(np.arccos(np.clip(R.from_quat(final_root[3:7]).as_matrix()[2, 2], -1, 1))))
    summary = {
        "input_npz": os.path.abspath(a.input_npz),
        "output_csv": os.path.abspath(a.output_csv),
        "fps": fps,
        "prefix_frames": int(n_prefix),
        "brake_frames": nb,
        "settle_frames": ns,
        "hold_frames_written": nh,
        "csv_rows": int(out.shape[0]),
        "expected_npz_frames_after_csv_to_npz": int(out.shape[0] - 1),
        "tail_start_frame": int(n_prefix),
        "hold_start_frame": int(n_prefix + nb + ns),
        "sole_offset_m": sole,
        "target_pose_sdk": target_sdk.tolist(),
        "seam_root_step_m": seam_pos,
        "seam_root_speed_m_s": seam_pos * fps,
        "seam_root_rot_step_deg": float(np.degrees(seam_rot)),
        "seam_root_rot_speed_rad_s": seam_rot * fps,
        "last_prefix_root_speed_m_s": float(np.linalg.norm(d["body_lin_vel_w"][-1, 0])),
        "last_prefix_root_ang_speed_rad_s": float(np.linalg.norm(d["body_ang_vel_w"][-1, 0])),
        "npz_vs_feet_fk_root_offset_m": float(np.linalg.norm(off_pos)),
        "npz_vs_feet_fk_root_offset_deg": float(np.degrees(off_rot.magnitude())),
        "max_tail_joint_speed_rad_s": float(tail_dq.max()),
        "max_ref_foot_drift_m": float(drifts.max()),
        "final_root_pos": final_root[:3].tolist(),
        "final_root_tilt_deg": tilt,
        "min_foot_z_in_tail_m": float(feet_z.min()),
        "last_prefix_root_vel_world": d["body_lin_vel_w"][-1, 0].tolist(),
    }
    if drifts.max() > a.max_foot_drift:
        sys.exit(f"参考脚位漂移 {drifts.max():.3f} m 超过 {a.max_foot_drift} m，目标站姿与落地脚位差太远")

    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    np.savetxt(a.output_csv, out, delimiter=",", fmt="%.9f")
    if a.summary_json:
        with open(a.summary_json, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
