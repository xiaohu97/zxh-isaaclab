"""把 jump 参考的"蹬伸 + 腾空"段重写成物理自洽的弹道，用于提升跳远距离。

背景（2026-09-16 实测）
----------------------
原参考 ``jump1_1m_waist15_recover.npz`` 的腾空段不可实现：

* 起跳竖直速度 0.448 m/s，弹道滞空只有 0.09 s，参考却腾空 0.58 s
* 腾空段水平速度 1.06→2.67 m/s 变化（弹道应恒定），竖直加速度 -3.23 m/s²（应 -9.81）
* 蹬伸只有 0.08 s，膝角 87.7°→85.6°，**根本没有蹬地动作**；根部竖直速度峰值出现在离地后第 8 帧

策略因此学不到蹬地：干净测量（1024 次，第 0 帧起跳、无随机化）实测距离 0.973 m，
比参考的 1.401 m 少 31%，且髋俯仰在蹬伸窗口 26% 时间顶满 88 N·m（膝只用到 74/139，
速度 13.4/20）——短板在髋不在膝。

本脚本做什么
------------
1. 保留 0..CROUCH_END 帧的蓄力段不动。
2. 合成真实蹬伸段：以"双脚钉在地面"为约束，用雅可比解出达到目标起跳速度所需的关节角速度，
   再用五次 Hermite 从蓄力姿态插到起跳姿态（末端速度非零 = 离地瞬间仍在加速）。
   根部位姿由 FK 从planted feet 反推，因此蹬伸段天然满足"脚不滑"。
3. 腾空段按真弹道积分根部位置，姿态从起跳姿态 slerp 回原参考的落地姿态；
   关节角沿用原腾空段的收/伸腿节奏，起点用衰减修正对齐到新的起跳姿态。
   落地时刻由 FK 判定（脚底触地），不是预设帧数。
4. 落地+恢复段沿用原参考 LAND_FRAME.. 末帧，整体平移到新的落地点。

输出 CSV（root_pos 3 + root_quat xyzw 4 + 29 关节电机序），再用 ``csv_to_npz.py``
在 Isaac 里重算全部刚体位姿/速度。FK 用 mujoco 的 g1_29dof.xml：其**腿链与训练 URDF 逐位一致**
（仅腰/肩差 9~19 mm，不参与 root_from_feet），已验证反推根部位姿误差 0.00 mm。

用法（需 mujoco + scipy + yaml，本机在 wham_gmr 环境）::

    python scripts/mimic/retime_jump_ballistic_g1.py \
        --input-npz  .../jump1_1mwithid_recover/jump1_1m_waist15_recover.npz \
        --output-csv .../jump1_1mwithid_far/jump1_1m_ballistic.csv \
        --target-distance 1.15 --takeoff-angle-deg 40 --summary-json .../summary.json
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
    sys.exit(f"需要 mujoco 做正运动学（本机在 wham_gmr 环境）: {exc}")
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_XML = "/home/zxh/ustc_humanoid/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
DEFAULT_IDS_YAML = os.path.join(REPO, "deploy/robots/g1_29dof/config/policy/mimic/jump3/params/deploy.yaml")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
G = 9.81
ISAAC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def quintic_hermite(p0, v0, p1, v1, T, t):
    """p(0)=p0, p'(0)=v0, p''(0)=0, p(T)=p1, p'(T)=v1, p''(T)=0；t 为 [0,T] 上的采样点。"""
    s = np.asarray(t, dtype=float)[:, None] / T
    p0, v0, p1, v1 = (np.asarray(x, dtype=float)[None, :] for x in (p0, v0, p1, v1))
    h = p1 - p0
    a3 = 10 * h - (6 * v0 + 4 * v1) * T
    a4 = -15 * h + (8 * v0 + 7 * v1) * T
    a5 = 6 * h - (3 * v0 + 3 * v1) * T
    return p0 + (v0 * T) * s + a3 * s**3 + a4 * s**4 + a5 * s**5


def cubic_hermite(p0, v0, p1, v1, T, t):
    """p(0)=p0, p'(0)=v0, p(T)=p1, p'(T)=v1（末端加速度自由）。

    蹬地时地面反力持续把身体往上加速，速度应在离地瞬间达到峰值。五次 Hermite 强制
    p''(T)=0，会让速度在中途超调、末端反而在减速，不能用于蹬伸段。配合 T=2h/v1
    （匀加速）时本式给出的正是恒定加速度剖面。
    """
    s = np.asarray(t, dtype=float)[:, None] / T
    p0, v0, p1, v1 = (np.asarray(x, dtype=float)[None, :] for x in (p0, v0, p1, v1))
    h = p1 - p0
    a2 = 3 * h - (2 * v0 + v1) * T
    a3 = -2 * h + (v0 + v1) * T
    return p0 + (v0 * T) * s + a2 * s**2 + a3 * s**3


class Fk:
    """mujoco 正运动学；只用腿链，与训练 URDF 逐位一致。"""

    def __init__(self, xml: str):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.mj_names = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        if len(self.mj_names) != 29:
            raise RuntimeError(f"期望 29 关节浮动基 G1，得到 {len(self.mj_names)}")
        self.feet = [self.m.body(n).id for n in FEET]
        self.iso2mj = np.array([self.mj_names.index(n) for n in ISAAC_JOINTS])

    def _kin(self, root_pos, root_quat_wxyz, q_mj):
        self.d.qpos[:3] = root_pos
        self.d.qpos[3:7] = root_quat_wxyz
        self.d.qpos[7:] = q_mj
        mujoco.mj_kinematics(self.m, self.d)

    def foot_poses(self, root_pos, root_quat_wxyz, q_mj):
        self._kin(root_pos, root_quat_wxyz, q_mj)
        return [(self.d.xpos[b].copy(), R.from_matrix(self.d.xmat[b].reshape(3, 3).copy())) for b in self.feet]

    def root_from_feet(self, feet_world, q_mj):
        """已知双脚世界位姿 + 关节角 -> 根部位姿（左右腿各一解取平均）。"""
        rel = self.foot_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q_mj)
        sols = []
        for (pw, Rw), (pr, Rr) in zip(feet_world, rel):
            R_root = Rw * Rr.inv()
            sols.append((pw - R_root.apply(pr), R_root))
        p = 0.5 * (sols[0][0] + sols[1][0])
        Rm = R.from_quat(np.stack([s[1].as_quat() for s in sols])).mean()
        return p, Rm, float(np.linalg.norm(sols[0][0] - sols[1][0]))


def solve_takeoff_pose(fk, q_ref_mj, feet_world, knee_deg, ankle_extra_deg, pitch_deg, push_dir, p_crouch, travel_m):
    """求起跳姿态。

    左右腿分别解 hip_pitch/knee/ankle_pitch（6 自由度）。残差里有两条关键约束：

    * **双腿一致性**：双脚同时钉地是过约束的，若两腿给同一组角度，蓄力姿态本身的左右不对称会让
      两腿解出不同根部位姿（实测差 233 mm），等价于脚在打滑。
    * **蹬伸方向**：蹬伸段按直线匀加速建模，起跳速度方向必然与 (p_to - p_crouch) 平行。
      想要 40° 起跳角，就得让起跳姿态把骨盆送到那个方向上——方向由姿态决定，不能靠事后校准。
    * **蹬伸行程** |p_to - p_crouch| = travel_m：只给方向约束时，解会停在离蓄力姿态 4 cm 的地方
      （膝几乎不伸），T=2h/v 于是压到 0.08 s、加速度 112 m/s²、关节角速度 50 rad/s，不可实现。
      行程直接决定加速度 a=v²/(2h)，必须显式约束。
    """
    nm = fk.mj_names
    hp = [nm.index("left_hip_pitch_joint"), nm.index("right_hip_pitch_joint")]
    kn = [nm.index("left_knee_joint"), nm.index("right_knee_joint")]
    ak = [nm.index("left_ankle_pitch_joint"), nm.index("right_ankle_pitch_joint")]
    idx = hp + kn + ak
    x0 = np.concatenate([q_ref_mj[hp], np.radians([knee_deg, knee_deg]), q_ref_mj[ak] - np.radians(ankle_extra_deg)])
    u = np.asarray(push_dir, dtype=float)
    u = u / np.linalg.norm(u)

    def resid(x):
        q = q_ref_mj.copy()
        q[idx] = x
        rel = fk.foot_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
        sols = []
        for (pw, Rw), (pr, Rr) in zip(feet_world, rel):
            R_root = Rw * Rr.inv()
            sols.append((pw - R_root.apply(pr), R_root))
        drift = sols[0][0] - sols[1][0]
        p = 0.5 * (sols[0][0] + sols[1][0])
        Rm = R.from_quat(np.stack([s_[1].as_quat() for s_ in sols])).mean()
        d = p - p_crouch
        n = np.linalg.norm(d)
        dir_err = (d / n - u) if n > 1e-6 else np.zeros(3)
        return np.concatenate([
            drift * 30.0,
            [(pitch_of(Rm) - (-abs(pitch_deg))) * 0.20],
            dir_err * 8.0,
            [(n - travel_m) * 25.0],
            (x[2:4] - np.radians(knee_deg)) * 0.05,
        ])

    sol = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, max_nfev=6000)
    q = q_ref_mj.copy()
    q[idx] = sol.x
    p, Rm, drift = fk.root_from_feet(feet_world, q)
    d = p - p_crouch
    ang = float(np.degrees(np.arcsin(np.clip(d[2] / max(np.linalg.norm(d), 1e-9), -1, 1))))
    return q, abs(pitch_of(Rm) - (-abs(pitch_deg))), drift, ang


def root_jacobian(fk, feet_world, q_mj, leg_idx, eps=1e-5):
    """d(root_pos)/d(q_leg)，双脚钉地约束下的数值雅可比 [3 x len(leg_idx)]。"""
    J = np.zeros((3, len(leg_idx)))
    for i, j in enumerate(leg_idx):
        qp, qm = q_mj.copy(), q_mj.copy()
        qp[j] += eps
        qm[j] -= eps
        pp, _, _ = fk.root_from_feet(feet_world, qp)
        pm, _, _ = fk.root_from_feet(feet_world, qm)
        J[:, i] = (pp - pm) / (2 * eps)
    return J


def pitch_of(Rm):
    g = Rm.inv().apply([0, 0, -1.0])
    return np.degrees(np.arctan2(-g[0], -g[2]))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--input-npz", required=True)
    ap_.add_argument("--output-csv", required=True)
    ap_.add_argument("--summary-json", default=None)
    ap_.add_argument("--mujoco-xml", default=DEFAULT_XML)
    ap_.add_argument("--joint-ids-map-yaml", default=DEFAULT_IDS_YAML)
    ap_.add_argument("--fps", type=float, default=50.0)
    ap_.add_argument("--crouch-end-frame", type=int, default=33, help="蓄力段末帧（含），之后重写")
    ap_.add_argument("--flight-start-frame", type=int, default=37, help="原参考的腾空起始帧")
    ap_.add_argument("--land-frame", type=int, default=66, help="原参考的落地帧；该帧起沿用原轨迹")
    ap_.add_argument("--target-distance", type=float, default=1.30,
                     help="双脚中点水平距离 [m]，口径 = 脚离地超过 --distance-clearance 的第一帧到最后一帧，"
                          "与评估脚本一致（脚本内部的 蹬伸末帧->触地帧 口径会高估约 20%）")
    ap_.add_argument("--distance-clearance", type=float, default=0.05, help="判定腾空的脚底离地阈值 [m]")
    ap_.add_argument("--takeoff-angle-deg", type=float, default=40.0)
    ap_.add_argument("--takeoff-pitch-deg", type=float, default=35.0, help="起跳瞬间骨盆前倾角（正数=前倾）")
    ap_.add_argument("--knee-takeoff-deg", type=float, default=25.0, help="起跳时膝角（越小越直）")
    ap_.add_argument("--ankle-extra-deg", type=float, default=12.0, help="起跳时额外跖屈")
    ap_.add_argument("--crouch-extra-m", type=float, default=0.0,
                     help="在蓄力段末尾再下蹲这么多米（竖直）。蹬伸行程 h 受腿可达限制, 原蓄力底部只给到 "
                          "h=0.19m, 而 a=v²/(2h) 把速度卡死; 蹲深才能在同样加速度下起跳更快")
    ap_.add_argument("--crouch-extra-frames", type=int, default=6, help="加深蓄力用的帧数")
    ap_.add_argument("--push-travel-m", type=float, default=0.28, help="蹬伸段根部行程 [m]；决定加速度 a=v²/(2h)")
    ap_.add_argument("--tail-blend-frames", type=int, default=10,
                     help="尾段用多少帧把弹道触地高度与站立高度的落差平滑吸收")
    ap_.add_argument("--max-foot-drift", type=float, default=0.06, help="起跳姿态双腿反推根部允许差 [m]")
    ap_.add_argument("--overwrite", action="store_true")
    a = ap_.parse_args()
    if os.path.exists(a.output_csv) and not a.overwrite:
        sys.exit(f"{a.output_csv} 已存在，加 --overwrite")

    fk = Fk(a.mujoco_xml)
    ids = yaml.safe_load(open(a.joint_ids_map_yaml))["joint_ids_map"]
    if not np.array_equal(np.array(ids), fk.iso2mj):
        sys.exit("deploy.yaml 的 joint_ids_map 与 mujoco 关节序不一致")

    npz = np.load(a.input_npz)
    P, Q, J = npz["body_pos_w"], npz["body_quat_w"], npz["joint_pos"]
    fps, dt = a.fps, 1.0 / a.fps
    C, F0, LD = a.crouch_end_frame, a.flight_start_frame, a.land_frame
    LA, RA = 18, 19  # body 序里的 ankle_roll
    q_mj = np.zeros((len(J), 29))
    q_mj[:, fk.iso2mj] = J

    feet_C = [(P[C, LA].astype(float), R.from_quat(Q[C, LA][[1, 2, 3, 0]])),
              (P[C, RA].astype(float), R.from_quat(Q[C, RA][[1, 2, 3, 0]]))]
    sole = float(min(P[:10, LA, 2].min(), P[:10, RA, 2].min()))

    # ---- 起跳姿态 + 目标速度 ----
    th = np.radians(a.takeoff_angle_deg)
    fmid0 = lambda t: (P[t, LA, :2] + P[t, RA, :2]) / 2
    d_hat = fmid0(LD) - fmid0(F0)
    d_hat = d_hat / np.linalg.norm(d_hat)
    push_dir = np.array([*(np.cos(th) * d_hat), np.sin(th)])
    p_C, R_C, _ = fk.root_from_feet(feet_C, q_mj[C])
    fmid = fmid0

    nm = fk.mj_names
    IK_JOINTS = [nm.index(n) for n in (
        "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint")]

    def ik_planted(p_want, pitch_want, q_guess):
        """解关节角, 使双脚钉地下根部落在 p_want、骨盆俯仰为 pitch_want。

        蹬伸段直接在根部空间定轨迹(直线匀加速), 再逐帧 IK —— 这样起跳速度是精确给定的。
        先前在关节空间插值再反推根部的做法, 中间帧不满足双脚钉地(两腿解差 221 mm),
        而事后用雅可比校正关节角速度会和投影互相打架, 实测直接发散到 233 rad/s。
        """
        def r(x):
            q = q_guess.copy()
            q[IK_JOINTS] = x
            rel = fk.foot_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
            sols = []
            for (pw, Rw), (pr, Rr) in zip(feet_C, rel):
                R_root = Rw * Rr.inv()
                sols.append((pw - R_root.apply(pr), R_root))
            p = 0.5 * (sols[0][0] + sols[1][0])
            Rm = R.from_quat(np.stack([s_[1].as_quat() for s_ in sols])).mean()
            return np.concatenate([
                (sols[0] [0]- sols[1][0]) * 50.0,
                (p - p_want) * 50.0,
                [(pitch_of(Rm) - pitch_want) * 0.15],
                (x - q_guess[IK_JOINTS]) * 0.02,
            ])
        sol = least_squares(r, q_guess[IK_JOINTS], xtol=1e-13, ftol=1e-13, max_nfev=1200)
        q = q_guess.copy()
        q[IK_JOINTS] = sol.x
        return q

    def build_crouch_extra():
        """在蓄力末尾继续下蹲 crouch_extra_m，扩大蹬伸行程。"""
        if a.crouch_extra_m <= 1e-6:
            return np.zeros((0, 36)), q_mj[C], p_C, R_C
        N = max(a.crouch_extra_frames, 2)
        pit = pitch_of(R_C)
        rows, qg = [], q_mj[C].copy()
        for k in range(N):
            f = (k + 1) / N
            p_want = p_C - np.array([0.0, 0.0, a.crouch_extra_m * f])
            qk = ik_planted(p_want, pit, qg)
            qg = qk
            p, Rm, _ = fk.root_from_feet(feet_C, qk)
            rows.append(np.concatenate([p, Rm.as_quat(), qk]))
        rows = np.stack(rows)
        return rows, rows[-1, 7:].copy(), rows[-1, :3].copy(), R.from_quat(rows[-1, 3:7])

    rows_crouch, q_base, p_base, R_base = build_crouch_extra()
    q_to, pitch_err, drift_to, push_ang = solve_takeoff_pose(
        fk, q_base, feet_C, a.knee_takeoff_deg, a.ankle_extra_deg, a.takeoff_pitch_deg, push_dir, p_base, a.push_travel_m)
    # 注意: drift_to 只是姿态求解器给的目标姿态的腿间差, 它仅用于定 p_to/R_to;
    # 真正输出的蹬伸段每帧都由 ik_planted 重解, 脚漂看 summary 的 takeoff.foot_drift_mm。
    if drift_to > 0.15:
        print(f"[warn] 起跳目标姿态腿间差 {drift_to*1000:.0f} mm 偏大, 若轨迹脚漂也大请放宽 --push-travel-m", file=sys.stderr)
    p_to, R_to, _ = fk.root_from_feet(feet_C, q_to)
    h_push = float(np.linalg.norm(p_to - p_base))
    v_mag = np.sqrt(max(a.target_distance, 1e-6) * G / np.sin(2 * np.radians(max(push_ang, 5.0))))

    def build_push(v_mag_want):
        """直线匀加速蹬伸：|v| 给定, 方向由起跳姿态保证; T = 2h/v 即恒定加速度。"""
        u = (p_to - p_base) / h_push
        v_end = u * v_mag_want
        T = 2 * h_push / v_mag_want
        N = max(int(round(T * fps)), 4)
        tq = np.arange(1, N + 1) * dt
        # 恒定加速度 a = v^2/(2h)：s(t) = ½ a t²
        acc = v_mag_want**2 / (2 * h_push)
        s_t = 0.5 * acc * tq**2
        pitch_C, pitch_T = pitch_of(R_base), pitch_of(R_to)
        rows, drs, qg = [], [], q_base.copy()
        for k in range(N):
            frac = float(np.clip(s_t[k] / h_push, 0.0, 1.0))
            p_want = p_base + u * min(s_t[k], h_push)
            pw = pitch_C + (pitch_T - pitch_C) * frac
            qk = ik_planted(p_want, pw, qg)
            qg = qk
            p, Rm, dr = fk.root_from_feet(feet_C, qk)
            rows.append(np.concatenate([p, Rm.as_quat(), qk]))
            drs.append(dr)
        rows = np.stack(rows)
        dq = np.abs(np.diff(np.concatenate([q_base[None, :], rows[:, 7:]]), axis=0)) * fps
        return rows, v_end, float(dq.max()), float(max(drs)), N

    # ---- 腾空段：真弹道 + 关节角从原参考时间伸缩、起点衰减修正 ----
    q_fl_src = q_mj[F0:LD + 1]
    R_fl_src = R.from_quat(Q[F0:LD + 1, 0][:, [1, 2, 3, 0]])

    def build_flight(n_fl, q_take, R_take, p_take, v_take):
        s_ = np.linspace(0.0, 1.0, n_fl + 1)[1:]
        src = np.linspace(0.0, 1.0, len(q_fl_src))
        qf = np.stack([np.interp(s_, src, q_fl_src[:, j]) for j in range(29)], axis=1)
        q0 = np.array([np.interp(0.0, src, q_fl_src[:, j]) for j in range(29)])
        qf = qf + (q_take - q0)[None, :] * ((1 - s_) ** 2)[:, None]
        Rf = Slerp(src, R_fl_src)(s_)
        R0 = Slerp(src, R_fl_src)([0.0])[0]
        dR = R_take * R0.inv()
        rot = R.from_rotvec(dR.as_rotvec()[None, :] * ((1 - s_) ** 2)[:, None])
        Rf = R.from_quat((rot * Rf).as_quat())
        tt = np.arange(1, n_fl + 1) * dt
        pf = p_take[None, :] + v_take[None, :] * tt[:, None]
        pf[:, 2] = p_take[2] + v_take[2] * tt - 0.5 * G * tt**2
        return qf, Rf, pf

    def fly_until_touchdown(q_take, R_take, p_take, v_take):
        """按弹道积分, 用 FK 判定脚底触地; 返回落地那一刻为止的腾空段。"""
        n_fl = max(int(round(2 * max(v_take[2], 0.1) / G * fps)), 4)
        for _ in range(12):
            qf, Rf, pf = build_flight(n_fl, q_take, R_take, p_take, v_take)
            touch = None
            for k in range(n_fl):
                fz = [fk.foot_poses(pf[k], Rf[k].as_quat()[[3, 0, 1, 2]], qf[k])[i][0][2] for i in (0, 1)]
                if k >= 2 and min(fz) <= sole:
                    touch = k
                    break
            if touch is None:
                n_fl += 3
                continue
            if touch < n_fl - 1:
                n_fl = touch + 1
                qf, Rf, pf = build_flight(n_fl, q_take, R_take, p_take, v_take)
            return qf, Rf, pf
        raise RuntimeError("腾空段未能判定落地")

    def feet_mid_of(row):
        fp = fk.foot_poses(row[:3], row[3:7][[3, 0, 1, 2]], row[7:])
        return (fp[0][0][:2] + fp[1][0][:2]) / 2

    # 外层: 按实测落地距离调整起跳速度大小, 命中 target-distance
    v_mag_cur = v_mag
    for it in range(12):
        rows_push, v_to_actual, max_qdot, drift_push, N_push = build_push(v_mag_cur)
        # 起点必须用蹬伸最后一帧 IK 解出的关节角, 不能用姿态求解器的 q_to ——
        # 后者自带腿漂、与 IK 轨迹终点不是同一姿态, 实测在边界造成 66.9 rad/s 的假尖峰
        qf, Rf, pf = fly_until_touchdown(rows_push[-1, 7:], R.from_quat(rows_push[-1, 3:7]), rows_push[-1, :3], v_to_actual)
        rows_fl = np.concatenate([pf, Rf.as_quat(), qf], axis=1)
        # 与评估同口径：脚底离地超过 distance_clearance 的第一帧到最后一帧
        seq = np.concatenate([rows_push, rows_fl], axis=0)
        fz_seq = np.array([min(fk.foot_poses(r[:3], r[3:7][[3, 0, 1, 2]], r[7:])[i][0][2] for i in (0, 1)) for r in seq])
        air_idx = np.where(fz_seq > sole + a.distance_clearance)[0]
        if len(air_idx) < 2:
            dist = float(np.linalg.norm(feet_mid_of(rows_fl[-1]) - feet_mid_of(rows_push[-1])))
        else:
            dist = float(np.linalg.norm(feet_mid_of(seq[air_idx[-1]]) - feet_mid_of(seq[air_idx[0]])))
        if abs(dist - a.target_distance) < 0.01:
            break
        v_mag_cur *= float(np.sqrt(max(a.target_distance, 1e-3) / max(dist, 1e-3)))

    # ---- 落地+恢复段：原轨迹整体平移到新落地点 ----
    tail = np.concatenate([P[LD:, 0], Q[LD:, 0][:, [1, 2, 3, 0]], q_mj[LD:]], axis=1).copy()
    # 只做水平平移：尾段是"脚踩在地面上"的站立/恢复动作，竖直方向必须保持对地高度。
    # 早先把 3D 位移整体加上去，弹道触地高度与原落地帧根部高度的差被带进尾段，
    # 实测整段悬空 1.64 cm（脚底 0.0497 而非站立的 0.0333）。
    tail[:, :2] += (rows_fl[-1, :2] - tail[0, :2])[None, :]
    dz = float(rows_fl[-1, 2] - tail[0, 2])
    K = min(a.tail_blend_frames, len(tail))
    ramp = np.concatenate([dz * (1.0 - np.arange(K) / K), np.zeros(len(tail) - K)])
    tail[:, 2] += ramp

    out = np.concatenate([
        np.concatenate([P[:C + 1, 0], Q[:C + 1, 0][:, [1, 2, 3, 0]], q_mj[:C + 1]], axis=1),
        rows_crouch, rows_push, rows_fl, tail[1:],
    ], axis=0)

    # ---- 一致性检查 ----
    zf = rows_fl[:, 2]
    tt = np.arange(1, len(zf) + 1) * dt
    az = np.polyfit(tt, zf, 2)[0] * 2
    vh = np.linalg.norm(np.diff(rows_fl[:, :2], axis=0) * fps, axis=1)
    dq = np.abs(np.diff(out[:, 7:], axis=0)) * fps

    summary = dict(
        input_npz=a.input_npz, output_csv=a.output_csv, fps=fps,
        frames=dict(total=int(out.shape[0]), crouch=int(C + 1), crouch_extra=int(len(rows_crouch)), push=int(N_push), flight=int(len(rows_fl)), tail=int(len(tail) - 1)),
        takeoff=dict(pose_pitch_deg=float(pitch_of(R_to)), pitch_target_err_deg=pitch_err,
                     knee_deg=[float(np.degrees(q_to[fk.mj_names.index(n)])) for n in ("left_knee_joint", "right_knee_joint")],
                     push_angle_deg=push_ang, v_actual=[float(x) for x in v_to_actual], v_mag=float(np.linalg.norm(v_to_actual)),
                     push_seconds=float(N_push * dt), push_travel_m=h_push, push_travel_target=a.push_travel_m,
                     accel_m_s2=float(np.linalg.norm(v_to_actual)**2 / (2 * h_push)),
                     max_leg_qdot_rad_s=max_qdot, foot_drift_mm=float(drift_push * 1000),
                     pose_leg_drift_mm=float(drift_to * 1000), outer_iters=int(it + 1)),
        flight=dict(seconds=float(len(rows_fl) * dt), vertical_accel_fit=float(az),
                    horiz_speed_min_max=[float(vh.min()), float(vh.max())],
                    apex_root_z=float(zf.max()), touchdown_root_z=float(zf[-1])),
        tail=dict(vertical_gap_m=float(rows_fl[-1, 2] - np.concatenate([P[LD:, 0]], axis=0)[0, 2]),
                  blend_frames=int(min(a.tail_blend_frames, len(P) - LD))),
        distance=dict(target_m=a.target_distance, achieved_m=dist,
                      original_m=float(np.linalg.norm(fmid(LD) - fmid(F0)))),
        joint_speed=dict(max_rad_s=float(dq.max()), max_joint=fk.mj_names[int(np.unravel_index(dq.argmax(), dq.shape)[1])]),
    )
    if drift_push > a.max_foot_drift:
        sys.exit(f"蹬伸段轨迹脚漂 {drift_push*1000:.0f} mm > {a.max_foot_drift*1000:.0f} mm（IK 未能保持双脚钉地）；"
                 f"减小 --push-travel-m 或 --crouch-extra-m")
    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    np.savetxt(a.output_csv, out, delimiter=",", fmt="%.9f")
    if a.summary_json:
        json.dump(summary, open(a.summary_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
