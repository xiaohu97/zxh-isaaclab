#!/usr/bin/env python3
"""Smooth a retargeting-noise window of a 27-DoF Humanoid Ultra reference NPZ.

The houtaitui reference ``ustc1_rightstand_stand_transition.npz`` carries
retarget jitter at the top of the kick: between 9.6 s and 10.4 s the left foot
drops 12 cm and rises 6.5 cm again, which reads on the robot as lifting the leg
twice.  The same window holds 72% of the training failures.

This tool replaces that window with a least-squares polynomial of the root pose
and the 27 joint angles, cross-faded into the original with a smoothstep so the
edges stay C1.  Everything outside the window is preserved bit-for-bit:

* ``joint_vel`` is the central difference of ``joint_pos`` in the source file
  (verified exact), so it is regenerated the same way;
* the root's ``body_lin_vel_w[:, 0]`` / ``body_ang_vel_w[:, 0]`` are the central
  and SO(3)-central derivatives of the root pose (verified exact);
* per-body poses and velocities come from MuJoCo forward kinematics on
  ``scene_27dof_identified.xml``, which reproduces the stored Isaac values to
  1e-6 m / 2e-7 quat.

Every regenerated array is written as ``original + (fk_new - fk_old)`` rather
than as a raw replacement, so frames the smoother did not touch keep their
exact source values instead of picking up reconstruction residue.

Run it in the ``gmr`` environment (the one that has MuJoCo)::

    conda run -n gmr python scripts/mimic/smooth_motion_apex_humanoid_ultra.py \
        --input  .../ustc1_rightstand_stand_transition.npz \
        --output .../ustc1_rightstand_stand_transition_smoothapex.npz

``--dry-run`` prints the report without writing.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import mujoco

SIM2SIM = Path("/home/zxh/ustc_humanoid/unitree_mujoco/sim2sim/humanoid_ultra/sim2sim.py")
SCENE = Path("/home/zxh/ustc_humanoid/unitree_mujoco/unitree_robots/humanoid_ultra/scene_27dof_identified.xml")
LEFT_FOOT_BODY = 20  # left_ankle_roll_link in the Isaac body order


def _sim2sim_names():
    """Joint/body order of the reference NPZ, taken from the deployment script."""
    spec = importlib.util.spec_from_file_location("_s2s", SIM2SIM)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_s2s"] = module
    spec.loader.exec_module(module)
    return module.ISAAC_27DOF_JOINTS, module.ISAAC_27DOF_BODIES, module.DEPLOYMENT_POSITION_LIMITS


def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_to_rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    vec = q[..., 1:]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm[..., 0], q[..., 0])
    return np.where(norm > 1e-12, vec / np.maximum(norm, 1e-12) * angle[..., None], np.zeros_like(vec))


def rotvec_to_quat(r):
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = angle / 2.0
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(half), r * scale], axis=-1)


class Kinematics:
    """MuJoCo forward kinematics in the Isaac joint/body order."""

    def __init__(self, joints, bodies):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.qpos_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints
        ]
        self.dof_adr = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints
        ]
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies]

    def run(self, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        frames, bodies = len(joint_pos), len(self.body_ids)
        pos = np.zeros((frames, bodies, 3))
        quat = np.zeros((frames, bodies, 4))
        lin = np.zeros((frames, bodies, 3))
        ang = np.zeros((frames, bodies, 3))
        vel6 = np.zeros(6)
        rot = np.zeros(9)
        data = self.data
        for f in range(frames):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = root_pos[f]
            data.qpos[3:7] = root_quat[f]
            for k, adr in enumerate(self.qpos_adr):
                data.qpos[adr] = joint_pos[f, k]
            mujoco.mju_quat2Mat(rot, root_quat[f])
            data.qvel[0:3] = root_lin_vel[f]
            # MuJoCo stores a free joint's angular velocity in the body frame.
            data.qvel[3:6] = rot.reshape(3, 3).T @ root_ang_vel[f]
            for k, adr in enumerate(self.dof_adr):
                data.qvel[adr] = joint_vel[f, k]
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            mujoco.mj_comVel(self.model, data)
            pos[f] = data.xpos[self.body_ids]
            quat[f] = data.xquat[self.body_ids]
            for i, body in enumerate(self.body_ids):
                mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel6, 0)
                ang[f, i] = vel6[0:3]
                lin[f, i] = vel6[3:6]
        return pos, quat, lin, ang


def blend_weight(frames, f0, f1, ramp):
    """1 inside the core, smoothstep to 0 at both window edges."""
    w = np.zeros(frames)
    core0, core1 = f0 + ramp, f1 - ramp
    w[core0 : core1 + 1] = 1.0
    for f in range(f0, core0):
        s = (f - f0) / ramp
        w[f] = s * s * (3 - 2 * s)
    for f in range(core1 + 1, f1 + 1):
        s = (f1 - f) / ramp
        w[f] = s * s * (3 - 2 * s)
    return w


def smooth_columns(x, idx, w, degree):
    t = (idx - idx.mean()) / (len(idx) / 2.0)
    out = x.copy()
    for col in range(x.shape[1]):
        fit = np.polyval(np.polyfit(t, x[idx, col], degree), t)
        out[idx, col] = w[idx] * fit + (1.0 - w[idx]) * x[idx, col]
    return out


def central_diff(a, fps):
    v = np.zeros_like(a)
    v[1:-1] = (a[2:] - a[:-2]) * fps / 2.0
    v[0] = (a[1] - a[0]) * fps
    v[-1] = (a[-1] - a[-2]) * fps
    return v


def so3_diff(q, fps):
    frames = len(q)
    out = np.zeros((frames, 3))
    for f in range(frames):
        i0, i1 = max(f - 1, 0), min(f + 1, frames - 1)
        rel = qmul(q[i1], qconj(q[i0]))
        vec = rel[1:]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            continue
        angle = 2.0 * np.arctan2(norm, rel[0])
        if angle > np.pi:
            angle -= 2.0 * np.pi
        out[f] = vec / norm * angle / ((i1 - i0) / fps)
    return out


def lift_report(foot_z, fps, t_lo=9.0, t_hi=11.0, land_z=0.30):
    """Peak height, and how far the foot climbs again after dipping."""
    lo, hi = int(t_lo * fps), int(t_hi * fps)
    peak = lo + int(np.argmax(foot_z[lo:hi]))
    end = peak
    while end < hi - 1 and foot_z[end] > land_z:
        end += 1
    hold = foot_z[peak : end + 1]
    # How far the foot climbs again after any dip: 0 for a clean single lift.
    running_min = np.minimum.accumulate(hold)
    rebound = float((hold - running_min).max())
    humps = sum(1 for i in range(1, len(hold) - 1) if hold[i] >= hold[i - 1] and hold[i] > hold[i + 1])
    deltas = np.diff(hold)
    signs = np.sign(deltas)
    signs = signs[signs != 0]
    reversals = int((np.diff(signs) != 0).sum())
    return {
        "peak": float(foot_z[peak]),
        "peak_t": peak / fps,
        "rebound": rebound,
        "extra_humps": humps,
        "reversals": reversals,
        "touchdown_t": end / fps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--f0", type=int, default=476, help="First frame of the smoothing window (inclusive).")
    parser.add_argument("--f1", type=int, default=524, help="Last frame of the smoothing window (inclusive).")
    parser.add_argument("--ramp", type=int, default=10, help="Smoothstep cross-fade length in frames.")
    parser.add_argument("--degree", type=int, default=3, help="Least-squares polynomial degree.")
    parser.add_argument("--keep-root", action="store_true", help="Leave the root pose untouched (joints only).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run is given")
    if args.f1 - args.f0 < 2 * args.ramp + 1:
        parser.error("window is too short for the requested ramp")

    joints, bodies, limits = _sim2sim_names()
    src = np.load(args.input)
    fps = float(np.asarray(src["fps"]).reshape(-1)[0])
    jp0 = src["joint_pos"].astype(np.float64)
    jv0 = src["joint_vel"].astype(np.float64)
    bp0 = src["body_pos_w"].astype(np.float64)
    bq0 = src["body_quat_w"].astype(np.float64)
    blv0 = src["body_lin_vel_w"].astype(np.float64)
    bav0 = src["body_ang_vel_w"].astype(np.float64)
    frames = len(jp0)

    w = blend_weight(frames, args.f0, args.f1, args.ramp)
    idx = np.arange(args.f0, args.f1 + 1)
    jp1 = smooth_columns(jp0, idx, w, args.degree)

    rp0, rq0 = bp0[:, 0].copy(), bq0[:, 0].copy()
    if args.keep_root:
        rp1, rq1 = rp0, rq0
    else:
        rp1 = smooth_columns(rp0, idx, w, args.degree)
        ref = rq0[(args.f0 + args.f1) // 2]
        rotvec = quat_to_rotvec(qmul(rq0, qconj(np.broadcast_to(ref, rq0.shape))))
        rq1 = qmul(rotvec_to_quat(smooth_columns(rotvec, idx, w, args.degree)), np.broadcast_to(ref, rq0.shape))
        rq1 /= np.linalg.norm(rq1, axis=-1, keepdims=True)
        rq1 = np.where((rq1 * rq0).sum(-1, keepdims=True) < 0, -rq1, rq1)

    jv1 = jv0 + (central_diff(jp1, fps) - central_diff(jp0, fps))
    rlv1 = blv0[:, 0] + (central_diff(rp1, fps) - central_diff(rp0, fps))
    rav1 = bav0[:, 0] + (so3_diff(rq1, fps) - so3_diff(rq0, fps))

    kin = Kinematics(joints, bodies)
    old = kin.run(rp0, rq0, blv0[:, 0], bav0[:, 0], jp0, jv0)
    new = kin.run(rp1, rq1, rlv1, rav1, jp1, jv1)
    bp1 = bp0 + (new[0] - old[0])
    bq1 = qmul(qmul(new[1], qconj(old[1])), bq0)
    bq1 /= np.linalg.norm(bq1, axis=-1, keepdims=True)
    bq1 = np.where((bq1 * bq0).sum(-1, keepdims=True) < 0, -bq1, bq1)
    blv1 = blv0 + (new[2] - old[2])
    bav1 = bav0 + (new[3] - old[3])

    # Frames the edit can reach: the window itself, plus one frame on each side
    # because the velocities are central differences.  Everything else is
    # restored to the source values so the two files stay bit-identical there.
    touched = np.zeros(frames, bool)
    touched[max(args.f0 - 1, 0) : min(args.f1 + 2, frames)] = True
    for new_array, old_array in ((jp1, jp0), (jv1, jv0), (bp1, bp0), (bq1, bq0), (blv1, blv0), (bav1, bav0)):
        new_array[~touched] = old_array[~touched]
    print(f"window frames {args.f0}..{args.f1}  ({args.f0 / fps:.2f}..{args.f1 / fps:.2f} s)  "
          f"degree={args.degree} ramp={args.ramp} root={'kept' if args.keep_root else 'smoothed'}")
    for name, a0, a1 in (
        ("joint_pos", jp0, jp1),
        ("joint_vel", jv0, jv1),
        ("body_pos_w", bp0, bp1),
        ("body_quat_w", bq0, bq1),
        ("body_lin_vel_w", blv0, blv1),
        ("body_ang_vel_w", bav0, bav1),
    ):
        delta = np.abs(a1 - a0)
        print(f"  {name:<15s} max|delta| in window {delta[touched].max():.4f}   outside {delta[~touched].max():.3e}")

    before = lift_report(bp0[:, LEFT_FOOT_BODY, 2], fps)
    after = lift_report(bp1[:, LEFT_FOOT_BODY, 2], fps)
    print("  left foot lift          before -> after")
    for key in ("peak", "peak_t", "rebound", "extra_humps", "reversals", "touchdown_t"):
        print(f"    {key:<18s} {before[key]:>8.3f} -> {after[key]:>8.3f}")
    print(f"  max |joint_vel| in window {np.abs(jv0[touched]).max():.2f} -> {np.abs(jv1[touched]).max():.2f} rad/s")

    violations = []
    for k, name in enumerate(joints):
        lo, hi = limits[name]
        margin = max(jp1[:, k].max() - hi, lo - jp1[:, k].min())
        if margin > 0:
            violations.append((name, float(margin)))
    print("  deployment joint limits:", "OK" if not violations else f"VIOLATED {violations}")

    if args.output:
        np.savez(
            args.output,
            fps=src["fps"],
            joint_pos=jp1.astype(np.float32),
            joint_vel=jv1.astype(np.float32),
            body_pos_w=bp1.astype(np.float32),
            body_quat_w=bq1.astype(np.float32),
            body_lin_vel_w=blv1.astype(np.float32),
            body_ang_vel_w=bav1.astype(np.float32),
        )
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
