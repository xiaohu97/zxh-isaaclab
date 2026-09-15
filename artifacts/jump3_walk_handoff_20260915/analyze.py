"""Offline reference/ONNX inspection. Does not connect to or command a robot.

Run with the ustc_isaaclab Python environment. Reference motion is a surrogate
for robot measurements; these outputs are not a closed-loop stability test.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from scipy.spatial.transform import Rotation
import yaml


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
POLICIES = ROOT / "deploy/robots/g1_29dof/config/policy"
JUMP = POLICIES / "mimic/jump3"
WALK = POLICIES / "velocity"
jump_cfg = yaml.safe_load((JUMP / "params/deploy.yaml").read_text())
walk_cfg = yaml.safe_load((WALK / "params/deploy.yaml").read_text())
canonical_walk = yaml.safe_load((WALK / "0914_yawfix/params/deploy.yaml").read_text())
ids = np.asarray(walk_cfg["joint_ids_map"], dtype=int)
reference = np.loadtxt(JUMP / "params/jump3_waist15.csv", delimiter=",")
fps = 120.0
t = np.arange(len(reference)) / fps
q_motor = reference[:, 7:]
q = q_motor[:, ids]
dq = np.diff(q, axis=0) * fps
dq = np.vstack([dq, dq[-1]])  # Same forward derivative/end duplication as C++.
base_rotation = Rotation.from_quat(reference[:, 3:7])
base_rpy = base_rotation.as_euler("xyz", degrees=True)
torso_rotation = (
    base_rotation
    * Rotation.from_euler("z", q_motor[:, 12])
    * Rotation.from_euler("x", q_motor[:, 13])
    * Rotation.from_euler("y", q_motor[:, 14])
)
torso_tilt = np.degrees(np.arccos(np.clip(torso_rotation.as_matrix()[:, 2, 2], -1, 1)))
root_velocity = np.gradient(reference[:, :3], 1 / fps, axis=0)
omega = (base_rotation[:-1].inv() * base_rotation[1:]).as_rotvec() * fps
omega = np.vstack([omega, omega[-1]])
gravity = base_rotation.inv().apply(np.tile([0.0, 0.0, -1.0], (len(t), 1)))
default = np.asarray(walk_cfg["default_joint_pos"])
action_cfg = walk_cfg["actions"]["JointPositionAction"]
scale = np.asarray(action_cfg["scale"])
offset = np.asarray(action_cfg["offset"])
leg = ids < 12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


runner = ReferenceEvaluator(onnx.load(str(WALK / "exported/policy.onnx")))


def infer(label, jp, jv, av, pg, previous):
    # C++ reset repeats one sample for each term; history is term-major.
    terms = {
        "base_ang_vel": av,
        "projected_gravity": pg,
        "velocity_commands": np.zeros(3),
        "joint_pos_rel": jp - default,
        "joint_vel_rel": jv,
        "last_action": previous,
    }
    packed = []
    for name, cfg in walk_cfg["observations"].items():
        value = terms[name].copy()
        if cfg["clip"] is not None:
            value = np.clip(value, *cfg["clip"])
        if cfg["scale"] is not None:
            value *= np.asarray(cfg["scale"])
        packed.append(np.tile(value, cfg["history_length"]))
    obs = np.concatenate(packed).astype(np.float32)[None, :]
    assert obs.shape == (1, 480)
    action = runner.run(None, {"obs": obs})[0][0]
    assert np.isfinite(action).all()
    processed = action.copy()
    if action_cfg.get("raw_clip") is not None:
        processed = np.clip(processed, *action_cfg["raw_clip"])
    target = offset + scale * processed
    if action_cfg.get("clip") is not None:
        limits = np.asarray(action_cfg["clip"])
        target = np.clip(target, limits[:, 0], limits[:, 1])
    waist = int(np.flatnonzero(ids == 14)[0])
    return {
        "scenario": label,
        "max_abs_raw_action": float(np.abs(action).max()),
        "max_leg_target_minus_reference_rad": float(np.abs(target[leg] - jp[leg]).max()),
        "waist_pitch_target_rad": float(target[waist]),
        "waist_pitch_target_deg": float(np.degrees(target[waist])),
        "action": action.tolist(),
        "target_in_policy_joint_order": target.tolist(),
    }


metrics = {
    "scope": "Reference-state analysis and single-step ONNX evaluation; not measured robot data or closed-loop simulation.",
    "reference_frames": len(t),
    "fps": fps,
    "loader_duration_s": len(t) / fps,
    "last_reference_timestamp_s": float(t[-1]),
    "nominal_automatic_exit_episode_time_s": 1.8,
    "terminal": {
        "pelvis_pitch_deg": float(base_rpy[-1, 1]),
        "torso_tilt_deg": float(torso_tilt[-1]),
        "waist_pitch_deg": float(np.degrees(q_motor[-1, 14])),
        "root_velocity_world_m_s": root_velocity[-1].tolist(),
        "root_horizontal_speed_m_s": float(np.linalg.norm(root_velocity[-1, :2])),
        "max_abs_leg_joint_velocity_rad_s": float(np.abs(dq[-1, leg]).max()),
        "max_leg_pose_difference_from_walk_default_rad": float(np.abs(q[-1, leg] - default[leg]).max()),
    },
    "equal_jump_walk_fields": {k: jump_cfg[k] == walk_cfg[k] for k in ["joint_ids_map", "step_dt", "stiffness", "damping"]},
    "default_pose_max_difference_rad": float(np.abs(np.asarray(jump_cfg["default_joint_pos"]) - default).max()),
    "walk_config_top_level_differences_from_yawfix": [k for k in canonical_walk if canonical_walk[k] != walk_cfg.get(k)],
    "walk_loaded_raw_clip": action_cfg.get("raw_clip"),
    "walk_canonical_raw_clip": canonical_walk["actions"]["JointPositionAction"].get("raw_clip"),
    "jump_sha256": digest(JUMP / "exported/policy.onnx"),
    "walk_sha256": digest(WALK / "exported/policy.onnx"),
    "jump_matches_0915_28000": digest(JUMP / "exported/policy.onnx") == digest(JUMP / "exported/0915_28000/policy.onnx"),
    "walk_matches_0914_yawfix": digest(WALK / "exported/policy.onnx") == digest(WALK / "0914_yawfix/exported/policy.onnx"),
    "offline_inference": [
        infer("walk default, stationary, zero command/action", default, np.zeros(29), np.zeros(3), np.array([0., 0., -1.]), np.zeros(29)),
        infer("terminal reference, reset history, zero command/action", q[-1], dq[-1], omega[-1], gravity[-1], np.zeros(29)),
        infer("terminal reference, seed action from reference position (not actual previous command)", q[-1], dq[-1], omega[-1], gravity[-1], (q[-1] - offset) / scale),
    ],
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
axes[0].plot(t, base_rpy[:, 1], label="Pelvis pitch")
axes[0].plot(t, torso_tilt, label="Torso tilt")
axes[0].set_ylabel("Angle (deg)")
axes[0].legend(loc="upper left")
axes[1].plot(t, root_velocity[:, 2], label="Vertical velocity")
axes[1].plot(t, np.linalg.norm(root_velocity[:, :2], axis=1), label="Horizontal speed")
axes[1].set_ylabel("Root velocity (m/s)")
axes[1].legend(loc="upper left")
axes[2].plot(t, np.abs(dq[:, leg]).max(axis=1), color="tab:purple")
axes[2].set_ylabel("Max leg |dq| (rad/s)")
axes[2].set_xlabel("Reference time (s)")
for ax in axes:
    ax.axvline(t[-1], color="tab:red", linestyle="--", alpha=.8)
    ax.axvline(1.8, color="black", linestyle=":", alpha=.8)
    ax.axhline(0, color="gray", linewidth=.6)
    ax.set_xlim(1.15, 1.83)
    ax.grid(alpha=.2)
axes[0].set_title("Jump3 reference ends during recovery\nRed: last CSV frame 1.775 s; black: nominal switch at episode time 1.80 s")
fig.savefig(OUT / "reference_end.png", dpi=160)
print(json.dumps({k: v for k, v in metrics.items() if k != "offline_inference"}, indent=2))
for case in metrics["offline_inference"]:
    print({k: v for k, v in case.items() if k not in {"action", "target_in_policy_joint_order"}})
