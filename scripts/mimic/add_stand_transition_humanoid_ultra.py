#!/usr/bin/env python3
"""Build a 50 Hz Humanoid Ultra motion source with stand transitions.

This script reads an existing Mimic NPZ, keeps its root/joint trajectory, and
emits the soma-style CSV accepted by ``csv_to_npz_humanoid_ultra.py``.  The
resulting sequence is:

    stand hold -> quintic transition -> original motion
               -> quintic transition -> stand hold

Run the emitted CSV through the Isaac Lab converter so all rigid-body poses and
velocities are recomputed from the current robot asset.  Do not construct a
training NPZ by interpolating its body arrays independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ISAAC_JOINT_ORDER = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
)

# Newton/URDF depth-first order required by csv_to_npz_humanoid_ultra.py.
CSV_JOINT_ORDER = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
)

DEFAULT_JOINT_POSITION = {
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.25,
    "right_shoulder_pitch_joint": -0.25,
    "left_hip_pitch_joint": 0.289936,
    "right_hip_pitch_joint": 0.289936,
    "left_shoulder_roll_joint": -0.05,
    "right_shoulder_roll_joint": 0.05,
    "left_knee_joint": 0.742326,
    "right_knee_joint": 0.742326,
    "left_shoulder_yaw_joint": -1.5707963,
    "right_shoulder_yaw_joint": 1.5707963,
    "left_ankle_pitch_joint": 0.409573,
    "right_ankle_pitch_joint": 0.409573,
    "left_elbow_joint": -0.6,
    "right_elbow_joint": 0.6,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "left_wrist_yaw_joint": 1.5707963,
    "right_wrist_yaw_joint": -1.5707963,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--stand-hold-seconds", type=float, default=2.0)
    parser.add_argument("--transition-seconds", type=float, default=4.0)
    parser.add_argument("--stand-root-height", type=float, default=1.005)
    args = parser.parse_args()
    if args.stand_hold_seconds <= 0.0 or args.transition_seconds <= 0.0:
        parser.error("hold and transition durations must be positive")
    if args.stand_root_height <= 0.0:
        parser.error("stand root height must be positive")
    return args


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = normalize_quaternion(quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray((np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)))


def quaternion_slerp(start: np.ndarray, end: np.ndarray, blend: np.ndarray) -> np.ndarray:
    start = normalize_quaternion(start)
    end = normalize_quaternion(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start[None, :] + blend[:, None] * (end - start)[None, :]
        return normalize_quaternion(result)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - blend) * angle)[:, None] / sin_angle * start[None, :]
        + np.sin(blend * angle)[:, None] / sin_angle * end[None, :]
    )


def quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def quintic_blend(sample_count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)[1:]
    return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3


def interpolate(start: np.ndarray, end: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return (1.0 - blend[:, None]) * start[None, :] + blend[:, None] * end[None, :]


def main() -> None:
    args = parse_args()
    input_path = args.input_npz.resolve()
    output_path = args.output_csv.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path) as data:
        required = {"fps", "joint_pos", "body_pos_w", "body_quat_w"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Input NPZ is missing fields: {sorted(missing)}")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = normalize_quaternion(data["body_quat_w"][:, 0])

    if not np.isclose(fps, 50.0):
        raise ValueError(f"Expected a 50 Hz source motion, got {fps:g} Hz")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(ISAAC_JOINT_ORDER):
        raise ValueError(f"Expected joint_pos shape [frames, 27], got {joint_pos.shape}")
    if root_pos.shape != (joint_pos.shape[0], 3) or root_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError("Root arrays do not match joint_pos frame count")

    hold_frames = round(args.stand_hold_seconds * fps)
    transition_frames = round(args.transition_seconds * fps)
    stand_joint = np.asarray(
        [DEFAULT_JOINT_POSITION[name] for name in ISAAC_JOINT_ORDER], dtype=np.float64
    )
    start_stand_pos = root_pos[0].copy()
    start_stand_pos[2] = args.stand_root_height
    end_stand_pos = root_pos[-1].copy()
    end_stand_pos[2] = args.stand_root_height
    start_stand_quat = yaw_quaternion(quaternion_yaw(root_quat[0]))
    end_stand_quat = yaw_quaternion(quaternion_yaw(root_quat[-1]))

    prepare_blend = quintic_blend(transition_frames)
    recover_blend = quintic_blend(transition_frames)

    output_joint = np.concatenate(
        (
            np.repeat(stand_joint[None, :], hold_frames, axis=0),
            interpolate(stand_joint, joint_pos[0], prepare_blend),
            joint_pos[1:],
            interpolate(joint_pos[-1], stand_joint, recover_blend),
            np.repeat(stand_joint[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )
    output_root_pos = np.concatenate(
        (
            np.repeat(start_stand_pos[None, :], hold_frames, axis=0),
            interpolate(start_stand_pos, root_pos[0], prepare_blend),
            root_pos[1:],
            interpolate(root_pos[-1], end_stand_pos, recover_blend),
            np.repeat(end_stand_pos[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )
    output_root_quat = np.concatenate(
        (
            np.repeat(start_stand_quat[None, :], hold_frames, axis=0),
            quaternion_slerp(start_stand_quat, root_quat[0], prepare_blend),
            root_quat[1:],
            quaternion_slerp(root_quat[-1], end_stand_quat, recover_blend),
            np.repeat(end_stand_quat[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )

    # The existing Isaac converter samples [0, duration), so append one
    # duplicate input row.  Its exported NPZ then contains exactly the desired
    # sequence above, including the final stand frame.
    output_joint = np.concatenate((output_joint, output_joint[-1:]), axis=0)
    output_root_pos = np.concatenate((output_root_pos, output_root_pos[-1:]), axis=0)
    output_root_quat = np.concatenate((output_root_quat, output_root_quat[-1:]), axis=0)

    csv_indices = np.asarray([ISAAC_JOINT_ORDER.index(name) for name in CSV_JOINT_ORDER])
    euler_deg = np.rad2deg(quaternion_to_euler_xyz(output_root_quat))
    rows = np.concatenate(
        (
            np.arange(1, output_joint.shape[0] + 1, dtype=np.float64)[:, None],
            100.0 * output_root_pos,
            euler_deg,
            np.rad2deg(output_joint[:, csv_indices]),
        ),
        axis=1,
    )
    header = ["Frame", "RootX_cm", "RootY_cm", "RootZ_cm", "RootRX_deg", "RootRY_deg", "RootRZ_deg"]
    header.extend(CSV_JOINT_ORDER)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, rows, delimiter=",", header=",".join(header), comments="", fmt="%.9f")

    desired_frames = output_joint.shape[0] - 1
    original_start = hold_frames + transition_frames - 1
    original_end = original_start + joint_pos.shape[0] - 1
    print(f"Wrote converter input: {output_path}")
    print(f"Expected exported frames: {desired_frames} at {fps:g} Hz ({desired_frames / fps:.2f}s)")
    print(
        "Segments: stand=[0, {}], prepare=[{}, {}], original=[{}, {}], "
        "recover=[{}, {}], stand_end=[{}, {}]".format(
            hold_frames - 1,
            hold_frames,
            original_start,
            original_start,
            original_end,
            original_end + 1,
            original_end + transition_frames,
            original_end + transition_frames + 1,
            desired_frames - 1,
        )
    )


if __name__ == "__main__":
    main()
