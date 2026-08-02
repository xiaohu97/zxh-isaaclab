#!/usr/bin/env python3
"""Add short Humanoid Ultra default-pose transitions to a SOMA CSV motion.

The input and output keep the SOMA layout and input frame rate.  The generated
motion starts at the current 27-DoF Mimic robot's default pose, reaches the
first captured frame through a quintic blend, preserves the original clip, and
returns to the default pose through a second quintic blend.  An optional
terminal hold can keep that final default pose in the exported reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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

# Mirrors HUMANOIDULTRA27DOF_CFG, which is copied by the identified Mimic asset.
DEFAULT_JOINT_POSITION = {
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_hip_pitch_joint": 0.289936,
    "left_knee_joint": 0.742326,
    "left_ankle_pitch_joint": 0.409573,
    "left_ankle_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_pitch_joint": 0.289936,
    "right_knee_joint": 0.742326,
    "right_ankle_pitch_joint": 0.409573,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": -1.5707963,
    "left_elbow_joint": -0.6,
    "left_wrist_yaw_joint": 1.5707963,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": -0.25,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 1.5707963,
    "right_elbow_joint": 0.6,
    "right_wrist_yaw_joint": -1.5707963,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
}

CSV_PREFIX = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)
EXPECTED_HEADER = CSV_PREFIX + tuple(f"{name}_dof" for name in CSV_JOINT_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, required=True)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--transition-seconds", type=float, default=1.0)
    parser.add_argument(
        "--terminal-hold-seconds",
        type=float,
        default=0.0,
        help="Append this many seconds of the final default standing pose.",
    )
    parser.add_argument("--stand-root-height-cm", type=float, default=100.5)
    args = parser.parse_args()
    if args.input_fps <= 0.0 or args.output_fps <= 0.0:
        parser.error("frame rates must be positive")
    if not 0.0 < args.transition_seconds <= 1.0:
        parser.error("--transition-seconds must be in (0, 1]")
    if args.terminal_hold_seconds < 0.0:
        parser.error("--terminal-hold-seconds must be non-negative")
    if args.stand_root_height_cm <= 0.0:
        parser.error("--stand-root-height-cm must be positive")
    return args


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(np.asarray(euler, dtype=np.float64), -1, 0)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return normalize_quaternion(
        np.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            axis=-1,
        )
    )


def quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


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
    return (
        np.sin((1.0 - blend) * angle)[:, None] / np.sin(angle) * start[None, :]
        + np.sin(blend * angle)[:, None] / np.sin(angle) * end[None, :]
    )


def quintic_blend(interval_count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, interval_count + 1, dtype=np.float64)
    return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3


def interpolate(start: np.ndarray, end: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return (1.0 - blend[:, None]) * start[None, :] + blend[:, None] * end[None, :]


def main() -> None:
    args = parse_args()
    input_path = args.input_csv.resolve()
    output_path = args.output_csv.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    header = tuple(input_path.read_text(encoding="utf-8").splitlines()[0].split(","))
    if header != EXPECTED_HEADER:
        raise ValueError("Unexpected Humanoid Ultra SOMA CSV header or joint order")
    source = np.loadtxt(input_path, delimiter=",", skiprows=1, ndmin=2)
    if source.shape[1] != len(EXPECTED_HEADER) or source.shape[0] < 2:
        raise ValueError(f"Unexpected motion shape: {source.shape}")

    root_pos_cm = source[:, 1:4]
    root_quat = euler_xyz_to_quaternion(np.deg2rad(source[:, 4:7]))
    joint_pos = np.deg2rad(source[:, 7:])
    default_joint = np.asarray([DEFAULT_JOINT_POSITION[name] for name in CSV_JOINT_ORDER])

    transition_frames = int(round(args.transition_seconds * args.input_fps))
    if transition_frames < 2:
        raise ValueError("Transition must contain at least two frame intervals")
    blend = quintic_blend(transition_frames)

    start_stand_pos = root_pos_cm[0].copy()
    start_stand_pos[2] = args.stand_root_height_cm
    end_stand_pos = root_pos_cm[-1].copy()
    end_stand_pos[2] = args.stand_root_height_cm
    start_stand_quat = yaw_quaternion(quaternion_yaw(root_quat[0]))
    end_stand_quat = yaw_quaternion(quaternion_yaw(root_quat[-1]))

    prepare_joint = interpolate(default_joint, joint_pos[0], blend)
    recover_joint = interpolate(joint_pos[-1], default_joint, blend)
    prepare_pos = interpolate(start_stand_pos, root_pos_cm[0], blend)
    recover_pos = interpolate(root_pos_cm[-1], end_stand_pos, blend)
    prepare_quat = quaternion_slerp(start_stand_quat, root_quat[0], blend)
    recover_quat = quaternion_slerp(root_quat[-1], end_stand_quat, blend)

    output_joint = np.concatenate((prepare_joint, joint_pos[1:], recover_joint[1:]), axis=0)
    output_root_pos = np.concatenate((prepare_pos, root_pos_cm[1:], recover_pos[1:]), axis=0)
    output_root_quat = np.concatenate((prepare_quat, root_quat[1:], recover_quat[1:]), axis=0)

    terminal_hold_frames = int(round(args.terminal_hold_seconds * args.input_fps))
    if args.terminal_hold_seconds > 0.0 and terminal_hold_frames < 1:
        raise ValueError("Terminal hold must contain at least one input frame interval")

    # csv_to_npz_humanoid_ultra samples [0, duration).  Extend the terminal
    # default pose by one output interval so the exported NPZ ends exactly on it.
    padding_frames = max(1, int(np.ceil(args.input_fps / args.output_fps)))
    appended_frames = terminal_hold_frames + padding_frames
    output_joint = np.concatenate(
        (output_joint, np.repeat(output_joint[-1:], appended_frames, axis=0)), axis=0
    )
    output_root_pos = np.concatenate(
        (output_root_pos, np.repeat(output_root_pos[-1:], appended_frames, axis=0)), axis=0
    )
    output_root_quat = np.concatenate(
        (output_root_quat, np.repeat(output_root_quat[-1:], appended_frames, axis=0)), axis=0
    )

    output = np.concatenate(
        (
            np.arange(output_joint.shape[0], dtype=np.float64)[:, None],
            output_root_pos,
            np.rad2deg(quaternion_to_euler_xyz(output_root_quat)),
            np.rad2deg(output_joint),
        ),
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        output,
        delimiter=",",
        header=",".join(EXPECTED_HEADER),
        comments="",
        fmt=["%d"] + ["%.9f"] * (output.shape[1] - 1),
    )

    start_error = np.max(np.abs(joint_pos[0] - default_joint))
    end_error = np.max(np.abs(joint_pos[-1] - default_joint))
    print(f"Wrote {output_path}")
    print(
        f"Frames: {source.shape[0]} source + {transition_frames} prepare + "
        f"{transition_frames} recover + {terminal_hold_frames} terminal hold + "
        f"{padding_frames} converter padding = {output.shape[0]}"
    )
    print(
        "Endpoint max joint deviation before blending: "
        f"start={np.rad2deg(start_error):.1f} deg, end={np.rad2deg(end_error):.1f} deg"
    )


if __name__ == "__main__":
    main()
