#!/usr/bin/env python3
"""Collect 1 kHz Humanoid Ultra identification data from MuJoCo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from sim2sim import (
    URDF_12DOF_JOINTS,
    URDF_27DOF_JOINTS,
    HumanoidUltraSim2Sim,
)


SAMPLE_RATE_HZ = 1000
SAMPLE_DT = 1.0 / SAMPLE_RATE_HZ
PD_RATE_HZ = 200
POLICY_RATE_HZ = 50
PD_INTERVAL = SAMPLE_RATE_HZ // PD_RATE_HZ
POLICY_INTERVAL = SAMPLE_RATE_HZ // POLICY_RATE_HZ
GRAVITY = 9.81
IMU_SITE_NAMES = {12: "base_imu", 27: "imu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--dof", type=int, choices=(12, 27), required=True)
    parser.add_argument("--duration", type=float, default=10.0, help="Recorded duration in seconds.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Unrecorded policy-controlled settling time.",
    )
    parser.add_argument(
        "--profile",
        choices=("constant", "identification"),
        default="constant",
        help="Constant/manual command or deterministic multi-frequency command excitation.",
    )
    parser.add_argument("--static-seconds", type=float, default=2.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--robot-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument(
        "--elastic-band",
        action="store_true",
        help="Enable shoulder support. Disabled by default because it is an unmeasured external force.",
    )
    parser.add_argument("--band-lift", type=float, default=0.0)
    parser.add_argument("--band-anchor-height", type=float, default=3.0)
    parser.add_argument("--band-stiffness", type=float, default=500.0)
    parser.add_argument("--band-damping", type=float, default=100.0)
    parser.add_argument("--band-support-ratio", type=float, default=0.3)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def identification_command(t: float, duration: float, static_seconds: float) -> np.ndarray:
    if t < static_seconds or t >= duration - static_seconds:
        return np.zeros(3, dtype=np.float64)

    active_time = t - static_seconds
    active_duration = duration - 2.0 * static_seconds
    ramp = min(1.0, active_time, active_duration - active_time)
    vx = 0.35 * math.sin(2.0 * math.pi * 0.23 * active_time)
    vx += 0.15 * math.sin(2.0 * math.pi * 0.53 * active_time)
    vy = 0.18 * math.sin(2.0 * math.pi * 0.17 * active_time)
    vy += 0.08 * math.sin(2.0 * math.pi * 0.41 * active_time)
    yaw = 0.55 * math.sin(2.0 * math.pi * 0.13 * active_time)
    yaw += 0.25 * math.sin(2.0 * math.pi * 0.37 * active_time)
    command = ramp * np.asarray([vx, vy, yaw], dtype=np.float64)
    return np.clip(command, (-0.6, -0.5, -1.57), (1.0, 0.5, 1.57))


class IdentificationRecorder:
    def __init__(
        self,
        simulator: HumanoidUltraSim2Sim,
        joint_names: tuple[str, ...],
        sample_count: int,
        csv_path: Path,
        contact_threshold: float,
    ):
        self.simulator = simulator
        self.model = simulator.model
        self.data = simulator.data
        self.joint_names = joint_names
        self.sample_count = sample_count
        self.contact_threshold = contact_threshold
        self.qpos_indices = np.asarray(
            [self.model.joint(name).qposadr[0] for name in joint_names], dtype=np.int32
        )
        self.qvel_indices = np.asarray(
            [self.model.joint(name).dofadr[0] for name in joint_names], dtype=np.int32
        )
        self.root_body_id = self.model.body("dummy_link").id
        self.foot_body_ids = (
            self.model.body("left_ankle_roll_link").id,
            self.model.body("right_ankle_roll_link").id,
        )
        self.foot_site_ids = (
            self.model.site("left_foot").id,
            self.model.site("right_foot").id,
        )

        dof = len(joint_names)
        self.low_q = np.empty((7 + dof, sample_count), dtype=np.float64)
        self.dq = np.empty((6 + dof, sample_count), dtype=np.float64)
        self.ddq = np.empty((6 + dof, sample_count), dtype=np.float64)
        self.tau = np.empty((dof, sample_count), dtype=np.float64)
        self.contact = np.empty((2, sample_count), dtype=np.int8)
        self.ee_force = np.empty((12, sample_count), dtype=np.float64)
        self.previous_quaternion_xyzw: np.ndarray | None = None
        self.body_velocity = np.zeros(6, dtype=np.float64)
        self.body_acceleration = np.zeros(6, dtype=np.float64)
        self.contact_wrench = np.zeros(6, dtype=np.float64)

        self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._csv_header())

    def _csv_header(self) -> list[str]:
        header = [
            "timestamp_s",
            "sim_time_s",
            "base_position_x_m",
            "base_position_y_m",
            "base_position_z_m",
            "base_quaternion_w",
            "base_quaternion_x",
            "base_quaternion_y",
            "base_quaternion_z",
            "base_velocity_body_x_mps",
            "base_velocity_body_y_mps",
            "base_velocity_body_z_mps",
            "imu_gyro_body_x_radps",
            "imu_gyro_body_y_radps",
            "imu_gyro_body_z_radps",
            "imu_accel_body_x_mps2",
            "imu_accel_body_y_mps2",
            "imu_accel_body_z_mps2",
            "base_angular_accel_body_x_radps2",
            "base_angular_accel_body_y_radps2",
            "base_angular_accel_body_z_radps2",
        ]
        for name in self.joint_names:
            header.extend(
                (
                    f"{name}_q_rad",
                    f"{name}_dq_radps",
                    f"{name}_ddq_radps2",
                    f"{name}_tau_nm",
                )
            )
        header.extend(
            (
                "left_contact",
                "right_contact",
                "left_foot_fx_world_n",
                "left_foot_fy_world_n",
                "left_foot_fz_world_n",
                "left_foot_mx_world_nm",
                "left_foot_my_world_nm",
                "left_foot_mz_world_nm",
                "right_foot_fx_world_n",
                "right_foot_fy_world_n",
                "right_foot_fz_world_n",
                "right_foot_mx_world_nm",
                "right_foot_my_world_nm",
                "right_foot_mz_world_nm",
                "command_vx_mps",
                "command_vy_mps",
                "command_yaw_rate_radps",
                "elastic_band_enabled",
            )
        )
        return header

    def _foot_wrenches(self) -> tuple[np.ndarray, np.ndarray]:
        wrenches = np.zeros((2, 6), dtype=np.float64)
        has_contact = np.zeros(2, dtype=bool)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            for foot_index, foot_body_id in enumerate(self.foot_body_ids):
                if body1 == foot_body_id:
                    sign = -1.0
                elif body2 == foot_body_id:
                    sign = 1.0
                else:
                    continue

                mujoco.mj_contactForce(
                    self.model, self.data, contact_index, self.contact_wrench
                )
                contact_rotation = np.asarray(contact.frame).reshape(3, 3)
                force_world = sign * (contact_rotation.T @ self.contact_wrench[:3])
                torque_world = sign * (contact_rotation.T @ self.contact_wrench[3:])
                foot_origin = self.data.site_xpos[self.foot_site_ids[foot_index]]
                torque_world += np.cross(contact.pos - foot_origin, force_world)
                wrenches[foot_index, :3] += force_world
                wrenches[foot_index, 3:] += torque_world
                if np.linalg.norm(force_world) > self.contact_threshold:
                    has_contact[foot_index] = True
        return wrenches, has_contact

    def capture(self, sample_index: int, timestamp: float, command: np.ndarray) -> None:
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_id,
            self.body_velocity,
            1,
        )
        mujoco.mj_objectAcceleration(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_id,
            self.body_acceleration,
            1,
        )

        quaternion_wxyz = self.data.qpos[3:7].copy()
        quaternion_wxyz /= np.linalg.norm(quaternion_wxyz)
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        if (
            self.previous_quaternion_xyzw is not None
            and np.dot(quaternion_xyzw, self.previous_quaternion_xyzw) < 0.0
        ):
            quaternion_xyzw *= -1.0
            quaternion_wxyz *= -1.0
        self.previous_quaternion_xyzw = quaternion_xyzw.copy()

        joint_pos = self.data.qpos[self.qpos_indices].copy()
        joint_vel = self.data.qvel[self.qvel_indices].copy()
        joint_acc = self.data.qacc[self.qvel_indices].copy()
        joint_torque = self.data.qfrc_actuator[self.qvel_indices].copy()
        base_linear_velocity = self.body_velocity[3:].copy()
        base_angular_velocity = self.data.sensor("BodyGyro").data.copy()
        imu_acceleration = self.data.sensor("BodyAcc").data.copy()
        base_angular_acceleration = self.body_acceleration[:3].copy()
        foot_wrenches, foot_contacts = self._foot_wrenches()
        contact_values = np.where(foot_contacts, 1, 2).astype(np.int8)

        self.low_q[:, sample_index] = np.concatenate(
            (self.data.qpos[:3], quaternion_xyzw, joint_pos)
        )
        self.dq[:, sample_index] = np.concatenate(
            (base_linear_velocity, base_angular_velocity, joint_vel)
        )
        self.ddq[:, sample_index] = np.concatenate(
            (imu_acceleration, base_angular_acceleration, joint_acc)
        )
        self.tau[:, sample_index] = joint_torque
        self.contact[:, sample_index] = contact_values
        self.ee_force[:, sample_index] = foot_wrenches.reshape(-1)

        row: list[float | int] = [
            timestamp,
            float(self.data.time),
            *self.data.qpos[:3],
            *quaternion_wxyz,
            *base_linear_velocity,
            *base_angular_velocity,
            *imu_acceleration,
            *base_angular_acceleration,
        ]
        for values in zip(joint_pos, joint_vel, joint_acc, joint_torque):
            row.extend(values)
        row.extend(
            (
                int(contact_values[0]),
                int(contact_values[1]),
                *foot_wrenches.reshape(-1),
                *command,
                int(self.simulator.elastic_band.enabled),
            )
        )
        self.csv_writer.writerow(row)

    def close(self) -> None:
        self.csv_file.close()

    def trim(self, sample_count: int) -> None:
        self.low_q = self.low_q[:, :sample_count]
        self.dq = self.dq[:, :sample_count]
        self.ddq = self.ddq[:, :sample_count]
        self.tau = self.tau[:, :sample_count]
        self.contact = self.contact[:, :sample_count]
        self.ee_force = self.ee_force[:, :sample_count]

    def validate(self) -> dict:
        matrices = {
            "low_q": self.low_q,
            "dq": self.dq,
            "ddq": self.ddq,
            "tau": self.tau,
            "contact": self.contact,
            "ee_force": self.ee_force,
        }
        columns = {matrix.shape[1] for matrix in matrices.values()}
        if len(columns) != 1:
            raise RuntimeError(f"DAT column counts differ: {columns}")
        for name, matrix in matrices.items():
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise RuntimeError(f"{name} contains invalid values or dimensions.")
        if not set(np.unique(self.contact)).issubset({1, 2}):
            raise RuntimeError("Contact labels must contain only 1 or 2.")

        first_base_state = self.low_q[:7, 0]
        if np.allclose(first_base_state, 0.0):
            raise RuntimeError("The first base state is uninitialized.")
        quaternion_norms = np.linalg.norm(self.low_q[3:7], axis=0)
        quaternion_dots = np.sum(self.low_q[3:7, 1:] * self.low_q[3:7, :-1], axis=0)
        if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-5:
            raise RuntimeError("Quaternion norm validation failed.")
        if quaternion_dots.size and np.min(quaternion_dots) < 0.0:
            raise RuntimeError("Quaternion sign continuity validation failed.")

        joint_ranges = np.asarray(
            [
                self.model.jnt_range[self.model.joint(name).id]
                for name in self.joint_names
            ]
        )
        joint_positions = self.low_q[7:]
        lower_violation = np.maximum(joint_ranges[:, :1] - joint_positions, 0.0)
        upper_violation = np.maximum(joint_positions - joint_ranges[:, 1:], 0.0)
        joint_limit_max_violation = float(
            max(np.max(lower_violation), np.max(upper_violation))
        )

        ee_force_peak = float(np.max(np.abs(self.ee_force)))
        if ee_force_peak <= 1.0e-6:
            raise RuntimeError("Foot wrench data is all zero.")

        torque_limit_by_name = {
            name: limit
            for name, limit in zip(
                self.simulator.profile.joint_names,
                self.simulator.profile.torque_limits,
            )
        }
        torque_limits = np.asarray(
            [torque_limit_by_name[name] for name in self.joint_names]
        )
        torque_saturated = np.abs(self.tau) >= torque_limits[:, None] - 1.0e-6
        return {
            "samples": int(next(iter(columns))),
            "first_base_state_initialized": True,
            "quaternion_norm_max_error": float(np.max(np.abs(quaternion_norms - 1.0))),
            "contact_values": [int(value) for value in np.unique(self.contact)],
            "mean_total_vertical_force_n": float(
                np.mean(self.ee_force[2] + self.ee_force[8])
            ),
            "foot_wrench_peak_abs": ee_force_peak,
            "mean_imu_accel_z_mps2": float(np.mean(self.ddq[2])),
            "mean_abs_joint_velocity_radps": float(np.mean(np.abs(self.dq[6:]))),
            "mean_abs_joint_acceleration_radps2": float(
                np.mean(np.abs(self.ddq[6:]))
            ),
            "max_base_height_step_m": float(
                np.max(np.abs(np.diff(self.low_q[2]))) if self.low_q.shape[1] > 1 else 0.0
            ),
            "joint_limit_max_violation_rad": joint_limit_max_violation,
            "joint_limits_passed": joint_limit_max_violation <= 1.0e-5,
            "torque_saturation_fraction": float(np.mean(torque_saturated)),
        }


def run_substeps(
    simulator: HumanoidUltraSim2Sim,
    steps: int,
    recorder: IdentificationRecorder | None = None,
    command_profile=None,
    viewer=None,
) -> int:
    recorded = 0
    wall_start = time.perf_counter()
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        timestamp = step * SAMPLE_DT
        if step % POLICY_INTERVAL == 0:
            if command_profile is not None:
                simulator.command[:] = command_profile(timestamp)
            simulator.update_policy()
        if step % PD_INTERVAL == 0:
            simulator.prepare_physics_step()
        else:
            simulator.elastic_band.apply(simulator.data)

        mujoco.mj_forward(simulator.model, simulator.data)
        if recorder is not None:
            recorder.capture(recorded, timestamp, simulator.command.copy())
            recorded += 1
        mujoco.mj_step(simulator.model, simulator.data)
        simulator.clip_joint_velocities()

        if viewer is not None and (step + 1) % POLICY_INTERVAL == 0:
            viewer.sync()
            deadline = wall_start + (step + 1) * SAMPLE_DT
            sleep_time = deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    return recorded


def write_dat_files(
    recorder: IdentificationRecorder, output_dir: Path, robot_name: str
) -> dict[str, list[int]]:
    files = {
        f"{robot_name}_robot_low_q.dat": recorder.low_q,
        f"{robot_name}_robot_dq.dat": recorder.dq,
        f"{robot_name}_robot_ddq.dat": recorder.ddq,
        f"{robot_name}_robot_tau.dat": recorder.tau,
        f"{robot_name}_robot_contact.dat": recorder.contact,
        f"{robot_name}_robot_ee_force.dat": recorder.ee_force,
    }
    shapes = {}
    for filename, matrix in files.items():
        fmt = "%d" if matrix.dtype == np.int8 else "%.9f"
        np.savetxt(output_dir / filename, matrix, delimiter="\t", fmt=fmt)
        shapes[filename] = list(matrix.shape)
    return shapes


def write_joint_mapping(
    output_dir: Path,
    robot_name: str,
    simulator: HumanoidUltraSim2Sim,
    joint_names: tuple[str, ...],
) -> None:
    path = output_dir / f"{robot_name}_joint_mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("mujoco_actuator_id", "urdf_joint_name", "q_row", "dq_row", "ddq_row", "tau_row")
        )
        for index, name in enumerate(joint_names):
            writer.writerow(
                (
                    simulator.model.actuator(name).id,
                    name,
                    8 + index,
                    7 + index,
                    7 + index,
                    1 + index,
                )
            )


def write_metadata(
    output_dir: Path,
    robot_name: str,
    simulator: HumanoidUltraSim2Sim,
    args: argparse.Namespace,
    joint_names: tuple[str, ...],
    shapes: dict[str, list[int]],
    validation: dict,
    recorded_samples: int,
) -> None:
    base_body_id = simulator.model.body("dummy_link").id
    imu_site_name = IMU_SITE_NAMES[args.dof]
    imu_site_id = simulator.model.site(imu_site_name).id
    rotation_world_base = simulator.data.xmat[base_body_id].reshape(3, 3)
    rotation_world_imu = simulator.data.site_xmat[imu_site_id].reshape(3, 3)
    rotation_base_imu = rotation_world_base.T @ rotation_world_imu
    translation_base_imu = rotation_world_base.T @ (
        simulator.data.site_xpos[imu_site_id] - simulator.data.xpos[base_body_id]
    )

    metadata = {
        "robot_name": robot_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "policy_path": str(args.policy.resolve()),
        "mujoco_model_path": str(simulator.model_path),
        "dof": args.dof,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_dt_s": SAMPLE_DT,
        "recorded_samples": recorded_samples,
        "recorded_duration_s": recorded_samples * SAMPLE_DT,
        "settle_seconds": args.settle_seconds,
        "policy_rate_hz": POLICY_RATE_HZ,
        "pd_rate_hz": PD_RATE_HZ,
        "mujoco_integration_rate_hz": SAMPLE_RATE_HZ,
        "command_profile": args.profile,
        "constant_command": [args.vx, args.vy, args.yaw_rate],
        "joint_order": list(joint_names),
        "joint_encoder_zero_offsets_rad": {name: 0.0 for name in joint_names},
        "base_frame": "dummy_link",
        "imu_site": imu_site_name,
        "left_foot_frame": "left_foot",
        "right_foot_frame": "right_foot",
        "world_frame": {
            "handedness": "right-handed",
            "x_axis": "forward",
            "y_axis": "left",
            "z_axis": "up",
        },
        "raw_csv_time": {
            "timestamp_s": "sample index divided by 1000 Hz; starts at zero",
            "sim_time_s": "MuJoCo data.time; includes the unrecorded settling interval",
            "clock_source": "MuJoCo simulation clock",
        },
        "sensor_extrinsics": {
            "imu_to_base": {
                "transform_convention": "p_base = R_base_imu * p_imu + t_base_imu",
                "translation_m": translation_base_imu.tolist(),
                "rotation_row_major": rotation_base_imu.tolist(),
            },
            "left_foot_sensor_to_endpoint": {
                "transform": "identity",
                "note": "Wrench is synthesized from MuJoCo contacts at left_foot.",
            },
            "right_foot_sensor_to_endpoint": {
                "transform": "identity",
                "note": "Wrench is synthesized from MuJoCo contacts at right_foot.",
            },
        },
        "units": {
            "position": "m",
            "quaternion": "unitless",
            "joint_position": "rad",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
            "linear_acceleration": "m/s^2",
            "angular_acceleration": "rad/s^2",
            "joint_torque": "Nm",
            "force": "N",
            "moment": "Nm",
            "timestamp": "s",
        },
        "low_q": {
            "base_position_frame": "world",
            "quaternion_order": "x,y,z,w",
            "quaternion_meaning": "body_to_world",
        },
        "dq": {
            "base_linear_velocity_frame": "body",
            "base_angular_velocity_frame": "body",
        },
        "ddq": {
            "imu_linear_acceleration_frame": "body",
            "includes_gravity": True,
            "static_upright_z_mps2": GRAVITY,
            "angular_acceleration_frame": "body",
            "source": "MuJoCo acceleration-stage ground truth; no numerical differentiation",
        },
        "tau": {
            "unit": "Nm",
            "source": "MuJoCo qfrc_actuator in URDF joint positive direction",
            "controller": "clipped explicit PD torque",
        },
        "contact": {"contact": 1, "no_contact": 2, "threshold_n": args.contact_threshold},
        "ee_force": {
            "frame": "world-aligned",
            "reference_points": ["left_foot", "right_foot"],
            "direction": "environment_on_robot",
        },
        "elastic_band": {
            "enabled": args.elastic_band,
            "support_ratio": args.band_support_ratio if args.elastic_band else 0.0,
            "warning": "External shoulder force is not included in ee_force.dat.",
        },
        "model_total_mass_kg": float(np.sum(simulator.model.body_mass)),
        "model_mass_source": "sum of MuJoCo model body masses",
        "files": shapes,
        "raw_csv": f"{robot_name}_raw.csv",
        "validation": validation,
    }
    path = output_dir / f"{robot_name}_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0 or args.settle_seconds < 0.0:
        raise ValueError("Duration must be positive and settle time non-negative.")
    if args.profile == "identification" and args.duration <= 2.0 * args.static_seconds:
        raise ValueError("Identification duration must exceed twice --static-seconds.")
    if args.contact_threshold < 0.0:
        raise ValueError("--contact-threshold must be non-negative.")

    robot_name = args.robot_name or f"humanoid_ultra_{args.dof}dof"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", robot_name):
        raise ValueError("--robot-name may contain only letters, digits, underscores, and hyphens.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("identification_data") / f"{robot_name}_{timestamp}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    initial_command = np.asarray([args.vx, args.vy, args.yaw_rate], dtype=np.float64)
    simulator = HumanoidUltraSim2Sim(
        dof=args.dof,
        policy_path=args.policy.resolve(),
        command=np.zeros(3, dtype=np.float64),
        elastic_band_enabled=args.elastic_band,
        band_lift=args.band_lift,
        band_anchor_height=args.band_anchor_height,
        band_stiffness=args.band_stiffness,
        band_damping=args.band_damping,
        band_support_ratio=args.band_support_ratio,
    )
    simulator.model.opt.timestep = SAMPLE_DT
    simulator.reset()
    joint_names = URDF_12DOF_JOINTS if args.dof == 12 else URDF_27DOF_JOINTS
    output_dir.mkdir(parents=True)

    settle_steps = round(args.settle_seconds * SAMPLE_RATE_HZ)
    print(f"Settling for {args.settle_seconds:.3f}s without recording...")
    run_substeps(simulator, settle_steps)
    simulator.command[:] = initial_command
    simulator.observation_history.clear()

    sample_count = round(args.duration * SAMPLE_RATE_HZ)
    csv_path = output_dir / f"{robot_name}_raw.csv"
    recorder = IdentificationRecorder(
        simulator, joint_names, sample_count, csv_path, args.contact_threshold
    )

    if args.profile == "identification":
        command_profile = lambda t: identification_command(
            t, args.duration, args.static_seconds
        )
    else:
        command_profile = None

    def run(viewer=None) -> int:
        return run_substeps(
            simulator,
            sample_count,
            recorder=recorder,
            command_profile=command_profile,
            viewer=viewer,
        )

    try:
        if args.headless:
            recorded_samples = run()
        else:
            import mujoco.viewer

            def key_callback(keycode: int) -> None:
                if args.profile != "constant":
                    return
                if keycode in (ord("X"), ord("x"), 32):
                    simulator.command[:] = 0.0
                elif keycode in (ord("W"), ord("w"), 265):
                    simulator.command[0] = min(1.0, simulator.command[0] + 0.1)
                elif keycode in (ord("S"), ord("s"), 264):
                    simulator.command[0] = max(-0.6, simulator.command[0] - 0.1)
                elif keycode in (ord("A"), ord("a")):
                    simulator.command[1] = min(0.5, simulator.command[1] + 0.1)
                elif keycode in (ord("D"), ord("d")):
                    simulator.command[1] = max(-0.5, simulator.command[1] - 0.1)
                elif keycode in (ord("Q"), ord("q"), 263):
                    simulator.command[2] = min(1.57, simulator.command[2] + 0.1)
                elif keycode in (ord("E"), ord("e"), 262):
                    simulator.command[2] = max(-1.57, simulator.command[2] - 0.1)
                else:
                    return
                print(
                    f"command vx={simulator.command[0]:.2f}, "
                    f"vy={simulator.command[1]:.2f}, yaw={simulator.command[2]:.2f}"
                )

            print("Click the MuJoCo window. Constant profile supports W/S, A/D, Q/E, X/Space.")
            with mujoco.viewer.launch_passive(
                simulator.model, simulator.data, key_callback=key_callback
            ) as viewer:
                recorded_samples = run(viewer)
    finally:
        recorder.close()

    if recorded_samples == 0:
        raise RuntimeError("No samples were recorded.")
    recorder.trim(recorded_samples)
    validation = recorder.validate()
    shapes = write_dat_files(recorder, output_dir, robot_name)
    write_joint_mapping(output_dir, robot_name, simulator, joint_names)
    write_metadata(
        output_dir,
        robot_name,
        simulator,
        args,
        joint_names,
        shapes,
        validation,
        recorded_samples,
    )
    print(
        f"Recorded {recorded_samples} aligned samples at {SAMPLE_RATE_HZ} Hz "
        f"({recorded_samples / SAMPLE_RATE_HZ:.3f}s)."
    )
    print(f"Output: {output_dir}")
    print(
        f"Mean left+right Fz: {validation['mean_total_vertical_force_n']:.3f} N; "
        f"model weight: {np.sum(simulator.model.body_mass) * GRAVITY:.3f} N"
    )
    if not validation["joint_limits_passed"]:
        print(
            "Warning: maximum joint-limit violation is "
            f"{validation['joint_limit_max_violation_rad']:.6f} rad."
        )


if __name__ == "__main__":
    main()
