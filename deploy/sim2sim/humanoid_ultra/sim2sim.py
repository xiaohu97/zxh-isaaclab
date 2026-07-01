#!/usr/bin/env python3
"""Run a Humanoid Ultra Isaac Lab policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch


ISAAC_12DOF_JOINTS = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

ISAAC_27DOF_JOINTS = (
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

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
LEFT_ARM_COMMAND_DIM = 2 * len(LEFT_ARM_JOINTS) + 1

URDF_12DOF_JOINTS = (
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
)

URDF_27DOF_JOINTS = URDF_12DOF_JOINTS + (
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

TRAINING_POSITION_LIMITS = {
    "left_hip_roll_joint": (-0.25, 1.5708),
    "right_hip_roll_joint": (-1.5708, 0.25),
    "left_hip_yaw_joint": (-1.5708, 1.5708),
    "right_hip_yaw_joint": (-1.5708, 1.5708),
    "left_hip_pitch_joint": (-1.5708, 1.5708),
    "right_hip_pitch_joint": (-1.5708, 1.5708),
    "left_knee_joint": (0.0, 2.356),
    "right_knee_joint": (0.0, 2.356),
    "left_ankle_pitch_joint": (-0.7, 0.95),
    "right_ankle_pitch_joint": (-0.7, 0.95),
    "left_ankle_roll_joint": (-0.5236, 0.5236),
    "right_ankle_roll_joint": (-0.5236, 0.5236),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-2.4, 1.2),
    "right_shoulder_pitch_joint": (-1.2, 2.4),
    "left_shoulder_roll_joint": (-0.3, 2.7),
    "right_shoulder_roll_joint": (-2.7, 0.3),
    "left_shoulder_yaw_joint": (-2.5, 2.5),
    "right_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-2.17, 0.0),
    "right_elbow_joint": (0.0, 2.17),
    "left_wrist_yaw_joint": (-2.5, 2.5),
    "right_wrist_yaw_joint": (-2.5, 2.5),
    "left_wrist_roll_joint": (-1.11, 1.11),
    "right_wrist_roll_joint": (-1.11, 1.11),
    "left_wrist_pitch_joint": (-1.05, 1.05),
    "right_wrist_pitch_joint": (-1.05, 1.05),
}


@dataclass(frozen=True)
class RobotProfile:
    dof: int
    root_height: float
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    torque_limits: np.ndarray
    velocity_limits: np.ndarray
    position_limits: np.ndarray

    @property
    def observation_dim(self) -> int:
        return 9 + 3 * self.dof


class GamepadCommandSource:
    """Poll an SDL-mapped gamepad without blocking the MuJoCo control loop."""

    AXIS_MAX = 32768.0
    RECONNECT_PERIOD = 1.0

    def __init__(self, index: int, deadzone: float):
        try:
            import pygame
            from pygame._sdl2 import controller
        except ImportError as exc:
            raise RuntimeError(
                "Gamepad control requires pygame. Install it with `pip install pygame`."
            ) from exc

        self._pygame = pygame
        self._controller_module = controller
        self.index = index
        self.deadzone = deadzone
        self._controller = None
        self._next_connect_attempt = 0.0
        self._waiting_reported = False
        self._previous_y = False
        self._previous_x = False
        self._previous_stop = False
        self.left_arm_toggle_requested = False
        self.enabled = True
        self.connected = False

        self._controller_module.init()
        self._try_connect()

    def _try_connect(self) -> None:
        now = time.monotonic()
        if now < self._next_connect_attempt:
            return
        self._next_connect_attempt = now + self.RECONNECT_PERIOD

        try:
            self._controller_module.update()
            if (
                self.index >= self._controller_module.get_count()
                or not self._controller_module.is_controller(self.index)
            ):
                if not self._waiting_reported:
                    print(
                        f"Gamepad {self.index} is unavailable or has no SDL mapping; "
                        "waiting for a controller."
                    )
                    self._waiting_reported = True
                return
            self._controller = self._controller_module.Controller(self.index)
        except (self._pygame.error, self._controller_module.error) as exc:
            if not self._waiting_reported:
                print(f"Cannot open gamepad {self.index}: {exc}; waiting for a controller.")
                self._waiting_reported = True
            return

        self.connected = True
        self._waiting_reported = False
        self._previous_y = False
        self._previous_x = False
        self._previous_stop = False
        self.left_arm_toggle_requested = False
        state = "enabled" if self.enabled else "disabled; press Y to enable"
        print(f"Gamepad connected: {self._controller.name} ({state}).")

    def _disconnect(self) -> None:
        if self._controller is not None:
            try:
                self._controller.quit()
            except self._pygame.error:
                pass
        self._controller = None
        self.connected = False
        self.enabled = False
        self._previous_y = False
        self._previous_x = False
        self._previous_stop = False
        self.left_arm_toggle_requested = False
        self._next_connect_attempt = 0.0
        if not self._waiting_reported:
            print("Gamepad disconnected: command cleared; reconnect and press Y to enable.")
            self._waiting_reported = True

    def _axis(self, axis: int) -> float:
        assert self._controller is not None
        value = float(np.clip(self._controller.get_axis(axis) / self.AXIS_MAX, -1.0, 1.0))
        magnitude = abs(value)
        if magnitude <= self.deadzone:
            return 0.0
        return float(np.copysign((magnitude - self.deadzone) / (1.0 - self.deadzone), value))

    def poll(self, mode: str) -> np.ndarray:
        self.left_arm_toggle_requested = False
        zero_command = np.zeros(3, dtype=np.float64)
        if self._controller is None:
            self._try_connect()
            if self._controller is None:
                return zero_command

        try:
            self._controller_module.update()
            if not self._controller.attached():
                self._disconnect()
                return zero_command

            y_pressed = bool(self._controller.get_button(self._pygame.CONTROLLER_BUTTON_Y))
            x_pressed = bool(self._controller.get_button(self._pygame.CONTROLLER_BUTTON_X))
            stop_pressed = bool(
                self._controller.get_button(self._pygame.CONTROLLER_BUTTON_LEFTSHOULDER)
                and self._controller.get_button(self._pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)
            )
            if stop_pressed and not self._previous_stop:
                self.enabled = False
                print("Gamepad LB+RB: command cleared and gamepad control disabled.")
            elif y_pressed and not self._previous_y:
                self.enabled = not self.enabled
                print(f"Gamepad control: {'ON' if self.enabled else 'OFF (command cleared)'}.")
            if x_pressed and not self._previous_x and not stop_pressed:
                self.left_arm_toggle_requested = True
            self._previous_y = y_pressed
            self._previous_x = x_pressed
            self._previous_stop = stop_pressed

            if not self.enabled:
                return zero_command

            left_x = self._axis(self._pygame.CONTROLLER_AXIS_LEFTX)
            left_y = self._axis(self._pygame.CONTROLLER_AXIS_LEFTY)
            right_x = self._axis(self._pygame.CONTROLLER_AXIS_RIGHTX)
        except (self._pygame.error, self._controller_module.error):
            self._disconnect()
            return zero_command

        if mode == "stand":
            return np.asarray((-left_y, -0.5 * left_x, 0.5 * right_x), dtype=np.float64)

        forward = -left_y
        vx = forward * (1.0 if forward >= 0.0 else 0.6)
        return np.asarray((vx, -0.5 * left_x, 1.57 * right_x), dtype=np.float64)

    def close(self) -> None:
        if self._controller is not None:
            self._controller.quit()
            self._controller = None
        self._controller_module.quit()


def _gain_for_joint(name: str) -> tuple[float, float, float]:
    if "hip_roll" in name:
        return 150.0, 2.5, 300.0
    if "hip_yaw" in name:
        return 80.0, 0.8, 90.0
    if "hip_pitch" in name or "knee" in name:
        return 180.0, 2.4, 300.0
    if "ankle_pitch" in name:
        return 40.0, 0.8, 27.0
    if "ankle_roll" in name:
        return 20.0, 0.4, 27.0
    if "waist" in name:
        return 150.0, 2.5, 150.0
    if "shoulder" in name:
        return 80.0, 1.5, 60.0
    if "elbow" in name:
        return 60.0, 1.2, 60.0
    if "wrist" in name:
        return 25.0, 0.8, 24.0
    raise ValueError(f"No actuator gains configured for joint: {name}")


def _velocity_limit_for_joint(name: str) -> float:
    if "ankle" in name:
        return 12.0
    if "waist" in name:
        return 12.56
    if any(part in name for part in ("shoulder", "elbow", "wrist")):
        return 10.0
    return 15.0


def make_profile(dof: int) -> RobotProfile:
    if dof == 12:
        names = ISAAC_12DOF_JOINTS
        root_height = 0.995
        leg_pitch = 0.346431
        knee = 0.755514
        ankle_pitch = 0.366252
    elif dof == 27:
        names = ISAAC_27DOF_JOINTS
        root_height = 1.005
        leg_pitch = 0.289936
        knee = 0.742326
        ankle_pitch = 0.409573
    else:
        raise ValueError(f"Unsupported number of joints: {dof}")

    default_by_name = {
        "left_hip_roll_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "left_hip_pitch_joint": leg_pitch,
        "right_hip_pitch_joint": leg_pitch,
        "left_knee_joint": knee,
        "right_knee_joint": knee,
        "left_ankle_pitch_joint": ankle_pitch,
        "right_ankle_pitch_joint": ankle_pitch,
        "left_ankle_roll_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.25,
        "right_shoulder_pitch_joint": -0.25,
        "left_shoulder_roll_joint": -0.05,
        "right_shoulder_roll_joint": 0.05,
        "left_shoulder_yaw_joint": -1.5707963,
        "right_shoulder_yaw_joint": 1.5707963,
        "left_elbow_joint": -0.6,
        "right_elbow_joint": 0.6,
        "left_wrist_yaw_joint": 1.5707963,
        "right_wrist_yaw_joint": -1.5707963,
        "left_wrist_roll_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
    }
    gains = [_gain_for_joint(name) for name in names]
    stiffness, damping, torque_limits = zip(*gains)
    return RobotProfile(
        dof=dof,
        root_height=root_height,
        joint_names=tuple(names),
        default_joint_pos=np.asarray([default_by_name[name] for name in names], dtype=np.float64),
        stiffness=np.asarray(stiffness, dtype=np.float64),
        damping=np.asarray(damping, dtype=np.float64),
        torque_limits=np.asarray(torque_limits, dtype=np.float64),
        velocity_limits=np.asarray(
            [_velocity_limit_for_joint(name) for name in names], dtype=np.float64
        ),
        position_limits=np.asarray(
            [TRAINING_POSITION_LIMITS[name] for name in names], dtype=np.float64
        ),
    )


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def policy_input_dim(policy: torch.jit.ScriptModule) -> int:
    """Read the actor input width from the exported TorchScript policy."""
    state_dict = policy.state_dict()
    first_layer = state_dict.get("actor.0.weight")
    if first_layer is None or first_layer.ndim != 2:
        raise RuntimeError(
            "Cannot determine policy input size: actor.0.weight is missing from policy.pt."
        )
    return int(first_layer.shape[1])


class LeftArmTrajectory:
    """Reproduce the 15-D left-arm command used by the Isaac Lab task."""

    def __init__(
        self,
        traj_path: Path,
        profile: RobotProfile,
        enabled: bool,
        period: float = 6.0,
        blend_time: float = 2.0,
        ref_vel_scale: float = 0.25,
    ):
        if profile.dof != 27:
            raise ValueError("Left-arm trajectory policies require the 27-DOF robot.")
        if not traj_path.is_file():
            raise FileNotFoundError(f"Left-arm trajectory CSV not found: {traj_path}")

        with traj_path.open("r", encoding="utf-8") as file:
            header = file.readline().strip().split(",")
        raw = np.loadtxt(traj_path, delimiter=",", skiprows=1)
        column_indices = [header.index(name) for name in LEFT_ARM_JOINTS]
        samples = raw[:, column_indices]

        sample_count = samples.shape[0]
        coefficients = np.fft.rfft(samples, axis=0)
        self.cos_coeff = (2.0 / sample_count) * coefficients.real
        self.sin_coeff = (-2.0 / sample_count) * coefficients.imag
        self.cos_coeff[0] *= 0.5
        if sample_count % 2 == 0:
            self.cos_coeff[-1] *= 0.5
            self.sin_coeff[-1] = 0.0

        harmonics = np.arange(coefficients.shape[0], dtype=np.float64)
        self.omega = 2.0 * np.pi * harmonics / period
        joint_index = {name: index for index, name in enumerate(profile.joint_names)}
        self.default_q = np.asarray(
            [profile.default_joint_pos[joint_index[name]] for name in LEFT_ARM_JOINTS],
            dtype=np.float64,
        )
        self.period = period
        self.blend_time = blend_time
        self.ref_vel_scale = ref_vel_scale
        self.default_enabled = enabled
        self.enabled = enabled
        self.elapsed = 0.0

    def reset(self) -> None:
        self.enabled = self.default_enabled
        self.elapsed = 0.0

    def toggle(self) -> None:
        self.enabled = not self.enabled
        if self.enabled:
            self.elapsed = 0.0

    def advance(self, dt: float) -> None:
        self.elapsed += dt

    def observation(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(LEFT_ARM_COMMAND_DIM, dtype=np.float32)

        phase = self.elapsed % self.period
        angles = phase * self.omega
        cosines = np.cos(angles)
        sines = np.sin(angles)
        q_traj = cosines @ self.cos_coeff + sines @ self.sin_coeff
        dq_traj = (
            -(sines * self.omega) @ self.cos_coeff
            + (cosines * self.omega) @ self.sin_coeff
        )
        delta = q_traj - self.default_q

        if self.blend_time > 0.0:
            u = np.clip(self.elapsed / self.blend_time, 0.0, 1.0)
            alpha = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
            alpha_dot = (
                30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2
            ) / self.blend_time
        else:
            alpha = 1.0
            alpha_dot = 0.0

        q_ref_rel = alpha * delta
        dq_ref = alpha_dot * delta + alpha * dq_traj
        return np.concatenate(
            (q_ref_rel, dq_ref * self.ref_vel_scale, np.ones(1))
        ).astype(np.float32)


class ElasticBand:
    """Apply two spring-damper suspension forces at the shoulder sockets."""

    GRAVITY = 9.81
    SHOULDER_BODY_NAMES = (
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        suspension_height: float,
        anchor_height: float,
        stiffness: float,
        damping: float,
        support_ratio: float,
        enabled: bool,
    ):
        if anchor_height <= suspension_height:
            raise ValueError("--band-anchor-height must be above the suspended robot height.")
        if stiffness <= 0.0 or damping < 0.0:
            raise ValueError("Elastic-band stiffness must be positive and damping non-negative.")
        if not 0.0 <= support_ratio <= 1.0:
            raise ValueError("--band-support-ratio must be between 0.0 and 1.0.")

        self.model = model
        self.trunk_body_id = model.body("trunk_link").id
        self.shoulder_body_ids = tuple(
            model.body(name).id for name in self.SHOULDER_BODY_NAMES
        )
        self.anchor_height = anchor_height
        self.points = np.zeros((2, 3), dtype=np.float64)
        self.stiffness = stiffness
        self.damping = damping
        self.support_ratio = support_ratio
        self.enabled = enabled
        self.suspension_height = suspension_height
        self.robot_weight = float(np.sum(model.body_mass)) * self.GRAVITY
        self.length = 0.0
        self.max_force_per_band = 1.25 * self.robot_weight
        self._body_velocity = np.zeros(6, dtype=np.float64)
        self._zero_torque = np.zeros(3, dtype=np.float64)

    def reset_anchors(self, data: mujoco.MjData) -> None:
        shoulder_positions = np.asarray(
            [data.xpos[body_id].copy() for body_id in self.shoulder_body_ids]
        )
        self.points[:, :2] = shoulder_positions[:, :2]
        self.points[:, 2] = self.anchor_height
        anchor_distance = float(self.anchor_height - np.mean(shoulder_positions[:, 2]))
        force_per_band = 0.5 * self.robot_weight * self.support_ratio
        self.length = max(0.0, anchor_distance - force_per_band / self.stiffness)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def adjust_length(self, delta: float) -> None:
        self.length = max(0.0, self.length + delta)

    def apply(self, data: mujoco.MjData) -> None:
        data.qfrc_applied[:] = 0.0
        if not self.enabled:
            return

        for anchor, shoulder_body_id in zip(self.points, self.shoulder_body_ids):
            shoulder_position = data.xpos[shoulder_body_id]
            displacement = anchor - shoulder_position
            distance = float(np.linalg.norm(displacement))
            if distance < 1.0e-9:
                continue

            direction = displacement / distance
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                shoulder_body_id,
                self._body_velocity,
                0,
            )
            velocity_along_band = float(np.dot(self._body_velocity[3:], direction))
            extension = max(0.0, distance - self.length)
            force_magnitude = self.stiffness * extension - self.damping * velocity_along_band
            force_magnitude = float(np.clip(force_magnitude, 0.0, self.max_force_per_band))
            mujoco.mj_applyFT(
                self.model,
                data,
                force_magnitude * direction,
                self._zero_torque,
                shoulder_position,
                self.trunk_body_id,
                data.qfrc_applied,
            )


class HumanoidUltraSim2Sim:
    SIM_DT = 0.005
    CONTROL_DECIMATION = 4
    ACTION_SCALE = 0.25
    HISTORY_LENGTH = 10

    def __init__(
        self,
        dof: int,
        policy_path: Path,
        command: np.ndarray,
        left_arm_traj_path: Path | None,
        left_arm_enabled: bool,
        elastic_band_enabled: bool,
        band_lift: float,
        band_anchor_height: float,
        band_stiffness: float,
        band_damping: float,
        band_support_ratio: float,
    ):
        self.profile = make_profile(dof)
        repository_root = Path(__file__).resolve().parents[2]
        model_path = (
            repository_root
            / "unitree_robots"
            / "humanoid_ultra"
            / f"scene_{self.profile.dof}dof.xml"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {model_path}")
        if not policy_path.is_file():
            raise FileNotFoundError(f"Exported TorchScript policy not found: {policy_path}")

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.SIM_DT
        self.elastic_band_default_enabled = elastic_band_enabled
        self.elastic_band = ElasticBand(
            model=self.model,
            suspension_height=self.profile.root_height + band_lift,
            anchor_height=band_anchor_height,
            stiffness=band_stiffness,
            damping=band_damping,
            support_ratio=band_support_ratio,
            enabled=elastic_band_enabled,
        )

        actuator_names = tuple(self.model.actuator(index).name for index in range(self.model.nu))
        if set(actuator_names) != set(self.profile.joint_names):
            raise RuntimeError(
                "MuJoCo actuators do not match the Isaac Lab policy joints.\n"
                f"Expected: {sorted(self.profile.joint_names)}\nActual:   {sorted(actuator_names)}"
            )

        self.actuator_indices = np.asarray(
            [self.model.actuator(name).id for name in self.profile.joint_names], dtype=np.int32
        )
        self.qpos_indices = np.asarray(
            [self.model.joint(name).qposadr[0] for name in self.profile.joint_names], dtype=np.int32
        )
        self.qvel_indices = np.asarray(
            [self.model.joint(name).dofadr[0] for name in self.profile.joint_names], dtype=np.int32
        )
        for name, limits in zip(self.profile.joint_names, self.profile.position_limits):
            self.model.jnt_range[self.model.joint(name).id] = limits
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        input_dim = policy_input_dim(self.policy)
        if input_dim % self.HISTORY_LENGTH != 0:
            raise RuntimeError(
                f"Policy input size {input_dim} is not divisible by history length "
                f"{self.HISTORY_LENGTH}."
            )
        frame_observation_dim = input_dim // self.HISTORY_LENGTH
        if frame_observation_dim == self.profile.observation_dim:
            self.left_arm_trajectory = None
        elif frame_observation_dim == self.profile.observation_dim + LEFT_ARM_COMMAND_DIM:
            trajectory_path = left_arm_traj_path or Path(__file__).with_name(
                "left_wrist_pitch_traj.csv"
            )
            self.left_arm_trajectory = LeftArmTrajectory(
                trajectory_path.resolve(), self.profile, left_arm_enabled
            )
        else:
            raise RuntimeError(
                "Unsupported policy observation size: "
                f"{frame_observation_dim} per frame ({input_dim} total). Expected "
                f"{self.profile.observation_dim} for standard stand/locomotion or "
                f"{self.profile.observation_dim + LEFT_ARM_COMMAND_DIM} for stand-leftarm."
            )
        self.command = command.astype(np.float64)
        self.previous_action = np.zeros(self.profile.dof, dtype=np.float64)
        self.target_joint_pos = self.profile.default_joint_pos.copy()
        self.observation_history: deque[np.ndarray] = deque(maxlen=self.HISTORY_LENGTH)

        with torch.inference_mode():
            test_output = self.policy(torch.zeros(1, input_dim, dtype=torch.float32))
        if not isinstance(test_output, torch.Tensor) or tuple(test_output.shape) != (1, self.profile.dof):
            raise RuntimeError(
                f"Policy shape mismatch: expected [1, {self.profile.dof}], got "
                f"{getattr(test_output, 'shape', type(test_output))}"
            )

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.elastic_band.set_enabled(self.elastic_band_default_enabled)
        root_height = (
            self.elastic_band.suspension_height
            if self.elastic_band.enabled
            else self.profile.root_height
        )
        self.data.qpos[:3] = (0.0, 0.0, root_height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos_indices] = self.profile.default_joint_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.previous_action.fill(0.0)
        self.target_joint_pos[:] = self.profile.default_joint_pos
        self.observation_history.clear()
        if self.left_arm_trajectory is not None:
            self.left_arm_trajectory.reset()
        mujoco.mj_forward(self.model, self.data)
        self.elastic_band.reset_anchors(self.data)

    def _current_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.qpos_indices]
        joint_vel = self.data.qvel[self.qvel_indices]
        body_angular_velocity = self.data.sensor("BodyGyro").data.copy()
        body_rotation = quaternion_to_rotation_matrix(self.data.qpos[3:7])
        projected_gravity = body_rotation.T @ np.asarray([0.0, 0.0, -1.0])
        observation_parts = [
            body_angular_velocity,
            projected_gravity,
            self.command,
            joint_pos - self.profile.default_joint_pos,
            joint_vel,
            self.previous_action,
        ]
        if self.left_arm_trajectory is not None:
            observation_parts.append(self.left_arm_trajectory.observation())
        observation = np.concatenate(observation_parts)
        return np.clip(observation, -100.0, 100.0).astype(np.float32)

    def update_policy(self) -> None:
        observation = self._current_observation()
        if not self.observation_history:
            for _ in range(self.HISTORY_LENGTH):
                self.observation_history.append(observation.copy())
        else:
            self.observation_history.append(observation)
        policy_input = np.concatenate(tuple(self.observation_history))
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(policy_input).unsqueeze(0)).squeeze(0).cpu().numpy()
        self.previous_action = np.clip(action, -100.0, 100.0).astype(np.float64)
        self.target_joint_pos = self.profile.default_joint_pos + self.ACTION_SCALE * self.previous_action

    def prepare_physics_step(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.qpos_indices]
        joint_vel = self.data.qvel[self.qvel_indices]
        torque = (
            self.profile.stiffness * (self.target_joint_pos - joint_pos)
            - self.profile.damping * joint_vel
        )
        applied_torque = np.clip(
            torque, -self.profile.torque_limits, self.profile.torque_limits
        )
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_indices] = applied_torque
        self.elastic_band.apply(self.data)
        return applied_torque

    def clip_joint_velocities(self) -> None:
        self.data.qvel[self.qvel_indices] = np.clip(
            self.data.qvel[self.qvel_indices],
            -self.profile.velocity_limits,
            self.profile.velocity_limits,
        )

    def physics_step(self) -> None:
        self.prepare_physics_step()
        mujoco.mj_step(self.model, self.data)
        self.clip_joint_velocities()

    def stand(self, duration: float) -> None:
        steps = max(0, int(duration / self.SIM_DT))
        for _ in range(steps):
            self.physics_step()
        self.observation_history.clear()
        if self.left_arm_trajectory is not None:
            self.left_arm_trajectory.reset()

    def control_step(self) -> None:
        self.update_policy()
        for _ in range(self.CONTROL_DECIMATION):
            self.physics_step()
        if self.left_arm_trajectory is not None:
            self.left_arm_trajectory.advance(self.SIM_DT * self.CONTROL_DECIMATION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported JIT policy.pt from Isaac Lab.")
    parser.add_argument("--dof", type=int, choices=(12, 27), required=True)
    parser.add_argument(
        "--mode",
        choices=("locomotion", "stand"),
        default="locomotion",
        help="Policy command semantics and keyboard bindings.",
    )
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw velocity command in rad/s.")
    parser.add_argument(
        "--height-command",
        type=float,
        default=0.0,
        help="Stand mode: positive crouches, negative raises the base.",
    )
    parser.add_argument(
        "--roll-command",
        type=float,
        default=0.0,
        help="Stand mode torso roll command.",
    )
    parser.add_argument(
        "--pitch-command",
        type=float,
        default=0.0,
        help="Stand mode torso pitch command.",
    )
    parser.add_argument(
        "--left-arm-traj",
        type=Path,
        default=None,
        help=(
            "Trajectory CSV for a stand-leftarm policy. Defaults to "
            "left_wrist_pitch_traj.csv next to this script."
        ),
    )
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument(
        "--left-arm-track",
        dest="left_arm_enabled",
        action="store_true",
        help="Start a stand-leftarm policy with trajectory tracking enabled.",
    )
    arm_group.add_argument(
        "--no-left-arm-track",
        dest="left_arm_enabled",
        action="store_false",
        help="Start a stand-leftarm policy with its arm command disabled.",
    )
    parser.set_defaults(left_arm_enabled=True)
    parser.add_argument(
        "--stand-seconds",
        type=float,
        default=0.0,
        help="Fixed-pose warmup before policy control. Keep at zero for trained locomotion policies.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration; zero means until viewer closes.")
    band_group = parser.add_mutually_exclusive_group()
    band_group.add_argument(
        "--elastic-band",
        dest="elastic_band_enabled",
        action="store_true",
        help="Force-enable the shoulder elastic suspension.",
    )
    band_group.add_argument(
        "--no-elastic-band",
        dest="elastic_band_enabled",
        action="store_false",
        help="Force-disable the shoulder elastic suspension.",
    )
    parser.set_defaults(elastic_band_enabled=None)
    parser.add_argument(
        "--band-lift",
        type=float,
        default=0.0,
        help="Initial root lift above the normal standing height in meters.",
    )
    parser.add_argument("--band-anchor-height", type=float, default=3.0)
    parser.add_argument("--band-stiffness", type=float, default=500.0)
    parser.add_argument("--band-damping", type=float, default=100.0)
    parser.add_argument(
        "--band-support-ratio",
        type=float,
        default=0.3,
        help="Fraction of robot weight supported by the two shoulder bands.",
    )
    parser.add_argument(
        "--gamepad",
        action="store_true",
        help="Read policy commands from an SDL-mapped gamepad using pygame.",
    )
    parser.add_argument("--gamepad-index", type=int, default=0, help="SDL gamepad device index.")
    parser.add_argument(
        "--gamepad-deadzone",
        type=float,
        default=0.08,
        help="Normalized joystick deadzone in [0, 1).",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.gamepad_index < 0:
        parser.error("--gamepad-index must be non-negative")
    if not 0.0 <= args.gamepad_deadzone < 1.0:
        parser.error("--gamepad-deadzone must be in [0, 1)")
    if args.gamepad and args.headless:
        parser.error("--gamepad cannot be combined with --headless")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "stand":
        command = np.asarray(
            [args.height_command, args.roll_command, args.pitch_command],
            dtype=np.float64,
        )
        command = np.clip(command, (-1.0, -0.5, -0.5), (1.0, 0.5, 0.5))
    else:
        command = np.asarray([args.vx, args.vy, args.yaw_rate], dtype=np.float64)
        command = np.clip(command, (-0.6, -0.5, -1.57), (1.0, 0.5, 1.57))
    if args.gamepad:
        command.fill(0.0)
    elastic_band_enabled = args.elastic_band_enabled
    if elastic_band_enabled is None:
        elastic_band_enabled = args.mode == "locomotion"

    simulator = HumanoidUltraSim2Sim(
        dof=args.dof,
        policy_path=args.policy.resolve(),
        command=command,
        left_arm_traj_path=args.left_arm_traj,
        left_arm_enabled=args.left_arm_enabled,
        elastic_band_enabled=elastic_band_enabled,
        band_lift=args.band_lift,
        band_anchor_height=args.band_anchor_height,
        band_stiffness=args.band_stiffness,
        band_damping=args.band_damping,
        band_support_ratio=args.band_support_ratio,
    )
    simulator.stand(args.stand_seconds)
    gamepad = (
        GamepadCommandSource(args.gamepad_index, args.gamepad_deadzone)
        if args.gamepad
        else None
    )
    if simulator.left_arm_trajectory is not None:
        state = "ON" if simulator.left_arm_trajectory.enabled else "OFF"
        print(f"Left-arm trajectory: {state} (L toggles tracking).")
    if args.mode == "stand":
        print(
            f"Loaded {args.dof}-DOF stand policy. Command: "
            f"height={command[0]:.2f}, roll={command[1]:.2f}, pitch={command[2]:.2f}"
        )
    else:
        print(
            f"Loaded {args.dof}-DOF locomotion policy. Command: "
            f"vx={command[0]:.2f}, vy={command[1]:.2f}, yaw={command[2]:.2f}"
        )

    start_time = time.perf_counter()
    control_dt = simulator.SIM_DT * simulator.CONTROL_DECIMATION

    if args.headless:
        duration = args.duration if args.duration > 0.0 else 10.0
        while simulator.data.time < args.stand_seconds + duration:
            simulator.control_step()
        print(
            f"Finished headless rollout: t={simulator.data.time:.3f}s, "
            f"base_z={simulator.data.qpos[2]:.3f}m"
        )
        return

    import mujoco.viewer

    glfw_key_right = 262
    glfw_key_left = 263
    glfw_key_down = 264
    glfw_key_up = 265

    def print_status() -> None:
        band_status = "ON" if simulator.elastic_band.enabled else "OFF"
        arm_status = (
            "N/A"
            if simulator.left_arm_trajectory is None
            else "ON" if simulator.left_arm_trajectory.enabled else "OFF"
        )
        if args.mode == "stand":
            command_status = (
                f"height={simulator.command[0]:.2f}, "
                f"roll={simulator.command[1]:.2f}, pitch={simulator.command[2]:.2f}"
            )
        else:
            command_status = (
                f"vx={simulator.command[0]:.2f}, "
                f"vy={simulator.command[1]:.2f}, yaw={simulator.command[2]:.2f}"
            )
        if gamepad is None:
            gamepad_status = "OFF"
        elif not gamepad.connected:
            gamepad_status = "DISCONNECTED"
        else:
            gamepad_status = "ON" if gamepad.enabled else "DISABLED"
        print(
            f"command {command_status} | "
            f"band={band_status}, length={simulator.elastic_band.length:.2f}m, "
            f"left_arm={arm_status}, gamepad={gamepad_status}"
        )

    def toggle_left_arm() -> None:
        if simulator.left_arm_trajectory is None:
            print("This policy has no left-arm trajectory observation.")
        else:
            simulator.left_arm_trajectory.toggle()

    def key_callback(keycode: int) -> None:
        handled = True
        if keycode in (ord("R"), ord("r")):
            simulator.reset()
            simulator.stand(args.stand_seconds)
        elif keycode in (ord("X"), ord("x"), 32):
            simulator.command[:] = 0.0
        elif keycode in (ord("W"), ord("w"), glfw_key_up):
            simulator.command[0] = min(1.0, simulator.command[0] + 0.1)
        elif keycode in (ord("S"), ord("s"), glfw_key_down):
            lower_bound = -1.0 if args.mode == "stand" else -0.6
            simulator.command[0] = max(lower_bound, simulator.command[0] - 0.1)
        elif keycode in (ord("A"), ord("a")):
            simulator.command[1] = min(0.5, simulator.command[1] + 0.1)
        elif keycode in (ord("D"), ord("d")):
            simulator.command[1] = max(-0.5, simulator.command[1] - 0.1)
        elif keycode in (ord("Q"), ord("q"), glfw_key_left):
            upper_bound = 0.5 if args.mode == "stand" else 1.57
            simulator.command[2] = min(upper_bound, simulator.command[2] + 0.1)
        elif keycode in (ord("E"), ord("e"), glfw_key_right):
            lower_bound = -0.5 if args.mode == "stand" else -1.57
            simulator.command[2] = max(lower_bound, simulator.command[2] - 0.1)
        elif keycode in (ord("7"),):
            simulator.elastic_band.adjust_length(-0.1)
        elif keycode in (ord("8"),):
            simulator.elastic_band.adjust_length(0.1)
        elif keycode in (ord("9"), ord("B"), ord("b")):
            simulator.elastic_band.toggle()
        elif keycode in (ord("L"), ord("l")):
            toggle_left_arm()
        else:
            handled = False
        if handled:
            print_status()

    print("Click the MuJoCo window first so it can receive keyboard input.")
    if args.mode == "stand":
        print("Stand: W crouch, S raise, A/D roll, Q/E pitch.")
    else:
        print("Move: W/S or Up/Down, A/D lateral, Q/E or Left/Right yaw.")
    print("Band: 7 shorter/raise, 8 longer/lower, 9/B release or attach.")
    if simulator.left_arm_trajectory is not None:
        print("Left arm: L toggles trajectory tracking or returns to the default pose.")
    print("Other: X/Space stop, R reset and restore the default band state.")
    if gamepad is not None:
        if args.mode == "stand":
            print("Gamepad: left Y=height, left X=roll, right X=pitch.")
        else:
            print("Gamepad: left Y=vx, left X=vy, right X=yaw rate.")
        print("Gamepad buttons: Y toggles control; X toggles the left-arm trajectory.")
        print("Gamepad safety: LB+RB clears the command and disables control.")
    print_status()
    try:
        with mujoco.viewer.launch_passive(
            simulator.model, simulator.data, key_callback=key_callback
        ) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                if gamepad is not None:
                    simulator.command[:] = gamepad.poll(args.mode)
                    if gamepad.left_arm_toggle_requested:
                        toggle_left_arm()
                        print_status()
                simulator.control_step()
                viewer.sync()
                if args.duration > 0.0 and time.perf_counter() - start_time >= args.duration:
                    break
                sleep_time = control_dt - (time.perf_counter() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    main()
