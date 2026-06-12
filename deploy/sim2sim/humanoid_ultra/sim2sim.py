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


LEG_JOINTS = [
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
]

UPPER_BODY_JOINTS = [
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
]


@dataclass(frozen=True)
class RobotProfile:
    dof: int
    root_height: float
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    torque_limits: np.ndarray

    @property
    def observation_dim(self) -> int:
        return 9 + 3 * self.dof


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


def make_profile(dof: int) -> RobotProfile:
    if dof == 12:
        names = LEG_JOINTS
        root_height = 0.995
        defaults = [
            0.0, 0.0, 0.346431, 0.755514, 0.366252, 0.0,
            0.0, 0.0, 0.346431, 0.755514, 0.366252, 0.0,
        ]
    elif dof == 27:
        names = LEG_JOINTS + UPPER_BODY_JOINTS
        root_height = 1.005
        defaults = [
            0.0, 0.0, 0.289936, 0.742326, 0.409573, 0.0,
            0.0, 0.0, 0.289936, 0.742326, 0.409573, 0.0,
            0.0, 0.25, -0.05, -1.5707963, -0.6, 1.5707963, 0.0, 0.0,
            -0.25, 0.05, 1.5707963, 0.6, -1.5707963, 0.0, 0.0,
        ]
    else:
        raise ValueError(f"Unsupported number of joints: {dof}")

    gains = [_gain_for_joint(name) for name in names]
    stiffness, damping, torque_limits = zip(*gains)
    return RobotProfile(
        dof=dof,
        root_height=root_height,
        joint_names=tuple(names),
        default_joint_pos=np.asarray(defaults, dtype=np.float64),
        stiffness=np.asarray(stiffness, dtype=np.float64),
        damping=np.asarray(damping, dtype=np.float64),
        torque_limits=np.asarray(torque_limits, dtype=np.float64),
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


class HumanoidUltraSim2Sim:
    SIM_DT = 0.005
    CONTROL_DECIMATION = 4
    ACTION_SCALE = 0.25
    HISTORY_LENGTH = 10

    def __init__(self, dof: int, policy_path: Path, command: np.ndarray):
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

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.SIM_DT

        actuator_names = tuple(self.model.actuator(index).name for index in range(self.model.nu))
        if actuator_names != self.profile.joint_names:
            raise RuntimeError(
                "MuJoCo actuator order does not match the Isaac Lab policy order.\n"
                f"Expected: {self.profile.joint_names}\nActual:   {actuator_names}"
            )

        self.qpos_indices = np.asarray(
            [self.model.joint(name).qposadr[0] for name in self.profile.joint_names], dtype=np.int32
        )
        self.qvel_indices = np.asarray(
            [self.model.joint(name).dofadr[0] for name in self.profile.joint_names], dtype=np.int32
        )
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.command = command.astype(np.float64)
        self.previous_action = np.zeros(self.profile.dof, dtype=np.float64)
        self.target_joint_pos = self.profile.default_joint_pos.copy()
        self.observation_history: deque[np.ndarray] = deque(maxlen=self.HISTORY_LENGTH)

        expected_input = self.profile.observation_dim * self.HISTORY_LENGTH
        with torch.inference_mode():
            test_output = self.policy(torch.zeros(1, expected_input, dtype=torch.float32))
        if not isinstance(test_output, torch.Tensor) or tuple(test_output.shape) != (1, self.profile.dof):
            raise RuntimeError(
                f"Policy shape mismatch: expected [1, {self.profile.dof}], got "
                f"{getattr(test_output, 'shape', type(test_output))}"
            )

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, self.profile.root_height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos_indices] = self.profile.default_joint_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.previous_action.fill(0.0)
        self.target_joint_pos[:] = self.profile.default_joint_pos
        self.observation_history.clear()
        mujoco.mj_forward(self.model, self.data)

    def _current_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.qpos_indices]
        joint_vel = self.data.qvel[self.qvel_indices]
        body_angular_velocity = self.data.sensor("BodyGyro").data.copy()
        body_rotation = quaternion_to_rotation_matrix(self.data.qpos[3:7])
        projected_gravity = body_rotation.T @ np.asarray([0.0, 0.0, -1.0])
        observation = np.concatenate(
            (
                body_angular_velocity,
                projected_gravity,
                self.command,
                joint_pos - self.profile.default_joint_pos,
                joint_vel,
                self.previous_action,
            )
        )
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

    def physics_step(self) -> None:
        joint_pos = self.data.qpos[self.qpos_indices]
        joint_vel = self.data.qvel[self.qvel_indices]
        torque = (
            self.profile.stiffness * (self.target_joint_pos - joint_pos)
            - self.profile.damping * joint_vel
        )
        self.data.ctrl[:] = np.clip(torque, -self.profile.torque_limits, self.profile.torque_limits)
        mujoco.mj_step(self.model, self.data)

    def stand(self, duration: float) -> None:
        steps = max(0, int(duration / self.SIM_DT))
        for _ in range(steps):
            self.physics_step()
        self.observation_history.clear()

    def control_step(self) -> None:
        self.update_policy()
        for _ in range(self.CONTROL_DECIMATION):
            self.physics_step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported JIT policy.pt from Isaac Lab.")
    parser.add_argument("--dof", type=int, choices=(12, 27), required=True)
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw velocity command in rad/s.")
    parser.add_argument("--stand-seconds", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration; zero means until viewer closes.")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = np.asarray([args.vx, args.vy, args.yaw_rate], dtype=np.float64)
    simulator = HumanoidUltraSim2Sim(args.dof, args.policy.resolve(), command)
    simulator.stand(args.stand_seconds)
    print(
        f"Loaded {args.dof}-DOF policy. Command: "
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

    def key_callback(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            simulator.reset()
            simulator.stand(args.stand_seconds)
        elif keycode in (ord("X"), ord("x"), 32):
            simulator.command[:] = 0.0
        elif keycode in (ord("W"), ord("w")):
            simulator.command[0] = min(1.0, simulator.command[0] + 0.1)
        elif keycode in (ord("S"), ord("s")):
            simulator.command[0] = max(-0.6, simulator.command[0] - 0.1)
        elif keycode in (ord("A"), ord("a")):
            simulator.command[1] = min(0.5, simulator.command[1] + 0.1)
        elif keycode in (ord("D"), ord("d")):
            simulator.command[1] = max(-0.5, simulator.command[1] - 0.1)
        elif keycode in (ord("Q"), ord("q")):
            simulator.command[2] = min(1.57, simulator.command[2] + 0.1)
        elif keycode in (ord("E"), ord("e")):
            simulator.command[2] = max(-1.57, simulator.command[2] - 0.1)
        print(
            f"command vx={simulator.command[0]:.2f}, "
            f"vy={simulator.command[1]:.2f}, yaw={simulator.command[2]:.2f}"
        )

    print("Keys: W/S forward, A/D lateral, Q/E yaw, X/Space stop, R reset.")
    with mujoco.viewer.launch_passive(
        simulator.model, simulator.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            simulator.control_step()
            viewer.sync()
            if args.duration > 0.0 and time.perf_counter() - start_time >= args.duration:
                break
            sleep_time = control_dt - (time.perf_counter() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
