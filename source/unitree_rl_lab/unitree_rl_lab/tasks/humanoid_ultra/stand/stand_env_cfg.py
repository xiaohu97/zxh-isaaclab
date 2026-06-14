"""Humanoid Ultra 27-DoF stand task based on Unitree-G1-29dof-Stand-v2."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.humanoid_ultra.base import mdp
from unitree_rl_lab.tasks.humanoid_ultra.base.base_config import RewardCfg
from unitree_rl_lab.tasks.humanoid_ultra.base.humanoidultra27dof_env_cfg import (
    Humanoidultra27dofFlatEnvCfg,
)
from unitree_rl_lab.tasks.humanoid_ultra.base.scene_cfg import SceneCfg


STAND_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)

FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
LEG_JOINT_PATTERNS = [
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
ARM_JOINT_PATTERNS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
]


def progressive_velocity_push(
    env,
    env_ids: torch.Tensor,
    curriculum_steps: int = 240_000,
    start_linear_velocity: float = 0.25,
    end_linear_velocity: float = 1.0,
    start_angular_velocity: float = 0.10,
    end_angular_velocity: float = 0.60,
) -> None:
    """Increase horizontal and torso impulse strength over training."""
    progress = min(env.common_step_counter / curriculum_steps, 1.0)
    linear_velocity = start_linear_velocity + progress * (
        end_linear_velocity - start_linear_velocity
    )
    angular_velocity = start_angular_velocity + progress * (
        end_angular_velocity - start_angular_velocity
    )
    mdp.push_by_setting_velocity(
        env,
        env_ids,
        velocity_range={
            "x": (-linear_velocity, linear_velocity),
            "y": (-linear_velocity, linear_velocity),
            "z": (-0.05, 0.05),
            "roll": (-angular_velocity, angular_velocity),
            "pitch": (-angular_velocity, angular_velocity),
            "yaw": (-0.5 * angular_velocity, 0.5 * angular_velocity),
        },
    )


def _torso_roll_pitch(
    env, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    torso_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(torso_quat_w)
    return roll, pitch


def track_height_command(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_height: float = 1.005,
) -> torch.Tensor:
    """Map command vx to standing height, matching the G1 Stand-v2 interface."""
    asset: Articulation = env.scene[asset_cfg.name]
    vx = torch.clamp(env.command_generator.command[:, 0], -1.0, 1.0)
    target_height = torch.where(
        vx >= 0.0,
        nominal_height - 0.455 * vx,
        nominal_height - 0.045 * vx,
    )
    target_height = torch.clamp(target_height, 0.55, 1.05)
    height_error = torch.abs(target_height - asset.data.root_pos_w[:, 2])
    return torch.where(
        height_error < 0.05,
        torch.ones_like(height_error),
        torch.where(
            height_error < 0.15,
            1.0 - (height_error - 0.05) / 0.10,
            -0.5 * torch.square(height_error - 0.15),
        ),
    )


def track_torso_attitude_command(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Map command vy to roll and command yaw to pitch."""
    roll, pitch = _torso_roll_pitch(env, asset_cfg)
    command = env.command_generator.command
    target_roll = 0.20 * torch.clamp(command[:, 1], -1.0, 1.0)
    target_pitch = 0.30 * torch.clamp(command[:, 2], -1.0, 1.0)
    attitude_error = torch.abs(target_roll - roll) + torch.abs(target_pitch - pitch)
    return torch.exp(-2.0 * attitude_error)


def maintain_upright_posture(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Keep the torso upright when the stand command is close to zero."""
    roll, pitch = _torso_roll_pitch(env, asset_cfg)
    command_magnitude = torch.linalg.norm(env.command_generator.command, dim=1)
    tolerance = 0.10 + 0.40 * command_magnitude
    angle_error = torch.abs(roll) + torch.abs(pitch)
    return torch.where(
        angle_error < tolerance,
        torch.ones_like(angle_error),
        torch.exp(-(angle_error - tolerance) / 0.20),
    )


def horizontal_velocity_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def yaw_rate_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w[:, 2])


def both_feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    return torch.all(contact, dim=1).float()


def both_feet_air(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    return torch.logical_not(torch.any(contact, dim=1)).float()


def single_foot_recovery(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > 1.0
    )
    single_contact = torch.sum(contact, dim=1) == 1
    moving = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1) > 0.15
    return torch.logical_and(single_contact, moving).float()


def selected_joint_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize named joints without assuming an array index layout."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


@configclass
class HumanoidUltra27dofStandRewardCfg(RewardCfg):
    height_command_tracking = RewTerm(
        func=track_height_command,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "nominal_height": 1.005},
    )
    torso_attitude_tracking = RewTerm(
        func=track_torso_attitude_command,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["trunk_link"])},
    )
    upright_without_command = RewTerm(
        func=maintain_upright_posture,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["trunk_link"])},
    )
    horizontal_velocity = RewTerm(
        func=horizontal_velocity_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_rate = RewTerm(
        func=yaw_rate_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_contact = RewTerm(
        func=both_feet_contact,
        weight=0.3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES)},
    )
    no_jump = RewTerm(
        func=both_feet_air,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES)},
    )
    step_recovery = RewTerm(
        func=single_foot_recovery,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    leg_joint_velocity = RewTerm(
        func=selected_joint_velocity_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
    )
    waist_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])
        },
    )
    arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_PATTERNS)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
        },
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class HumanoidUltra27dofStandEnvCfg(Humanoidultra27dofFlatEnvCfg):
    reward = HumanoidUltra27dofStandRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene_context.terrain_generator = STAND_FLAT_TERRAIN_CFG
        self.scene_context.max_init_terrain_level = 0
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )

        self.commands.resampling_time_range = (3.0, 6.0)
        self.commands.rel_standing_envs = 0.6
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.debug_vis = True
        self.commands.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)

        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-3.14, 3.14),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.05, 0.05),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.2, 0.2),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.25, 0.25)
        self.events.push_robot.func = progressive_velocity_push
        self.events.push_robot.interval_range_s = (2.0, 4.0)
        self.events.push_robot.params = {
            "curriculum_steps": 240_000,
            "start_linear_velocity": 0.25,
            "end_linear_velocity": 1.0,
            "start_angular_velocity": 0.10,
            "end_angular_velocity": 0.60,
        }

        self.robot.terminate_base_height = 0.45
        self.robot.terminate_base_orientation = 1.2


@configclass
class HumanoidUltra27dofStandTrainEnvCfg(HumanoidUltra27dofStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene_context.num_envs = 4096
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )


@configclass
class HumanoidUltra27dofStandPlayEnvCfg(HumanoidUltra27dofStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene_context.num_envs = 50
        self.noise.add_noise = False
        self.commands.rel_standing_envs = 1.0
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.events.push_robot = None
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
