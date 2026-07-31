from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_anchor_xy_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward horizontal recovery to the reference anchor position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error_xy = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return torch.exp(-torch.sum(torch.square(error_xy), dim=-1) / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward selected joints for matching the reference motion positions."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_ids = asset_cfg.joint_ids
    error = torch.square(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids])
    return torch.exp(-error.mean(dim=-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def jump_landing_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_distance: float,
    target_distance: float,
    stable_contact_time: float,
    max_feet_contact_time_diff: float,
    min_air_time: float,
    landing_step_range: tuple[int, int],
    max_torso_ori_error: float,
    max_height_below_ref: float,
    max_horizontal_speed: float,
    max_vertical_speed: float,
    bad_contact_sensor_cfg: SceneEntityCfg | None = None,
    bad_contact_threshold: float = 1.0,
) -> torch.Tensor:
    """One-shot landing distance reward gated by stable two-foot landing."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = command.robot_anchor_pos_w.device

    paid_attr = "_jump_landing_distance_paid"
    paid = getattr(env, paid_attr, None)
    if paid is None or paid.shape[0] != env.num_envs:
        paid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        setattr(env, paid_attr, paid)

    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is not None:
        paid[episode_length <= 1] = False

    foot_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    foot_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    foot_contact_min = torch.min(foot_contact_time, dim=-1).values
    foot_contact_diff = torch.max(foot_contact_time, dim=-1).values - foot_contact_min

    feet_stable = foot_contact_min >= stable_contact_time
    feet_contact_close = foot_contact_diff <= max_feet_contact_time_diff
    enough_air_time = torch.all(foot_air_time >= min_air_time, dim=-1)

    start_step, end_step = landing_step_range
    in_landing_phase = (command.time_steps >= start_step) & (command.time_steps <= end_step)

    torso_ori_error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    torso_ori_ok = torso_ori_error <= max_torso_ori_error
    torso_height_ok = command.robot_anchor_pos_w[:, 2] >= command.anchor_pos_w[:, 2] - max_height_below_ref
    torso_horizontal_speed = torch.linalg.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    torso_horizontal_ok = torso_horizontal_speed <= max_horizontal_speed
    torso_vertical_ok = torch.abs(command.robot_anchor_lin_vel_w[:, 2]) <= max_vertical_speed

    no_bad_contact = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    if bad_contact_sensor_cfg is not None:
        bad_contact_sensor: ContactSensor = env.scene.sensors[bad_contact_sensor_cfg.name]
        net_forces = bad_contact_sensor.data.net_forces_w_history[:, :, bad_contact_sensor_cfg.body_ids]
        bad_contact = torch.max(torch.linalg.norm(net_forces, dim=-1), dim=1).values > bad_contact_threshold
        no_bad_contact = ~torch.any(bad_contact, dim=-1)

    anchor_index = command.motion_anchor_body_index
    motion_xy = command.motion.body_pos_w[:, anchor_index, :2]
    jump_direction = motion_xy[-1] - motion_xy[0]
    jump_direction = jump_direction / torch.clamp(torch.linalg.norm(jump_direction), min=1.0e-6)
    start_xy = motion_xy[0] + env.scene.env_origins[:, :2]
    distance = torch.sum((command.robot_anchor_pos_w[:, :2] - start_xy) * jump_direction, dim=-1)
    distance_range = max(target_distance - min_distance, 1.0e-6)
    score = torch.clamp((distance - min_distance) / distance_range, min=0.0, max=1.0)

    stable_landing = (
        feet_stable
        & feet_contact_close
        & enough_air_time
        & in_landing_phase
        & torso_ori_ok
        & torso_height_ok
        & torso_horizontal_ok
        & torso_vertical_ok
        & no_bad_contact
        & ~paid
    )
    paid[stable_landing] = True
    return score * stable_landing.float() / env.step_dt
