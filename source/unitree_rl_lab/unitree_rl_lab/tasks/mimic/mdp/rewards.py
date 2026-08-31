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


def single_support_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,
    tilt_scale: float = 0.20,
    angular_velocity_scale: float = 1.5,
) -> torch.Tensor:
    """Reward a quiet, upright torso while exactly one foot supports the robot.

    This is deliberately gated by measured foot contact, so it does not reward
    freezing in the air or merely matching the reference pose.  The exponential
    score is one near upright/low angular velocity and smoothly approaches zero
    as the single-support torso becomes unstable.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_contact = torch.linalg.norm(foot_forces, dim=-1) > contact_threshold
    single_support = foot_contact.sum(dim=-1) == 1

    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    angular_velocity = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=-1)
    score = torch.exp(
        -torch.square(tilt / tilt_scale)
        - torch.square(angular_velocity / angular_velocity_scale)
    )
    return score * single_support.float()


def feet_impact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Measure downward foot speed on the first simulation step of contact.

    The reward manager applies the configured negative weight, so this term
    penalizes hard touchdown without penalizing the normal vertical velocity
    while the foot is still in swing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    downward_velocity = torch.clamp(
        -asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=0.0
    )
    return torch.sum(first_contact * downward_velocity, dim=-1)


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str],
    reference_height_threshold: float = 0.30,
    max_height_error: float = 0.50,
) -> torch.Tensor:
    """Give dense swing-foot height tracking reward during the lifted phase.

    The existing exponential ankle-position reward is almost zero when a foot
    remains on the floor while the reference is high.  This linear score keeps
    a useful learning signal across that large error: zero at
    ``max_height_error`` and one at the reference height.  It deliberately does
    not depend on contact state; contact mismatch is handled by the separate
    swing-foot contact term, so unloading a foot cannot unlock this reward.
    """
    if max_height_error <= 0.0:
        raise ValueError(f"max_height_error must be positive, got {max_height_error}")

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_idxs = _get_body_indexes(command, body_names)
    reference_z = command.body_pos_relative_w[:, body_idxs, 2]
    actual_z = command.robot_body_pos_w[:, body_idxs, 2]
    reference_in_swing = reference_z > reference_height_threshold
    height_error = torch.abs(reference_z - actual_z)
    height_score = 1.0 - torch.clamp(height_error / max_height_error, min=0.0, max=1.0)
    return torch.sum(reference_in_swing.float() * height_score, dim=-1)


def swing_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str],
    contact_threshold: float = 10.0,
    reference_height_threshold: float = 0.30,
) -> torch.Tensor:
    """Penalize foot contact when reference shows foot should be in swing.

    When the reference foot is above ``reference_height_threshold`` (indicating swing phase),
    any measured contact force is penalized. This prevents the "shift weight but
    keep foot on ground" exploit.

    Args:
        command_name: Motion command term name
        sensor_cfg: Contact sensor config for the feet
        body_names: Specific foot body names to track
        contact_threshold: Contact force magnitude threshold (N)
        reference_height_threshold: Reference Z height defining swing phase (m)

    Returns:
        Count of feet that should be swinging but are contacting ground
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_idxs = _get_body_indexes(command, body_names)
    reference_z = command.body_pos_relative_w[:, body_idxs, 2]

    # Real contact state
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_contact = torch.linalg.norm(foot_forces, dim=-1) > contact_threshold
    if reference_z.shape[1] != foot_contact.shape[1]:
        raise ValueError(
            "swing-foot body/contact count mismatch: "
            f"{reference_z.shape[1]} reference bodies vs {foot_contact.shape[1]} contact bodies"
        )

    # Should be swinging (ref high) but is contacting (bad)
    reference_in_swing = reference_z > reference_height_threshold
    violation = reference_in_swing & foot_contact

    return torch.sum(violation.float(), dim=-1)


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


def feet_contact_force_excess(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold_body_weights: float = 1.5,
) -> torch.Tensor:
    """Penalize foot contact force above a multiple of body weight.

    This replaces ``feet_impact_velocity`` for the landing objective, which
    cannot work as written: ``compute_first_contact`` is true on the env step
    *after* contact is established, and physics runs 4 substeps per env step, so
    the foot's vertical velocity has already been arrested by the time the term
    reads it.  Measured on the identified plant over 60 seeds, the true
    pre-impact speed is 0.401 m/s median, which at weight -0.6 should log about
    0.049; the term actually logs -0.00046, two orders of magnitude down, and
    tripling the weight from -0.2 moved nothing.

    Force is read from the sensor's history and reduced with a max rather than
    the instantaneous value, so a spike that peaks between env steps is still
    seen.  The excess is normalized by body weight, which keeps the term O(1)
    and makes the threshold mean what it says regardless of the payload.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak = torch.max(torch.linalg.norm(forces, dim=-1), dim=1).values
    weight = env.scene["robot"].data.default_mass.sum(dim=-1).to(peak.device) * 9.81
    excess = torch.clamp(peak / weight.unsqueeze(-1) - threshold_body_weights, min=0.0)
    return torch.sum(excess, dim=-1)


def motion_body_speed_overshoot(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str],
    tolerance: float = 1.15,
) -> torch.Tensor:
    """Penalize end-effector speed beyond what the reference itself moves at.

    This attacks the source of the twist rather than its outlets.  Measured on
    the identified plant over 40 seeds at 15 ms delay, against a reference whose
    swing foot peaks at 2.756 m/s:

                    swing-foot peak   vertical |Lz|   torso yaw rate   falls/100
      0725              2.596 m/s        3.844          2.829 rad/s        0
      J1@62000          4.517 m/s        6.588          3.819 rad/s       11

    Same lift height (0.547 vs 0.545), same moment arm (0.851 vs 0.845 m), same
    lateral offset (0.820 vs 0.822 m) -- the whole difference is speed.  0725
    tracks the reference to within 6% and generates little angular momentum
    about vertical; the overshooting policy generates 71% more, which has to
    leave through the torso (the yaw twist) or through horizontal translation
    (drift, and falls).  ``anchor_yaw`` closes the first outlet without reducing
    what needs to leave, which is why adding it cut yaw from 37.7 to 23 degrees
    and pushed falls from 5 up to 11-61.

    ``motion_body_lin_vel`` does not cover this: it is an exponential on the
    velocity *error*, wide enough that a 64% overshoot costs little, and it
    rewards matching the reference direction rather than bounding the magnitude.

    The penalty is one-sided, so tracking slower than the reference is free.
    Only overshoot is charged, and only above ``tolerance``, which keeps normal
    tracking noise out of it.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    reference = torch.linalg.norm(command.body_lin_vel_w[:, body_indexes], dim=-1)
    actual = torch.linalg.norm(command.robot_body_lin_vel_w[:, body_indexes], dim=-1)
    excess = torch.clamp(actual - tolerance * reference, min=0.0)
    return torch.sum(excess, dim=-1)


def motion_joint_deviation_excess(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge only the part of a joint's tracking error above ``threshold``.

    ``motion_joint_position_error_exp`` on the hip yaws was tried and moved the
    wrong statistic.  Its reward climbed from 0.007 to 0.22 of a 3.65 total
    while the measured deviation at the worst instant went from -70.8 to -83.5
    degrees on the right hip: an exponential on mean squared error pays for the
    easy majority of the episode and leaves the peak alone, and the defect is
    entirely in the peak.

    One-sided and thresholded, so tracking inside the band is free and the
    gradient exists only where the deviation is actually objectionable.  The
    same shape is what made ``feet_contact_force_excess`` work.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_ids = asset_cfg.joint_ids
    error = torch.abs(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids])
    return torch.sum(torch.clamp(error - threshold, min=0.0), dim=-1)


def raw_action_excess(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    joint_names: list[str],
    threshold: float,
) -> torch.Tensor:
    """Charge the part of the *raw* policy output beyond ``threshold``.

    Clipping an action in the environment bounds what the robot receives, but
    it does not bound what the network asks for: every raw value past the clip
    produces the identical transition, so PPO gets no gradient distinguishing
    them and the output magnitude on that dimension is free to drift.  V1
    demonstrated both halves of this -- under the training clip (+-0.20 rad)
    the achieved ankle roll settled at -0.2..-0.4 deg, better than 0808, while
    the same checkpoint evaluated under sim2sim's looser +-0.5236 rad clip
    reverted to -25..-29 deg commands, because nothing had ever constrained
    the raw output that the tighter clip was silently absorbing.

    That matters because the training clip does not travel with the exported
    policy: sim2sim and the real controller apply their own, wider limits.  So
    this penalises the raw request itself, making the bound a property of the
    network rather than of whichever environment happens to run it.

    ``threshold`` is in raw-action units.  With ``use_default_offset`` and a
    default of zero, the joint target is ``raw * scale``, so the raw value
    matching a position clip of ``c`` rad is ``c / scale`` -- for ankle roll,
    0.20 / 0.25 = 0.8.  One-sided, so ordinary use of the full clipped range
    stays free and only the unreachable surplus is charged.
    """
    term = env.action_manager.get_term(action_term_name)
    asset = env.scene[term.cfg.asset_name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=False)
    term_ids = term._joint_ids
    if isinstance(term_ids, slice):
        columns = joint_ids
    else:
        lookup = {int(j): i for i, j in enumerate(term_ids)}
        columns = [lookup[int(j)] for j in joint_ids]
    raw = term.raw_actions[:, columns]
    return torch.sum(torch.clamp(torch.abs(raw) - threshold, min=0.0), dim=-1)
