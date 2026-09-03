"""Run 任务专用奖励。

⚠️ **不要复用 locomotion.mdp 里这几个奖励**：``feet_gait`` / ``stand_still`` /
``feet_contact_without_cmd`` / ``feet_height_body`` / ``joint_position_penalty``。
它们都用 ``norm(get_command("base_velocity"))`` 判断"有没有指令"，而本任务的 command
是 8 维、包含躯干高度(~0.75)这类恒正分量，这个范数永远不会小于阈值 —— 复用会静默失效
（不报错，只是奖励恒为 0 或恒为满）。
需要"零指令"语义时改用 ``GaitCommand.is_standing``。
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait_command import IDX_BODY_HEIGHT, IDX_BODY_PITCH, IDX_LIN_VEL_X, GaitCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gait_command(env: ManagerBasedRLEnv, command_name: str) -> GaitCommand:
    return env.command_manager.get_term(command_name)


"""
步态时序 / 足端。
"""


def gait_contact(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励实际触地状态与指令步态时序一致，取值 [0, 2]（每条腿 0 或 1）。

    支撑相比例 θ<0.5 时两腿支撑相不重叠，"两只脚同时不该触地"的时段就是腾空期 ——
    这一项是逼出跑步（而不是快走）的主要驱动力。

    ``sensor_cfg.body_ids`` 顺序必须是 [左脚, 右脚]，与 ``GaitCommand.leg_phase`` 对齐；
    配置里请用 ``preserve_order=True`` 显式列名，不要靠正则的字典序。
    """
    command = _gait_command(env, command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    return (command.desired_contact == is_contact).float().sum(dim=1)


def foot_height_tracking(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """跟踪指令摆动腿高度（参考轨迹 = 摆动相内的半个正弦拱，支撑相 = 贴地）。

    对着完整参考轨迹跟踪而不是只奖励"抬得高"，摆动腿高度才真的成为一个可以在手柄上
    连续拧的量；顺带也惩罚了支撑腿拖地。
    """
    command = _gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    error = torch.square(foot_z - command.desired_foot_height).sum(dim=1)
    return torch.exp(-error / std**2)


"""
躯干姿态（都是指令量，不能再用 flat_orientation_l2 / base_height_l2 一起压死）。
"""


def body_height_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """跟踪指令躯干高度。用 exp 核而不是 L2 惩罚，跑步时的正常上下起伏不会被重罚。"""
    command = _gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    error = torch.square(height - command.command[:, IDX_BODY_HEIGHT])
    return torch.exp(-error / std**2)


def body_pitch_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """跟踪指令躯干俯仰。

    用投影重力的 x 分量近似 sin(pitch)（纯俯仰时严格相等，横滚已被 ``base_roll_l2``
    压到 ~0），省掉四元数→欧拉转换，也没有角度绕圈问题。
    """
    command = _gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    target = torch.sin(command.command[:, IDX_BODY_PITCH])
    error = torch.square(asset.data.projected_gravity_b[:, 0] - target)
    return torch.exp(-error / std**2)


def base_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """只惩罚横滚。俯仰是指令量，交给 ``body_pitch_tracking``。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


"""
摆臂。
"""


def arm_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    max_amplitude: float = 0.5,
    vel_scale: float = 0.25,
) -> torch.Tensor:
    """奖励与步态相位锁定的对侧摆臂，幅度随指令速度增大。

    相位约定：φ=0 时左腿刚触地（处于最前），对侧摆臂要求此刻左臂在最后。G1 的
    shoulder_pitch 与 hip_pitch 同为 +y 轴，正角度都把肢体往后摆，所以左肩目标取
    ``+A·cos(2πφ)``、右肩取相反数。跟踪的是相对默认位姿的偏移量。

    ``asset_cfg.joint_ids`` 顺序必须是 [左肩 pitch, 右肩 pitch]（``preserve_order=True``）。
    另外配置里的 ``joint_deviation_arms`` 必须把 shoulder_pitch 排除掉，否则两项互相打架。
    """
    command = _gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    q_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    amplitude = (command.command[:, IDX_LIN_VEL_X].abs() * vel_scale).clamp(max=max_amplitude)
    swing = amplitude * torch.cos(2.0 * math.pi * command.phase)
    target = torch.stack([swing, -swing], dim=1)
    # 站立时不要求摆臂
    target = torch.where(command.is_standing.unsqueeze(1), torch.zeros_like(target), target)

    return torch.exp(-torch.square(q_rel - target).sum(dim=1) / std**2)


"""
站立。
"""


def stand_still_joint_deviation(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """站立指令下惩罚关节偏离默认位姿（替代 locomotion.mdp.stand_still，见文件头警告）。"""
    command = _gait_command(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return deviation * command.is_standing.float()
