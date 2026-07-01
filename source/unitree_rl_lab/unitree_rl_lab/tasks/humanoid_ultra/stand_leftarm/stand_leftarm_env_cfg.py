"""Humanoid Ultra 27-DoF 站立 + 左臂轨迹激励跟踪 配置。

在 ``HumanoidUltra27dofStandEnvCfg`` 基础上叠加：
1. ``arm_command`` 配置块（轨迹文件 / 周期 / 渐入 / 开关比例 / 速度缩放）。
2. 左臂位置/速度跟踪奖励；并把原 ``arm_deviation`` 收窄为只管右臂（左臂交给跟踪）。
3. 观测每帧 +15 维（在 env 里追加），observation/state space 同步更新。

对高度调节做了修正（见 _commanded_height / track_height_command_squat）：
基类 stand 的高度奖励最低目标 0.55m、前推斜率 -0.455（≈55% 腿长、贴近 0.45 死亡线），
且 stand 奖励集继承的是空 RewardCfg、并未继承 locomotion 的 feet_distance/knee_distance/hip
偏离等抗劈叉项 —— 没有任何项阻止机器人靠岔腿降高度。故这里对齐 G1 Stand-LeftArmTrack：
改用分段平滑高度奖励并新增 hip_posture_penalty；当前最低目标为 0.60m。高位目标收至
1.03m，并用 knee_limit_margin 在膝盖伸直限位前保留余量。
"""
from __future__ import annotations

import os
from dataclasses import MISSING

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.humanoid_ultra.base import mdp
from unitree_rl_lab.tasks.humanoid_ultra.base.base_config import EventCfg
from unitree_rl_lab.tasks.humanoid_ultra.base.scene_cfg import SceneCfg
from unitree_rl_lab.tasks.humanoid_ultra.stand.stand_env_cfg import (
    HumanoidUltra27dofStandEnvCfg,
    HumanoidUltra27dofStandRewardCfg,
)

# 负载所在 link：手腕链为 yaw->roll->pitch，最远端(手/末端执行器)是 left_wrist_pitch_link，
# 一只手持物的质量实际作用在这里、对全臂力臂最大。若想与 G1 一致改成 "left_wrist_yaw_link"。
PAYLOAD_LINK = "left_wrist_pitch_link"

# CSV 列顺序 = 跟踪关节顺序
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
# 左臂被跟踪后，右臂仍按名字保持默认位姿
RIGHT_ARM_JOINTS = ["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]
_TRAJ_FILE = os.path.join(os.path.dirname(__file__), "left_wrist_pitch_traj.csv")

# 抗劈叉：约束髋外展(roll)/外旋(yaw)，让降高度时腿留在身体下方而非岔开（对齐 G1）。
HIP_LATERAL_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
]

# 高度目标常量（替换基类 -0.455 斜率 / 最低 0.55m 的版本）。
# 最低目标 0.60m，与 0.45m 终止高度保留 15cm 余量。
_HEIGHT_NOMINAL = 1.005
_HEIGHT_FWD_SLOPE = 0.405  # vx=+1 -> 0.60m（在原 0.70m 基础上再降 10cm）
_HEIGHT_BWD_SLOPE = 0.025  # vx=-1 -> 1.03m（原 1.05m，避免膝盖伸直碰限位）
_HEIGHT_MIN = 0.60
_HEIGHT_MAX = 1.03


# ==============================================================================
# 奖励函数（读取 env 上由 env 子类计算好的左臂参考）
# ==============================================================================
def track_left_arm_pos(env, std: float) -> torch.Tensor:
    """左臂关节位置跟踪（按关节取 mean，保证早期有梯度）。关闭跟踪的 env 目标=默认位姿。"""
    cur_rel = env.robot.data.joint_pos[:, env.arm_joint_ids] - env.arm_default_q
    err = torch.mean(torch.square(cur_rel - env.arm_q_ref_rel), dim=1)
    return torch.exp(-err / std**2)


def track_left_arm_vel(env, std: float) -> torch.Tensor:
    """左臂关节速度跟踪（提供运动方向，抑制相位滞后）。"""
    cur = env.robot.data.joint_vel[:, env.arm_joint_ids]
    err = torch.mean(torch.square(cur - env.arm_dq_ref), dim=1)
    return torch.exp(-err / std**2)


def knee_straight_margin(env, asset_cfg: SceneEntityCfg, margin: float = 0.15) -> torch.Tensor:
    """膝关节接近“伸直”(下软限位)时的余量惩罚：仅在膝角低于 (软下限+margin) 时激活。

    站直时膝盖会朝下限位(伸直)靠拢；此项在到达限位前就给惩罚，
    促使策略保留一点屈膝、远离限位。正常/下蹲姿态下膝角远高于阈值 -> 惩罚为 0。
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    violation = torch.clamp((lower + margin) - q, min=0.0)
    return torch.sum(torch.square(violation), dim=1)


# ==============================================================================
# 高度调节奖励（修正：抬高最低目标 + 平滑分段 + 抗劈叉）
# ==============================================================================
def _commanded_height(env) -> torch.Tensor:
    """由 vx 指令映射目标站立高度（前推蹲下 / 后拉略升）。"""
    vx = torch.clamp(env.command_generator.command[:, 0], -1.0, 1.0)
    target = torch.where(
        vx >= 0.0,
        _HEIGHT_NOMINAL - _HEIGHT_FWD_SLOPE * vx,
        _HEIGHT_NOMINAL - _HEIGHT_BWD_SLOPE * vx,
    )
    return torch.clamp(target, _HEIGHT_MIN, _HEIGHT_MAX)


def track_height_command_squat(env, asset_cfg, nominal_height: float = 1.005, std: float = 0.10) -> torch.Tensor:
    """分段高度跟踪，替换基类 exp 版：误差大时仍保留平滑梯度，引导屈膝下蹲而非站死。

    nominal_height / std 仅为兼容基类 RewTerm 的 params 签名而保留，实际目标由模块常量给出。
    """
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    h_error = torch.abs(_commanded_height(env) - curr_h)
    return torch.where(
        h_error < 0.05,
        torch.ones_like(h_error),
        torch.where(
            h_error < 0.15,
            1.0 - (h_error - 0.05) / 0.1,
            -0.5 * torch.square(h_error - 0.15),
        ),
    )


def height_command_error_squat(env, asset_cfg, nominal_height: float = 1.005) -> torch.Tensor:
    """与跟踪项共用同一目标高度的 L1 误差（配负权重直接 shaping/日志）。"""
    asset = env.scene[asset_cfg.name]
    return torch.abs(_commanded_height(env) - asset.data.root_pos_w[:, 2])


def hip_lateral_deviation_l2(env, asset_cfg) -> torch.Tensor:
    """抗劈叉：髋 roll/yaw 偏离默认位姿的平方和（正值，配负权重）。"""
    asset = env.scene[asset_cfg.name]
    rel = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(rel), dim=1)


# ==============================================================================
# 配置块
# ==============================================================================
@configclass
class ArmCommandCfg:
    """左臂轨迹激励配置（被 env 子类读取）。"""

    joint_names: list = MISSING
    traj_file: str = MISSING
    period: float = 6.0
    blend_time_s: float = 2.0  # 默认位姿与轨迹起点差异很大(肩偏航~3.4rad)，渐入设长一些
    rel_enabled_envs: float = 0.8
    randomize_start_phase: bool = True
    ref_vel_scale: float = 0.25


@configclass
class StandLeftArmRewardCfg(HumanoidUltra27dofStandRewardCfg):
    left_arm_pos_tracking = RewTerm(func=track_left_arm_pos, weight=4.0, params={"std": 0.15})
    left_arm_vel_tracking = RewTerm(func=track_left_arm_vel, weight=0.7, params={"std": 1.4})
    # 站高时避免膝盖伸直碰限位（限位前留余量；正常姿态下为 0，不扰动稳定行为）
    knee_limit_margin = RewTerm(
        func=knee_straight_margin,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]), "margin": 0.15},
    )
    # 抗劈叉：罚髋 roll/yaw 偏离默认 -> 降高度时屈膝而非岔腿（对齐 G1）。
    hip_posture_penalty = RewTerm(
        func=hip_lateral_deviation_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_LATERAL_JOINTS)},
    )


@configclass
class StandLeftArmEventCfg(EventCfg):
    """在基类事件之外，新增左手腕 ~3kg 负载质量随机化（模拟手持物）。"""

    add_left_wrist_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=PAYLOAD_LINK),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )


# ==============================================================================
# 环境配置
# ==============================================================================
@configclass
class HumanoidUltra27dofStandLeftArmEnvCfg(HumanoidUltra27dofStandEnvCfg):
    reward = StandLeftArmRewardCfg()
    events = StandLeftArmEventCfg()
    arm_command: ArmCommandCfg = ArmCommandCfg()

    def __post_init__(self):
        super().__post_init__()

        self.arm_command.joint_names = LEFT_ARM_JOINTS
        self.arm_command.traj_file = _TRAJ_FILE

        # 左臂交给跟踪奖励 -> 原 arm_deviation 只罚右臂，避免与跟踪冲突
        self.reward.arm_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=RIGHT_ARM_JOINTS
        )

        # 分段平滑高度奖励 + 抗劈叉：目标范围 0.60--1.03m，最低处仍高于 0.45m 终止线。
        self.reward.height_command_tracking.func = track_height_command_squat
        self.reward.height_command_error.func = height_command_error_squat

        # 每帧观测 +15 (q_ref_rel 7 + dq_ref 7 + enabled 1)；与基类约定一致按“每帧维度”记
        self.observation_space = 90 + 15
        self.state_space = 159 + 15


@configclass
class HumanoidUltra27dofStandLeftArmTrainEnvCfg(HumanoidUltra27dofStandLeftArmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene_context.num_envs = 4096
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )


@configclass
class HumanoidUltra27dofStandLeftArmPlayEnvCfg(HumanoidUltra27dofStandLeftArmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene_context.num_envs = 50
        self.noise.add_noise = False
        self.commands.rel_standing_envs = 1.0
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.events.push_robot = None
        # 演示：左臂全程跟踪、从相位 0 起
        self.arm_command.rel_enabled_envs = 1.0
        self.arm_command.randomize_start_phase = False
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
