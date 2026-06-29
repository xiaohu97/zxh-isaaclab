"""Humanoid Ultra 27-DoF 站立 + 左臂轨迹激励跟踪 配置。

在 ``HumanoidUltra27dofStandEnvCfg`` 基础上叠加：
1. ``arm_command`` 配置块（轨迹文件 / 周期 / 渐入 / 开关比例 / 速度缩放）。
2. 左臂位置/速度跟踪奖励；并把原 ``arm_deviation`` 收窄为只管右臂（左臂交给跟踪）。
3. 观测每帧 +15 维（在 env 里追加），observation/state space 同步更新。

站立高度/姿态控制沿用 humanoid_ultra stand（最低高度已是 0.55m + feet_slide/feet_distance，
本就不易劈叉），此处不再额外加抗劈叉项。
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
