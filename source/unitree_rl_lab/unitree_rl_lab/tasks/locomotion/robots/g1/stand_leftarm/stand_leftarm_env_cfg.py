"""G1 29DOF 站立 + 左臂轨迹跟踪任务 (Stand-LeftArmTrack-v0)。

在 ``Unitree-G1-29dof-Stand-v2`` 的基础上叠加：
1. 一个左臂关节轨迹命令项 ``left_arm``（Fourier 还原 + 五次 smoothstep 渐入 + 开关）。
2. 观测末尾追加 ``arm_command``（q_ref_rel 7 + dq_ref 7 + enabled 1 = 15 维，history=1）。
3. 左臂位置/速度跟踪奖励。
4. ``left_wrist_yaw_link`` 上 ~3kg 质量随机化（模拟手持负载）。

同时**顺手修复**原 Stand-v2 里按硬编码索引（[12,13,14] / 15..28 / :12）选关节导致的
错位 bug —— Isaac 内部关节顺序是按树深度的左右交替序，与 SDK 线性序不同，必须按名字解析。
本任务为独立任务，不改动原 Stand-v2。
"""
from __future__ import annotations

import os
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.stand.stand_env_cfg_v2 import (
    CommandsCfg,
    EventCfg,
    G1StandEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

# CSV 列顺序（= 命令项跟踪的关节顺序）
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# 左臂交给跟踪奖励，右臂仍按名字保持默认位姿。
# 腰/腿速/双臂 的索引修正已在 Stand-v2 基类（stand_env_cfg_v2）完成并被继承。
RIGHT_ARM_JOINTS = ["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]

# 抗劈叉：约束髋外展(roll)/外旋(yaw)，让降高度时腿留在身体下方而非岔开。
HIP_LATERAL_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
]

_TRAJ_FILE = os.path.join(os.path.dirname(__file__), "left_wrist_traj.csv")


# ==============================================================================
# 奖励函数
# ==============================================================================
def track_left_arm_pos(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """左臂关节位置跟踪（按关节数取 mean，保证早期有学习信号）。

    误差以“相对默认位姿”口径计算，与命令项 ``q_ref_rel`` 对齐。关闭跟踪的 env 其
    ``q_ref_rel=0``，该奖励退化为“保持默认位姿”，从而实现收手。
    """
    cmd = env.command_manager.get_term(command_name)
    cur_rel = cmd.robot.data.joint_pos[:, cmd.joint_ids] - cmd.default_q
    err = torch.mean(torch.square(cur_rel - cmd.q_ref_rel), dim=1)
    return torch.exp(-err / std**2)


def track_left_arm_vel(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """左臂关节速度跟踪（提供运动方向信息，抑制相位滞后）。"""
    cmd = env.command_manager.get_term(command_name)
    cur = cmd.robot.data.joint_vel[:, cmd.joint_ids]
    err = torch.mean(torch.square(cur - cmd.dq_ref), dim=1)
    return torch.exp(-err / std**2)


def track_height_command_squat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(A) 高度跟踪——与 Stand-v2 同款分段奖励，但**最低目标抬到 0.50m**。

    0.28m 双脚踩平在运动学上几乎只能靠劈叉达成；抬到 0.50m 后，屈膝(蹲)成为可行解，
    配合 (B) 髋外展惩罚即可让机器人屈膝下蹲而非劈叉。
    """
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]

    cmd = env.command_manager.get_term("base_velocity").command
    vx_norm = torch.clamp(cmd[:, 0], -1.0, 1.0)
    target_h = torch.where(
        vx_norm >= 0.0,
        0.78 - 0.28 * vx_norm,  # 前推: vx=1.0 -> 0.50m（原 0.28m）
        0.78 - 0.07 * vx_norm,  # 后拉: vx=-1.0 -> 0.85m
    )
    target_h = torch.clamp(target_h, 0.50, 0.85)

    h_error = torch.abs(target_h - curr_h)
    return torch.where(
        h_error < 0.05,
        torch.ones_like(h_error),
        torch.where(
            h_error < 0.15,
            1.0 - (h_error - 0.05) / 0.1,
            -0.5 * torch.square(h_error - 0.15),
        ),
    )


def penalize_joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """按名字解析的关节偏离默认位姿惩罚（用于 (B) 髋外展约束）。"""
    asset = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return -torch.sum(torch.square(rel), dim=1)


# ==============================================================================
# 命令 / 观测 / 奖励 / 事件 子配置
# ==============================================================================
@configclass
class LeftArmCommandsCfg(CommandsCfg):
    """保留 base_velocity，新增左臂轨迹命令。"""

    left_arm = mdp.LeftArmJointTrajectoryCommandCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS,
        traj_file=_TRAJ_FILE,
        period=6.0,
        # 只在 reset 时重采样相位/开关；设很大避免 episode 中途跳变
        resampling_time_range=(1.0e9, 1.0e9),
        blend_time_s=1.5,
        rel_enabled_envs=0.8,
        randomize_start_phase=True,
        ref_vel_scale=0.25,
        debug_vis=False,
    )


@configclass
class LeftArmObservationsCfg(ObservationsCfg):
    """在 policy/critic 末尾追加 arm_command；本体项保持 5 帧历史，arm_command 仅 1 帧。"""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        arm_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_arm"}, history_length=1
        )

        def __post_init__(self):
            super().__post_init__()  # 设 group history=5 / corruption / concatenate
            # group 置 None -> 改为 per-term 控制历史，避免 arm_command 被 ×5
            self.history_length = None
            for name in [
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action",
            ]:
                getattr(self, name).history_length = 5
            self.arm_command.history_length = 1

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        arm_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_arm"}, history_length=1
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = None
            for name in [
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action",
            ]:
                getattr(self, name).history_length = 5
            self.arm_command.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class LeftArmRewardsCfg(RewardsCfg):
    """保留站立奖励，新增左臂跟踪。原错位惩罚在 env __post_init__ 内按名字修正。"""

    left_arm_pos_tracking = RewTerm(
        func=track_left_arm_pos,
        weight=4.0,
        params={"command_name": "left_arm", "std": 0.15},
    )
    left_arm_vel_tracking = RewTerm(
        func=track_left_arm_vel,
        weight=0.7,
        params={"command_name": "left_arm", "std": 1.4},
    )
    # (B) 抗劈叉：罚髋 roll/yaw 偏离默认 -> 腿保持在身体下方，降高度时屈膝而非岔腿
    hip_posture_penalty = RewTerm(
        func=penalize_joint_deviation_l2,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_LATERAL_JOINTS)},
    )


@configclass
class LeftArmEventCfg(EventCfg):
    """新增 left_wrist_yaw_link 质量随机化（手持负载, 0~3kg）。"""

    add_left_wrist_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )


# ==============================================================================
# 主环境配置
# ==============================================================================
@configclass
class G1StandLeftArmEnvCfg(G1StandEnvCfg):
    commands: LeftArmCommandsCfg = LeftArmCommandsCfg()
    observations: LeftArmObservationsCfg = LeftArmObservationsCfg()
    rewards: LeftArmRewardsCfg = LeftArmRewardsCfg()
    events: LeftArmEventCfg = LeftArmEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Stand-v2 基类已把 腰/腿速/双臂 惩罚改为按名字解析；这里只需把“双臂保姿”
        # 收窄为“仅右臂”，左臂由 left_arm 跟踪奖励控制（避免与跟踪冲突）。
        self.rewards.arm_penalty.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)
        }

        # (A) 抗劈叉：把高度跟踪换成最低 0.50m 的版本（屈膝可达，劈叉不再被逼出）。
        self.rewards.height_command_tracking.func = track_height_command_squat


@configclass
class G1StandLeftArmEnvCfg_TRAIN(G1StandLeftArmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.observations.policy.enable_corruption = True


@configclass
class G1StandLeftArmEnvCfg_PLAY(G1StandLeftArmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        # 站立演示：速度命令清零
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # 左臂全程跟踪、从相位 0 起，方便观察
        self.commands.left_arm.rel_enabled_envs = 1.0
        self.commands.left_arm.randomize_start_phase = False


@configclass
class G1StandLeftArmPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_stand_leftarm"
