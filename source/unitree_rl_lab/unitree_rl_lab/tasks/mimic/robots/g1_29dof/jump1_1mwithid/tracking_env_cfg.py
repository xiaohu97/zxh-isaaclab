from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.jumpwithid.tracking_env_cfg import stage_robot_urdf
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
    VELOCITY_RANGE,
    RobotEnvCfg as BaseRobotEnvCfg,
)


# 左臂（含手）与双腿的 body 分组。0907 那版 URDF 把 left_rubber_hand 从 0.17kg 加到
# 2.67kg（左手 2.5kg 负载），摆臂惯量涨了 15 倍，跟踪误差随之放大 —— 而参考动作在
# 0.52s 处左手离左髋只有 5.2cm(原点距)，碰撞网格实测表面间隙仅 0.6mm，
# 空手勉强擦过去，手里再拿个 20cm 的球就直接穿进大腿 9.3cm。
LEFT_ARM_BODIES = ["left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link"]
LEG_BODIES = [
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
]


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/../jump1_1m/jump1_1m.npz",
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    """在基类之上只加"别拿左手撞腿"这一件事。

    两层：``left_arm_leg_clearance`` 在碰上之前就给梯度，``left_arm_contact`` 是碰上之后
    的兜底。只加接触惩罚是不够的 —— 没碰到时它恒为 0，策略拿不到"该往外让"的方向。

    权重定标：clearance 惩罚的量纲是"侵入深度(m)之和"。参考动作最深侵入
    margin-0.052 = 0.138 m，配 -20 的权重即最坏 -2.76/步，与 motion_body_pos 的
    weight 1.0（14 个 body 的指数分数，满分 1.0）同量级 —— 足够让策略把手让开，又不会
    压过主体动作跟踪。左腕在跟踪列表里的偏离代价很小：std=0.3 时偏 0.1m 只掉 11%，
    且被 14 个 body 平摊。
    """

    left_arm_leg_clearance = RewTerm(
        func=mdp.self_body_clearance,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_BODIES),
            "other_cfg": SceneEntityCfg("robot", body_names=LEG_BODIES),
            # 原点间距，不是表面间距，数值由碰撞网格实测反推：
            #   空手时 left_wrist_yaw_link 到腿的最小表面间隙只有 0.6mm(第25帧)，
            #   手上那个 20cm 球(半径 0.10, 球心按手原点算)则有 25/90 帧穿透大腿，
            #   最深 92.5mm @第26帧(0.52s)，该帧手-髋原点距 0.0523m。
            # 要让球面刚好脱离腿面需要 0.145m，留 5cm 安全余量 -> 0.19。
            # 换更大的物件就按 (0.145 - 0.10 + 新半径 + 0.05) 重算。
            "margin": 0.19,
        },
    )
    left_arm_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            # 基类 undesired_contacts 的正则把 left_wrist_yaw_link 排除在外，且权重只有
            # -0.1、摊在全身几十个 body 上，对"手碰腿"几乎没有区分度。这里单独把左臂末端
            # 拎出来重罚。跳跃过程中这几个 body 本来就不该碰到任何东西。
            #
            # 注意 left_rubber_hand 在 URDF 里【没有碰撞体】(collision: 0)，仿真里手能
            # 直接穿过大腿、不产生任何接触力 —— 这正是该问题只在真机暴露的原因。所以本项
            # 实际只能罚到两个腕 link，兜不住手本身，真正起作用的是上面的 clearance 项。
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEFT_ARM_BODIES),
            "threshold": 1.0,
        },
    )

@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """``jump1_1m`` 换上 withid 那份 URDF。

    蓝本是 logs/.../unitree_g1_29dof_mimic_jump1_1m/2026-06-03_15-33-42 保存的 cfg，
    即产出线上 Mimic_Jump3 策略的那次训练：从零训、奖励/终止/随机化全部沿用 dance_102
    基类，只换动作片段。当前 jump1_1m/tracking_env_cfg.py 里 06-04 之后加的
    RewardsCfg（jump_landing_distance 等）和 TerminationsCfg 从没训过，这里刻意不带。
    """

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 与 jumpwithid / jumpwithid_stage2 共用同一个 URDF 文件，切版本只改
        # jumpwithid/tracking_env_cfg.py 里的 _URDF_NAME，三个任务一起变
        self.scene.robot.spawn.asset_path = stage_robot_urdf()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


@configclass
class Jump1_1mWithIdPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_mimic_jump1_1mwithid"
    run_name = "identified"
