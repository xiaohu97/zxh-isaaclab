from __future__ import annotations

import os

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.jumpwithid.tracking_env_cfg import stage_robot_urdf
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import (
    VELOCITY_RANGE,
    RobotEnvCfg as BaseRobotEnvCfg,
    TerminationsCfg as BaseTerminationsCfg,
)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/../jump1/jump1.npz",
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
class TerminationsCfg(BaseTerminationsCfg):
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.55,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        },
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """jumpwithid 的第二阶段：机器人沿用 0521 辨识版 URDF，其余全部回到 ``jump1`` 的
    完整域随机化 + 严格终止阈值。

    第一阶段（``jumpwithid``，继承自 jump1_warmup）把 push_robot / base_com /
    add_joint_default_pos 全关了、终止阈值也放宽到 anchor_ori 1.5 rad，那是为了让策略
    先看到完整的跳跃段。但部署侧 State_Mimic 的安全检查是躯干绝对倾角超 1.0 rad 就切
    Passive，训练容差比部署阈值还宽，策略等于没被约束在安全包络内；再加上没有 CoM 和
    推力随机化，上机落地必然脆。这一阶段把这些都收回去。
    """

    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 和第一阶段共用同一个 URDF 文件（jumpwithid/g1_29dof_rev_1_0_identified0521.urdf），
        # 热启动才成立：换了机器人再 resume 就没有意义了
        self.scene.robot.spawn.asset_path = stage_robot_urdf()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


@configclass
class Stage2PPORunnerCfg(BasePPORunnerCfg):
    # 与第一阶段共用实验目录。train.py 的 resume 是在
    # logs/rsl_rl/<experiment_name>/ 里按 load_run 找 checkpoint 的，
    # 名字不一致就 resume 不到 jumpwithid 那次的 model_29999.pt。
    experiment_name = "unitree_g1_29dof_mimic_jumpwithid"
    run_name = "stage2"
