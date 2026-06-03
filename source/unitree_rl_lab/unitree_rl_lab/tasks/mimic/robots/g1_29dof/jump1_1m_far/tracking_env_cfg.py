from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import (
    VELOCITY_RANGE,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    TerminationsCfg as BaseTerminationsCfg,
)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/jump1_1m_far.npz",
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
    """Fine-tuning rewards for a 1.60m stretched jump reference."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5e-2)
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.30},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.9},
    )
    jump_landing_distance = RewTerm(
        func=mdp.jump_landing_distance,
        weight=0.35,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "min_distance": 1.45,
            "target_distance": 1.65,
            "stable_contact_time": 0.12,
            "max_feet_contact_time_diff": 0.12,
            "min_air_time": 0.15,
            "landing_step_range": (65, 89),
            "max_torso_ori_error": 0.40,
            "max_height_below_ref": 0.14,
            "max_horizontal_speed": 1.0,
            "max_vertical_speed": 0.6,
            "bad_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                    "left_knee_link",
                    "right_knee_link",
                    "left_elbow_link",
                    "right_elbow_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "bad_contact_threshold": 5.0,
        },
    )


@configclass
class TerminationsCfg(BaseTerminationsCfg):
    """Keep fall contacts hard, but allow minor limb brushes during jump-distance tuning."""

    bad_body_contacts = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link"],
            ),
            "threshold": 5.0,
        },
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
