from __future__ import annotations

import os

from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.humanoid_ultra import HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG
from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc1_pick.tracking_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
)


@configclass
class CommandsCfg:
    motion = BaseCommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc_taitui_right_stand_transition.npz"
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    motion_arm_joint_pos = None


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.terminations.motion_end = None


@configclass
class RobotLeftArm2P5kgEnvCfg(RobotEnvCfg):
    """Right-leg tracking with the identified 2.5 kg left-arm payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )


class RobotLeftArm2P5kgPlayEnvCfg(RobotPlayEnvCfg):
    """Play configuration for the 2.5 kg left-arm payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
