from __future__ import annotations

import os

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc1_pick.tracking_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
)


@configclass
class CommandsCfg:
    motion = BaseCommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc1_spin_stand_transition_hold_2p5s.npz",
        debug_vis=False,
        # The paused policy succeeds from random phases but fails all nominal
        # frame-0 rollouts near frames 107-115.  Reserve part of fine-tuning for
        # the accumulated full-clip state distribution.
        frame_zero_probability=0.25,
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    motion_arm_joint_pos = None


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Headless training and nominal measurements do not need marker prims.
        self.scene.contact_forces.debug_vis = False


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.terminations.motion_end = None
