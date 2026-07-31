from __future__ import annotations

import os

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc1_pick.tracking_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
)


@configclass
class CommandsCfg:
    motion = BaseCommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc1_wave_stand_transition.npz"
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.terminations.motion_end = None
