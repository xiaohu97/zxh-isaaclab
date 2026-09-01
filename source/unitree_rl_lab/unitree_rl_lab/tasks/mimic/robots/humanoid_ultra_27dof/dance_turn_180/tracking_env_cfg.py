from __future__ import annotations

import os

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc1_pick.tracking_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
)


@configclass
class CommandsCfg:
    # Retargeted SEED clip "dance_basic_turn_v1_180_R_loop_001__A321", wrapped in
    # 1 s quintic default-pose transitions and a 2 s terminal standing hold.
    motion = BaseCommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/dance_turn_180_stand_transition.npz",
        debug_vis=False,
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    # The arm joint-position reward is kept: unlike Walk/Wave/Spin, the arm
    # choreography is the point of these clips.
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Headless training does not need marker prims.
        self.scene.contact_forces.debug_vis = False


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.debug_vis = True
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.terminations.motion_end = None
