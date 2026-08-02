from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc1_pick.tracking_env_cfg import (
    EventCfg as BaseEventCfg,
)
from unitree_rl_lab.tasks.mimic.robots.humanoid_ultra_27dof.ustc_taitui_right.tracking_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
)


TARGET_FRAME_RANGE = (280, 345)
TARGETED_PUSH_LOCAL_VELOCITY_RANGE = {
    # The model_29500 stress failures after the frame-303 kick clustered in
    # local backward-right velocity (mean approximately [-0.173, -0.223]
    # m/s).  Body +Y points left, so both negative X and negative Y reproduce
    # that vulnerable direction.  The inherited random push event continues
    # to cover all linear/angular directions.
    "x": (-0.22, -0.10),
    "y": (-0.28, -0.14),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


@configclass
class CommandsCfg:
    motion = BaseCommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc_taitui_right_stand_transition_hold_2p5s.npz",
        debug_vis=False,
        frame_zero_probability=0.25,
        targeted_frame_range=TARGET_FRAME_RANGE,
        targeted_frame_probability=0.50,
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    # Keep the previously removed arm-joint tracking reward disabled.
    motion_arm_joint_pos = None
    # The failed stress rollouts lose the elevated right ankle first around
    # frames 332-337.  Match the established task-specific ankle reward used
    # by the G1/Humanoid Ultra rear-leg motions.
    motion_right_ankle_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "std": 0.08, "body_names": ["right_ankle_roll_link"]},
    )


@configclass
class EventCfg(BaseEventCfg):
    # Preserve the original random 1-3 s pushes and add one phase-targeted
    # heading-frame backward-right velocity kick to half of the episodes that
    # enter the hard motion window.  Polling every policy step only schedules
    # the event; the class term applies at most one kick per episode.
    targeted_push_robot = EventTerm(
        func=mdp.phase_targeted_velocity_push,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "motion",
            "frame_range": TARGET_FRAME_RANGE,
            "probability": 0.50,
            "velocity_range": TARGETED_PUSH_LOCAL_VELOCITY_RANGE,
        },
    )


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.contact_forces.debug_vis = False


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.terminations.motion_end = None
        # Phase-targeted kicks are a training intervention, not part of a
        # nominal visual playback.  The measurement harness injects its own
        # reproducible pushes when a push profile is requested.
        self.events.targeted_push_robot = None
