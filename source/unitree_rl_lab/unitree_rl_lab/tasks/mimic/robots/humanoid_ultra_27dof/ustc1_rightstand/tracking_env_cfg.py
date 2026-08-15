from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.domain_randomization import PlantRandomizationEventCfg

from unitree_rl_lab.assets.robots.humanoid_ultra import (
    HUMANOIDULTRA27DOF_MIMIC_CFG as ROBOT_CFG,
    HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG,
)
from unitree_rl_lab.tasks.mimic.terrain import LocalGridPlaneTerrainImporter

##
# Scene definition
##

# Flat action scale (matches the direct-workflow humanoid_ultra env action_scale=0.25)
ROBOT_ACTION_SCALE = 0.25

# Position-command safety boundary used by the real Humanoid Ultra controller.
# These are command limits, not the articulation's physical joint limits.
DEPLOYMENT_JOINT_POSITION_LIMITS = {
    "left_hip_roll_joint": (-0.25, 1.5708),
    "left_hip_yaw_joint": (-1.5708, 1.5708),
    "left_hip_pitch_joint": (-1.5708, 1.5708),
    "left_knee_joint": (0.0, 2.356),
    "left_ankle_pitch_joint": (-0.7, 0.95),
    "left_ankle_roll_joint": (-0.5236, 0.5236),
    "right_hip_roll_joint": (-1.5708, 0.25),
    "right_hip_yaw_joint": (-1.5708, 1.5708),
    "right_hip_pitch_joint": (-1.5708, 1.5708),
    "right_knee_joint": (0.0, 2.356),
    "right_ankle_pitch_joint": (-0.7, 0.95),
    "right_ankle_roll_joint": (-0.5236, 0.5236),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-2.4, 1.2),
    "left_shoulder_roll_joint": (-0.3, 2.7),
    "left_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-2.17, 0.0),
    "left_wrist_yaw_joint": (-2.5, 2.5),
    "left_wrist_roll_joint": (-1.11, 1.11),
    "left_wrist_pitch_joint": (-1.05, 1.05),
    "right_shoulder_pitch_joint": (-1.2, 2.4),
    "right_shoulder_roll_joint": (-2.7, 0.3),
    "right_shoulder_yaw_joint": (-2.5, 2.5),
    "right_elbow_joint": (0.0, 2.17),
    "right_wrist_yaw_joint": (-2.5, 2.5),
    "right_wrist_roll_joint": (-1.11, 1.11),
    "right_wrist_pitch_joint": (-1.05, 1.05),
}

# The deployment controller updates at 50 Hz, so this permits at most 0.12 rad
# of position-target change per policy step.
DEPLOYMENT_TARGET_VELOCITY = 6.0

# Action EMA used by the houtaituiEMA task only.  The alpha=0.7 run still
# plateaued well behind the unfiltered policy after the first 30-minute window;
# alpha=0.85 retains a small amount of touchdown smoothing with less task-band
# phase lag during single-support recovery.
ACTION_EMA_ALPHA = 0.85

# Anchor body: central torso link used to align robot vs. reference (G1 uses "torso_link").
ANCHOR_BODY_NAME = "trunk_link"

# Body/link names tracked by the motion command. Same set as the G1 mimic env,
# mapped onto the humanoid_ultra 27dof URDF link names:
#   G1 "pelvis"          -> "base_link"
#   G1 "torso_link"      -> "trunk_link"
#   G1 "*_wrist_yaw_link"-> "*_wrist_pitch_link" (HU wrist chain is yaw->roll->pitch)
TRACKED_BODY_NAMES = [
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "trunk_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_pitch_link",
]

# End-effector links (feet + hands) allowed to make contact.
EE_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]

# Root-velocity spread applied once per episode reset, on top of the reference
# body velocity.  Unchanged from the shared mimic value.
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Interval push disturbance.  Deliberately separate from VELOCITY_RANGE: the two
# used to share one constant, which sized the recurring push off a reset-time
# spread.
#
# This clip holds single support for 6.56 s (reference left foot above 0.30 m,
# frames 178-505), so at the previous 1-3 s interval a lift saw about 3.3
# pushes.  A 57 kg robot with its CoM at 0.95 m has an ankle-strategy capture
# limit of 0.101 m x 3.21 rad/s = 0.32 m/s fore-aft and 0.037 m x 3.21 rad/s =
# 0.12 m/s lateral, from the foot's contact-rail half-extents.  The lateral
# figure drops to about 0.06 m/s once the ankle-roll position-target clip
# (+-0.5236 rad at kp = 20 Nm/rad, so 10.5 Nm, so 1.87 cm of CoP) is taken into
# account.  A +-0.5 m/s lateral push is therefore 4-8x beyond what the support
# ankle can absorb, and the swing-foot reward and termination forbid the step
# that would otherwise recover it.
#
# These ranges sit just inside the fore-aft limit and inside the kp-limited
# lateral one, at an interval that leaves room to settle between pushes.  Widen
# them again only after the lift itself is reliable.
PUSH_VELOCITY_RANGE = {
    "x": (-0.15, 0.15),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
    "roll": (-0.15, 0.15),
    "pitch": (-0.20, 0.20),
    "yaw": (-0.30, 0.30),
}
PUSH_INTERVAL_RANGE_S = (4.0, 8.0)

# Command-delay range in physics steps (sim.dt = 0.005 s), applied to every
# actuator group of this task only. The shared mimic asset ships 0-2 steps
# (0-10 ms), which is well short of the real loop: the houtaitui deployment log
# (ustc-humanoid-identification/results/houtaitui_0813) shows a 12.1 Hz
# whole-body limit cycle that needs roughly 150 deg of loop lag to sustain.
#
# Held back at the shared 0-10 ms while the Phase-1 lift curriculum is still
# failing.  The 0-20 ms widening was introduced in the same change as the lift
# curriculum, the swing-foot termination and the plant randomization, so a
# failed run cannot attribute blame between them.  The policy observes a single
# frame and cannot identify the per-step delay, so widening the range only buys
# a more conservative policy until an observation history exists.  Restore 4
# once the lift is reliable, or once the delay has been measured on hardware.
ACTUATOR_MAX_DELAY = 2


def _with_command_delay(robot_cfg: ArticulationCfg, max_delay: int = ACTUATOR_MAX_DELAY) -> ArticulationCfg:
    """Copy ``robot_cfg`` with every actuator group's command delay widened."""
    cfg = robot_cfg.copy()
    for actuator in cfg.actuators.values():
        actuator.max_delay = max_delay
    return cfg


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        class_type=LocalGridPlaneTerrainImporter,
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = _with_command_delay(ROBOT_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/ustc1_rightstand.npz",
        anchor_body_name=ANCHOR_BODY_NAME,
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
        body_names=TRACKED_BODY_NAMES,
        # The clip is one-shot: it ends standing on both feet, not in a state
        # that loops back to frame 0.  The default "resample" teleports the
        # robot to a freshly sampled reference frame mid-episode without
        # emitting a termination, so GAE bootstraps across the discontinuity.
        # "hold" freezes the final reference frame instead and never rewrites
        # robot state; the paired ``motion_end`` termination closes the episode.
        motion_end_behavior="hold",
    )


@configclass
class StandTransitionCommandsCfg(CommandsCfg):
    """RightStand reference with deployment-ready standing entry and exit."""

    motion = CommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc1_rightstand_stand_transition.npz"
    )


@configclass
class Phase1LiftCommandsCfg(CommandsCfg):
    """Lift curriculum on the 615-frame source clip.

    The source clip naturally spends 53% of its duration above 0.30 m.  The
    targeted reset range therefore covers the take-off transition instead of
    spawning every targeted environment with the leg already high.  Exact
    frame-zero resets retain full stand-to-lift rollouts; the rest remain under
    failure-adaptive sampling.
    """

    motion = CommandsCfg().motion.replace(
        frame_zero_probability=0.20,
        targeted_frame_range=(130, 220),
        targeted_frame_probability=0.40,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=ROBOT_ACTION_SCALE, use_default_offset=True
    )


@configclass
class DeploymentSafeActionsCfg:
    """Position commands constrained exactly like the deployment controller."""

    JointPositionAction = mdp.DeploymentLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=ROBOT_ACTION_SCALE,
        use_default_offset=True,
        clip=DEPLOYMENT_JOINT_POSITION_LIMITS,
        max_target_velocity=DEPLOYMENT_TARGET_VELOCITY,
    )


@configclass
class EmaDeploymentSafeActionsCfg(DeploymentSafeActionsCfg):
    """Deployment-safe commands with the action EMA enabled.

    The 0813 houtaitui log (ustc-humanoid-identification/results/houtaitui_0813)
    shows a 12.1 Hz whole-body limit cycle that the controller sustains by
    injecting about 8 W: 11 of 12 leg joints have a positive band-limited
    <tau * dq>.  The filter does not touch the 10-24 Hz mechanical modes that
    footstrikes ring; it removes the loop gain that keeps re-exciting one of
    them.  At alpha=0.85 the 12 Hz loop gain is about 0.85 while the task band
    below 3 Hz loses less than 0.2 dB.

    The same filter must run in deployment.  Training with it and deploying
    without it (or the reverse) puts the policy on a plant it never saw.
    """

    JointPositionAction = mdp.DeploymentLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=ROBOT_ACTION_SCALE,
        use_default_offset=True,
        clip=DEPLOYMENT_JOINT_POSITION_LIMITS,
        max_target_velocity=DEPLOYMENT_TARGET_VELOCITY,
        ema_alpha=ACTION_EMA_ALPHA,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class DeploymentSafeObservationsCfg(ObservationsCfg):
    """Expose the normalized command that is actually sent to the PD actuator."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        last_action = ObsTerm(
            func=mdp.last_applied_action,
            params={"action_name": "JointPositionAction"},
        )

    @configclass
    class PrivilegedCfg(ObservationsCfg.PrivilegedCfg):
        actions = ObsTerm(
            func=mdp.last_applied_action,
            params={"action_name": "JointPositionAction"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg(PlantRandomizationEventCfg):
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ANCHOR_BODY_NAME),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # Narrowed from the shared 0.6-1.4x while the lift curriculum is still
    # failing.  Rotor inertia dominates the joint-space inertia of the support
    # ankle, so a 40% spread is a large plant change for exactly the joint this
    # motion depends on most.  Restore the shared range from
    # PlantRandomizationEventCfg once the lift is reliable.
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_RANGE_S,
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    hip_yaw_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint"],
            )
        },
    )
    knee_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_knee_joint"],
            )
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    # Every motion_relative_* term is computed against a reference re-anchored
    # to the robot's own horizontal position and yaw each step, so all of them
    # are exactly invariant to a rigid horizontal translation of the robot.
    # motion_global_anchor_pos is the only term that sees the drift at all, and
    # at std=0.3 with weight 0.5 it is worth under 0.02 once the error passes
    # half a metre.  This term restores a usable horizontal gradient; the
    # trunk's own reference travel is only 0.244 m over the whole clip, so
    # std=0.15 is loose relative to the motion but tight relative to drift.
    motion_anchor_xy = RewTerm(
        func=mdp.motion_anchor_xy_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.15},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_base_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.08, "body_names": ["base_link"]},
    )
    motion_base_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.20, "body_names": ["base_link"]},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    motion_left_ankle_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "std": 0.08, "body_names": ["left_ankle_roll_link"]},
    )

    # Targeted single-support and touchdown terms.  These are intentionally
    # separate from generic action-rate/joint-acceleration penalties so the
    # training log exposes whether the lifted-leg transition is actually quiet.
    single_support_stability = RewTerm(
        func=mdp.single_support_stability,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_threshold": 10.0,
            "tilt_scale": 0.20,
            "angular_velocity_scale": 1.5,
        },
    )

    # Phase 1: Direct lift incentives to break the "stand-only" local optimum
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance,
        weight=2.0,
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "reference_height_threshold": 0.30,
            "max_height_error": 0.50,
        },
    )
    swing_foot_contact_penalty = RewTerm(
        func=mdp.swing_foot_contact_penalty,
        weight=-2.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "contact_threshold": 10.0,
            "reference_height_threshold": 0.30,
        },
    )

    feet_impact_velocity = RewTerm(
        func=mdp.feet_impact_velocity,
        weight=-0.2,  # Reduced from -1.0 during phase 1: let it learn to lift first
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_pitch_link$)(?!right_wrist_pitch_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class EmaRewardsCfg(RewardsCfg):
    """EMA variant with phase 1 lift incentives.

    Phase 1 curriculum: strong direct lift rewards + weak landing penalty.
    Once the policy lifts consistently (timeout rate > 85%, swing_foot_clearance > 0.3),
    transition to phase 2 by reducing swing weights and increasing landing penalty.
    """

    # Inherit all base rewards including the new swing terms
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.55},
    )
    # anchor_pos only checks z, so before this term nothing bounded horizontal
    # drift.  The trunk reference travels 0.244 m horizontally over the clip, so
    # 0.35 m cannot fire on legitimate tracking.
    anchor_pos_xy = DoneTerm(
        func=mdp.bad_anchor_pos_xy,
        params={"command_name": "motion", "threshold": 0.35},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.55,
            "body_names": EE_BODY_NAMES,
        },
    )
    # First rung of the lift curriculum.  At 0.25 this fires across frames
    # 179-504 (6.52 s) for a foot left on the floor, so it ends essentially
    # every episode before the policy has learned to lift at all.  At 0.45 the
    # same foot only trips it over frames 194-232 (0.78 s) around the 0.542 m
    # reference peak: the stand-only optimum is still closed off, but the rest
    # of the single-support hold is no longer a termination.  Tighten toward
    # 0.25 once Episode_Termination/swing_foot_height has fallen and
    # Episode_Reward/swing_foot_clearance is rising.
    swing_foot_height = DoneTerm(
        func=mdp.bad_swing_foot_height,
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link"],
            "reference_height_threshold": 0.30,
            "max_height_shortfall": 0.45,
        },
    )
    # The clip is one-shot and motion_end_behavior is "hold", so the episode has
    # to end when the reference does.  time_out=True keeps it bootstrapped
    # rather than treated as a failure.
    motion_end = DoneTerm(
        func=mdp.motion_clip_finished,
        params={"command_name": "motion"},
        time_out=True,
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the humanoid_ultra 27dof motion-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class RobotDeploySafeEnvCfg(RobotEnvCfg):
    """RightStand training with the real-controller position and target-rate boundary."""

    actions: DeploymentSafeActionsCfg = DeploymentSafeActionsCfg()
    observations: DeploymentSafeObservationsCfg = DeploymentSafeObservationsCfg()


@configclass
class RobotHoutaituiEnvCfg(RobotDeploySafeEnvCfg):
    """Phase-1 houtaitui curriculum that first learns a real foot lift."""

    commands: Phase1LiftCommandsCfg = Phase1LiftCommandsCfg()


@configclass
class RobotHoutaituiEmaEnvCfg(RobotHoutaituiEnvCfg):
    """Houtaitui with the action EMA, aimed at the 12.1 Hz deployment limit cycle."""

    actions: EmaDeploymentSafeActionsCfg = EmaDeploymentSafeActionsCfg()
    rewards: EmaRewardsCfg = EmaRewardsCfg()


class RobotHoutaituiEmaPlayEnvCfg(RobotHoutaituiEmaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # Keep displaying the held final frame instead of resetting.
        self.terminations.motion_end = None


class RobotHoutaituiPlayEnvCfg(RobotHoutaituiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # Keep displaying the held final frame instead of resetting.
        self.terminations.motion_end = None


@configclass
class RobotHoutaituiLeftArm2P5kgEnvCfg(RobotHoutaituiEnvCfg):
    """Houtaitui tracking with the identified 2.5 kg left-arm payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _with_command_delay(
            HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")


class RobotHoutaituiLeftArm2P5kgPlayEnvCfg(RobotHoutaituiPlayEnvCfg):
    """Play configuration for the 2.5 kg left-arm payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _with_command_delay(
            HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")
