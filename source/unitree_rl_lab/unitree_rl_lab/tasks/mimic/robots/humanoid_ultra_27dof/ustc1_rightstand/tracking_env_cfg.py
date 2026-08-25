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
# These are command limits, not the articulation's physical joint limits -- but
# they are intersected with them, because a target outside the mechanical range
# is dead action budget: the joint stops at the URDF limit and the remaining
# command only presses into the hard stop.
#
# Six entries were outside ``humanoid_ultra_27dof_description_identified.urdf``
# and have been clipped to it:
#
#   *_ankle_roll_joint    +-0.5236 -> +-0.45     (0.0736 rad, 16% of the range)
#   left_shoulder_yaw       +2.5   -> +1.623156
#   right_shoulder_yaw      -2.5   -> -1.623156
#   left_wrist_yaw          -2.5   -> -1.1519173
#   right_wrist_yaw         +2.5   -> +1.1519173
#
# The ankle-roll entry is the one that changes the physics reasoning in this
# file: the single-support CoP budget below was derived from kp * 0.5236 =
# 10.47 Nm, but only kp * 0.45 = 9.0 Nm is reachable through free motion.
# Isaac clips resets and the joint-limit penalty at ``0.9 *`` the URDF value
# (+-0.405 rad for ankle roll), which is tighter still.
#
# Hip and knee entries differ from the URDF by <= 0.001 rad (1.5708 vs 1.57,
# 2.356 vs 2.36).  That is rounding, not a reachability problem, and is left
# alone so the table still reads as the controller's own numbers.
#
# ustc1_pick/tracking_env_cfg.py carries its own copy of this table with the
# same six mismatches; it has not been touched.
DEPLOYMENT_JOINT_POSITION_LIMITS = {
    "left_hip_roll_joint": (-0.25, 1.5708),
    "left_hip_yaw_joint": (-1.5708, 1.5708),
    "left_hip_pitch_joint": (-1.5708, 1.5708),
    "left_knee_joint": (0.0, 2.356),
    "left_ankle_pitch_joint": (-0.7, 0.95),
    "left_ankle_roll_joint": (-0.45, 0.45),
    "right_hip_roll_joint": (-1.5708, 0.25),
    "right_hip_yaw_joint": (-1.5708, 1.5708),
    "right_hip_pitch_joint": (-1.5708, 1.5708),
    "right_knee_joint": (0.0, 2.356),
    "right_ankle_pitch_joint": (-0.7, 0.95),
    "right_ankle_roll_joint": (-0.45, 0.45),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-2.4, 1.2),
    "left_shoulder_roll_joint": (-0.3, 2.7),
    "left_shoulder_yaw_joint": (-2.5, 1.623156),
    "left_elbow_joint": (-2.17, 0.0),
    "left_wrist_yaw_joint": (-1.1519173, 2.5),
    "left_wrist_roll_joint": (-1.11, 1.11),
    "left_wrist_pitch_joint": (-1.05, 1.05),
    "right_shoulder_pitch_joint": (-1.2, 2.4),
    "right_shoulder_roll_joint": (-2.7, 0.3),
    "right_shoulder_yaw_joint": (-1.623156, 2.5),
    "right_elbow_joint": (0.0, 2.17),
    "right_wrist_yaw_joint": (-2.5, 1.1519173),
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
# body velocity.
#
# REVERTED to +-0.5 m/s on 2026-08-19.  The +-0.30 experiment ran from
# 2026-08-18_11-14-52 and has to be undone: it improved every number Isaac
# logs and made the policy materially worse.
#
# The narrowing was argued from a braking-distance estimate, and the run's own
# telemetry appeared to confirm it -- error_anchor_pos fell 0.2546 -> 0.2140 and
# the anchor_pos_xy termination share fell 52% -> 39%.  Both readings were
# artifacts:
#
#   * ``Metrics/motion/error_*`` is the error at termination, not an episode
#     mean (see compare_mimic_runs.py), so it moves with the termination mix.
#   * The two runs were measured under their own reset distributions.  A policy
#     handed a smaller impulse trips a fixed 0.35 m drift threshold less often
#     no matter how well it brakes.
#
# Rolling both checkpoints out in MuJoCo under one fixed condition inverts the
# ranking.  At the +-0.30 reset the +-0.30-trained policy is the worse one, and
# it degrades monotonically as it trains:
#
#   iter            10k    15k    20k    24k    30k    36k    42k
#   +-0.5 drift    0.166  0.174  0.163  0.254    -      -      -     median [m]
#   +-0.5 falls      1      1      2      1      -      -      -     of 30
#   +-0.3 drift    0.139  0.166  0.269  0.383  0.313  0.557  0.454
#   +-0.3 falls      1      4      8     10      6     11      9
#
# Same seed (42), same everything else, so this is the reset spread and not
# run-to-run noise.  The impulse the term was removing is what taught the
# policy to null drift at all; without it, drift rejection decays with
# training.  Keep the 0.35 m anchor_pos_xy threshold: under +-0.5 it fires on
# real failures, which is the point.
VELOCITY_RANGE = {
    "x": (-0.50, 0.50),
    "y": (-0.50, 0.50),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Interval push disturbance.  Deliberately separate from VELOCITY_RANGE: the two
# used to share one constant, which sized the recurring push off a reset-time
# spread.
#
# A 57 kg robot with its CoM at 0.95 m has an ankle-strategy capture limit of
# 0.32 m/s fore-aft and 0.12 m/s lateral from the foot's contact-rail
# half-extents (0.101 m and 0.037 m, times sqrt(g/h) = 3.21 rad/s).  The
# lateral figure drops to about 0.05 m/s once the ankle-roll position-target
# clip is taken into account: the clip is now +-0.45 rad (the URDF stop; it
# used to read +-0.5236, which the joint cannot reach), so at kp = 20 Nm/rad
# the joint caps at 9.0 Nm, which is only 1.61 cm of CoP travel against the
# foot's 3.7 cm, i.e. 0.166 m/s^2 of horizontal deceleration.  The
# original +-0.5 m/s every 1-3 s was therefore 4-8x beyond what the support
# ankle can absorb, while the swing-foot reward and termination forbid the step
# that would otherwise recover it.
#
# Both horizontal axes carry the lateral limit rather than one each.
# push_by_setting_velocity works on root_vel_w, i.e. WORLD axes, but this
# reference faces world -Y (anchor yaw runs -96.6 deg to -52.6 deg), so world X
# is the robot's lateral direction here and the mapping rotates by 44 deg over
# the clip.  Splitting the budget per world axis silently assigns the loose
# fore-aft bound to whichever axis happens to be lateral.  mdp.
# phase_targeted_velocity_push resolves its ranges in the heading frame and
# would allow per-direction budgets, but it fires at most one kick per episode
# and cannot replace a recurring interval push.
PUSH_VELOCITY_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
    "roll": (-0.15, 0.15),
    "pitch": (-0.15, 0.15),
    "yaw": (-0.30, 0.30),
}
PUSH_INTERVAL_RANGE_S = (4.0, 8.0)

# Command-delay range in physics steps (sim.dt = 0.005 s), applied to every
# actuator group of this task only. The shared mimic asset ships 0-2 steps
# (0-10 ms), which is well short of the real loop: the houtaitui deployment log
# (ustc-humanoid-identification/results/houtaitui_0813) shows a 12.1 Hz
# whole-body limit cycle that needs roughly 150 deg of loop lag to sustain.
#
# 2026-08-25, reverted to the shared 0-10 ms the same day it was widened.
#
# It was set to 7-10 steps and then 5-7 on the reading that houtaitui2.csv had
# measured a 42.5 ms transport delay.  That reading does not hold up:
#
#   * The number came from cross-correlating targetPos against measured
#     position at 12.2 Hz and converting 187 deg to time as though the path
#     were a pure delay.  It is not.  targetPos is the policy's response to the
#     measured state, so command and motion are mutually caused, and a PD loop
#     plus plant already approaches -180 deg on its own well above its natural
#     frequency.  Near-antiphase says the loop cannot track 12 Hz; it does not
#     say where the phase comes from.
#   * One common delay tau produces the same lag 2*pi*f*tau at a given
#     frequency for every joint.  The measured lags were 11, -11, 66, 187, -187
#     and -231 deg.  That spread rules out a single fixed delay -- and the
#     write-up that set this constant explained the sign flips away as period
#     wrapping while passing over the two joints sitting at +-11 deg.
#   * Loop instrumentation the same day (sim2real_humanoidultra27dof_walk_TEST.py
#     -> houtaitui.csv) puts the entire in-process path at about 1.3 ms: leg
#     state age 0.53 ms median, inference 0.29, publish 0.32.  A 42 ms transport
#     delay would have to sit entirely in a bus whose own overrun statistics
#     quote 1.1 ms cycles.
#
# What survives: the 12.2 Hz tone is present in targetPos itself, so the
# oscillation is generated inside the control loop rather than excited
# mechanically, and the loop is near-antiphase there.  The cause is open, and
# the MuJoCo scores that made delay look dominant (C@42000 falling 100/100 at a
# 40 ms command delay) were measuring an assumed condition, not a measured one.
#
# Put a measured value here only after an open-loop test: drive one joint with
# a swept sine, robot unloaded and kp low, and fit phase against frequency.  A
# transport delay gives phase linear in f with the same slope on every joint;
# PD dynamics do not.  A single-frequency closed-loop number cannot separate
# them.
#
# 1-3 steps is a deliberately modest standing assumption rather than a measured
# value.  Zero would be wrong in the other direction: the loop instrumentation
# puts the in-process path alone at ~1.3 ms median (leg state age 0.53,
# inference 0.29, publish 0.32) and the driver side has never been measured, so
# some latency certainly exists.  5-15 ms brackets that without asserting a
# figure the data does not support.
ACTUATOR_MIN_DELAY = 1   # 5 ms
ACTUATOR_MAX_DELAY = 3   # 15 ms, mean 10 ms -- an assumption, not a measurement


def _with_command_delay(
    robot_cfg: ArticulationCfg,
    min_delay: int = ACTUATOR_MIN_DELAY,
    max_delay: int = ACTUATOR_MAX_DELAY,
) -> ArticulationCfg:
    """Copy ``robot_cfg`` with every actuator group's command delay retimed.

    Both ends are set: the delay is resampled per episode from
    ``[min_delay, max_delay]``, so leaving ``min_delay`` at 0 would halve the
    mean and miss the measured value.
    """
    cfg = robot_cfg.copy()
    for actuator in cfg.actuators.values():
        actuator.min_delay = min_delay
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
        motion_file=f"{os.path.dirname(__file__)}/ustc1_rightstand_stand_transition.npz",
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
    """Lift curriculum on the 941-frame stand-transition clip.

    The targeted reset range covers the take-off transition instead of spawning
    every targeted environment with the leg already high.  Exact frame-zero
    resets retain full stand-to-lift rollouts; the rest remain under
    failure-adaptive sampling.

    The range is clip-specific and must be re-derived whenever ``motion_file``
    changes.  On this clip the reference left foot leaves the floor at frame
    453, crosses 0.30 m at frame 468 and peaks at 0.542 m at frame 478, so the
    window spans take-off through the early hold and is 53% single support.
    The previous (130, 220) was derived from the 615-frame ustc1_rightstand
    clip; on this clip those frames are the standing hold, where the foot never
    leaves 0.060-0.082 m, so 40% of resets would have bought no lift practice
    at all.
    """

    motion = CommandsCfg().motion.replace(
        frame_zero_probability=0.20,
        targeted_frame_range=(420, 494),
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
    # Zeroed as a decisive experiment, not as tuning.  This term scores
    # exp(-(tilt/0.20)^2 - (w/1.5)^2) against VERTICAL, where tilt is the root's
    # own projected gravity.  But the reference itself holds base_link at a
    # median 35.3 deg lean through single support (it is a rear-leg lift; the
    # torso counterbalances the raised leg), so a policy that reproduced the
    # reference exactly would score an Episode_Reward of 9.65e-5.
    #
    # The 2026-08-19 runs logged 0.000889, i.e. 9.2x what perfect tracking pays.
    # The only way to collect that is to hold the pelvis more upright than the
    # reference: inverting the score gives roughly 19 deg against the
    # reference's 35 deg.  Measured error_body_rot over the same window is
    # 0.2660 rad = 15.2 deg, consistent with this term buying its own reward
    # with attitude-tracking error.
    #
    # That inversion assumes ~83 single-support steps per episode and compares
    # against a 14-body rotation average, so it is a hypothesis, not a result.
    # Zeroing the weight is the cheap test: if error_body_rot drops materially,
    # the term was distorting the motion and should be re-specified as a
    # deviation from the REFERENCE attitude (quat_error against
    # body_quat_relative_w) rather than from vertical.  If error_body_rot does
    # not move, restore 0.25 and look elsewhere.
    single_support_stability = RewTerm(
        func=mdp.single_support_stability,
        weight=0.0,
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
    # This is the only term that requires the lift to be *held*.  On this clip
    # a foot left on the floor trips it across frames 469-524 at 0.25 (1.12 s,
    # the whole reference lift) but only over frames 476-489 at 0.45 (0.28 s).
    #
    # 0.45 was tried as a curriculum first rung and has to stay reverted: it
    # interacts badly with motion_end.  Reaching the final frame is a
    # bootstrapped time_out, i.e. a successful terminal state, so "lift briefly
    # around the peak, put the foot back down, coast to the end" becomes a
    # cheap high-value path.  That is exactly the stand-only optimum the
    # Phase-1 curriculum exists to close, and it did not exist before
    # motion_end was added.
    #
    # 0.25 is comfortably satisfiable here: the reference only asks for a
    # 1.16 s hold, and under PUSH_VELOCITY_RANGE every push inside it is within
    # the support ankle's capture limit.
    swing_foot_height = DoneTerm(
        func=mdp.bad_swing_foot_height,
        params={
            "command_name": "motion",
            "body_names": ["left_ankle_roll_link"],
            "reference_height_threshold": 0.30,
            "max_height_shortfall": 0.25,
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


##
# Yaw-guarded variant
##


@configclass
class YawGuardedRewardsCfg(RewardsCfg):
    """Unchanged from the base rewards.  The std widening was tried and reverted.

    The first attempt raised ``motion_global_anchor_ori`` from std 0.3 to 0.6 on
    the argument that at 0.3 the term is dead where it is needed:
    ``exp(-0.96^2 / 0.3^2) = 4e-5`` at the 55 degrees a rollout actually
    reaches, i.e. no reward and no gradient.  That half was right, but widening
    the same Gaussian is the wrong instrument, because it also flattens the
    reward near zero -- the gradient ``-2e/std^2`` is 4x weaker at small error
    with std 0.6.  The policy traded away the tightness it had in the normal
    regime to buy gradient in a regime it visits rarely.

    Measured at iteration 36000, 30 seeds, reset +-0.5, both variants warm
    started from the same checkpoint:

                    drift median   falls/30   completes/30   yaw median   yaw p90
      std 0.3 (A)      0.196 m         1           29           21.5 deg   52.2 deg
      std 0.6 (B)      0.445 m         5           25           32.8 deg   43.5 deg

    Exactly the predicted shape: p90 yaw improved 52.2 -> 43.5 deg, the far
    field the change was aimed at, while median yaw got *worse* 21.5 -> 32.8
    deg and drift and falls degraded with it.  The far-field gain did not pay
    for the near-field loss.  Getting both would need a second, wider term
    added alongside std 0.3 rather than one term stretched to cover both --
    untried.
    """

    pass


@configclass
class YawGuardedTerminationsCfg(TerminationsCfg):
    """Close the yaw loophole that no existing termination could see.

    ``anchor_ori`` compares projected-gravity z, which is tilt and is blind to
    rotation about vertical; ``anchor_pos_xy`` sees translation; the swing-foot
    and end-effector terms are z-only.  The threshold is measured, not guessed,
    and was re-measured against the current policy once the reset spread went
    back to +-0.5.  Over 30 perturbed rollouts of the iteration-36000 checkpoint
    the yaw error stays under 0.311 rad through the entire lift and reaches a
    median of 0.443 rad (25.4 deg) once the leg starts coming down.

    0.7 rad was the first setting, sized off the +-0.3 run, and it was too far
    out to do any work: it accounted for 0.9% of terminations across 12000
    iterations, because half the episodes are already ended by anchor_pos_xy
    before the descent begins.  0.55 rad keeps a 1.8x margin over the worst
    legitimate lift-phase error and fires on 43% of the descents instead of
    33%.
    """

    anchor_yaw = DoneTerm(
        func=mdp.bad_anchor_yaw,
        params={"command_name": "motion", "threshold": 0.55},
    )


@configclass
class RobotHoutaituiYawEnvCfg(RobotHoutaituiEnvCfg):
    """Houtaitui plus the yaw termination, and nothing else.

    Now exactly one term apart from the plain houtaitui task, so the pair
    answers a single question: does making the twist a terminable failure help?
    The first version of this config also widened the anchor-orientation reward
    std, which turned out to be the dominant effect and a harmful one -- see
    YawGuardedRewardsCfg for the measurement that reverted it.
    """

    rewards: YawGuardedRewardsCfg = YawGuardedRewardsCfg()
    terminations: YawGuardedTerminationsCfg = YawGuardedTerminationsCfg()


class RobotHoutaituiYawPlayEnvCfg(RobotHoutaituiYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # Keep displaying the held final frame instead of resetting.
        self.terminations.motion_end = None


##
# Yaw termination + arm balance variant
##


@configclass
class ArmBalanceRewardsCfg(RewardsCfg):
    """Pay the arms to keep the reference's counter-swing during the lift.

    The reference clip cannot be executed by an ankle strategy alone: its own
    lateral CoM acceleration in single support averages 0.30 m/s^2 and peaks at
    0.95, which needs 15.3 / 48.6 Nm against a support ankle whose commandable
    span is about 21 Nm.  The motion is only feasible with angular-momentum
    (hip/arm) control, and the reference supplies it through the arms.

    The policy currently does not: measured over 20 perturbed rollouts of
    A@36000, the six arm bodies sit 0.027 m from the reference for the whole
    clip but open up to 0.152 m median (0.215 p90) exactly across the lift
    window.  So the arms abandon the reference precisely when their counter-
    swing is needed, and the angular momentum is absorbed by the torso and
    support hip instead -- which is the twist that ``anchor_yaw`` terminates,
    and, once that route is closed, shows up as horizontal drift.

    std = 0.20 puts the current 0.152 m operating point essentially on the
    Gaussian's steepest gradient (max at e = std/sqrt(2) = 0.141), which is the
    property the reverted std 0.6 experiment showed matters most: a term whose
    operating point sits in a saturated tail has no reward *and* no gradient.
    The whole-clip part of this term is nearly constant at 0.98 and therefore
    contributes nothing to the policy gradient; the lift window is where it
    varies and where its leverage is.
    """

    motion_arm_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.5,
        params={
            "command_name": "motion",
            "std": 0.20,
            "body_names": [
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_pitch_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_pitch_link",
            ],
        },
    )


@configclass
class RobotHoutaituiYawArmEnvCfg(RobotHoutaituiEnvCfg):
    """Give the swing leg's angular momentum a legal outlet, then close the twist.

    ``anchor_yaw`` alone was tested and is not enough: across five checkpoints
    it pinned the twist at 20-28 degrees, where the unconstrained line ranged
    18-61, but bought no robustness -- drift stayed at 0.30 and the fall rate
    swung between 12% and 87% on checkpoints 1000 iterations apart.  Blocking
    the torso route without opening another one leaves the momentum nowhere to
    go but horizontal translation.
    """

    rewards: ArmBalanceRewardsCfg = ArmBalanceRewardsCfg()
    terminations: YawGuardedTerminationsCfg = YawGuardedTerminationsCfg()


class RobotHoutaituiYawArmPlayEnvCfg(RobotHoutaituiYawArmEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # Keep displaying the held final frame instead of resetting.
        self.terminations.motion_end = None


@configclass
class RobotHoutaituiYawArmLeftArm2P5kgEnvCfg(RobotHoutaituiYawArmEnvCfg):
    """The yaw+arm recipe carrying the identified 2.5 kg left-arm payload.

    Same clip as the unloaded variant, so the two measured constants transfer
    unchanged: the ``anchor_yaw`` threshold was sized against this clip's lift
    window and the ``motion_arm_pos`` std against this clip's arm excursion.
    The payload only changes the plant.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _with_command_delay(
            HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")


class RobotHoutaituiYawArmLeftArm2P5kgPlayEnvCfg(RobotHoutaituiYawArmPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _with_command_delay(
            HUMANOIDULTRA27DOF_MIMIC_LEFTARM2P5KG_CFG
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")
