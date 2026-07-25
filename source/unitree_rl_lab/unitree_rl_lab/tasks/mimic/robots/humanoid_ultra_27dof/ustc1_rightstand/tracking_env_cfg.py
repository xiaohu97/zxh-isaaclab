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

from unitree_rl_lab.assets.robots.humanoid_ultra import HUMANOIDULTRA27DOF_IDENTIFIED_CFG as ROBOT_CFG

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

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
    )


@configclass
class StandTransitionCommandsCfg(CommandsCfg):
    """RightStand reference with deployment-ready standing entry and exit."""

    motion = CommandsCfg().motion.replace(
        motion_file=f"{os.path.dirname(__file__)}/ustc1_rightstand_stand_transition.npz"
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
class EventCfg:
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

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
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
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
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
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.55},
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
    """Deployment-safe houtaitui with standing entry and recovery segments."""

    commands: StandTransitionCommandsCfg = StandTransitionCommandsCfg()


class RobotHoutaituiPlayEnvCfg(RobotHoutaituiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
