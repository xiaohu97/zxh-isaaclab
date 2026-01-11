from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat

# 引入你的机器人配置
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# ==============================================================================
# 1. 自定义 Command 类：生成高度和姿态目标
# ==============================================================================

@dataclass
class HeightPostureCommandCfg(CommandTermCfg):
    """Configuration for the height and posture command generator."""
    class_type: type = None # Will be set below
    
    # [Fix 1] 必须显式定义 asset_name，否则传入 asset_name="robot" 会报错
    asset_name: str = "robot" 
    
    # [min, max] ranges
    height_range: tuple[float, float] = (0.65, 1.2) # 目标高度范围 (米)
    pitch_range: tuple[float, float] = (-0.3, 0.3)  # 目标俯仰角范围 (弧度)
    roll_range: tuple[float, float] = (-0.2, 0.2)   # 目标侧倾角范围 (弧度)
    
    resampling_time_range: tuple[float, float] = (2.5, 5.0) # 每隔多久换一次目标
    debug_vis: bool = True

# ==============================================================================
# 1. 修正后的 Command 类
# ==============================================================================

class HeightPostureCommand(CommandTerm):
    """Generates commands for height, pitch, and roll.
    Command vector shape: (num_envs, 3) -> [height, pitch, roll]
    """
    cfg: HeightPostureCommandCfg

    def __init__(self, cfg: HeightPostureCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # [修改 1] 使用内部变量 _command 存储数据，避免与 @property command 冲突
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        # 用于存储 metrics 的字典
        self.metrics = {}

    # [修改 2] 必须实现 command 属性
    @property
    def command(self) -> torch.Tensor:
        """The current command tensor."""
        return self._command

    def _resample_command(self, env_ids: torch.Tensor):
        # Index 0: Height
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.height_range)
        # Index 1: Pitch
        self._command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.pitch_range)
        # Index 2: Roll
        self._command[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.roll_range)

    def _update_command(self):
        # 在部署/Play时，这里可以接入 Joystick 映射逻辑
        pass 

    # [修改 3] 必须实现 _update_metrics 方法
    def _update_metrics(self):
        """Update metrics for logging."""
        # ✅ 正确写法：直接存储所有环境的目标高度 (Vector)，不要加 torch.mean()
        self.metrics["height_target"] = self._command[:, 0]
        
        # 如果你想记录 Pitch 或 Roll，也一样：
        # self.metrics["pitch_target"] = self._command[:, 1]

# 记得绑定实现类
HeightPostureCommandCfg.class_type = HeightPostureCommand


# ==============================================================================
# 2. 自定义 Reward 函数
# ==============================================================================

def track_height_command(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for tracking the target height."""
    commands = env.command_manager.get_command(command_name)
    target_h = commands[:, 0]
    
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    
    error = torch.square(target_h - curr_h)
    return torch.exp(-error / 0.04)

def track_posture_command(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for tracking target pitch and roll."""
    commands = env.command_manager.get_command(command_name)
    target_pitch = commands[:, 1]
    target_roll = commands[:, 2]
    
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    error = torch.square(target_pitch - pitch) + torch.square(target_roll - roll)
    return torch.exp(-error / 0.05)

def penalize_horizontal_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize XY velocity to encourage standing in place."""
    asset = env.scene[asset_cfg.name]
    vel_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return -torch.square(vel_xy)

def penalize_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize Yaw rate to prevent spinning."""
    asset = env.scene[asset_cfg.name]
    return -torch.square(asset.data.root_ang_vel_w[:, 2])

# ==============================================================================
# 3. 环境配置
# ==============================================================================

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=0, 
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        # [Fix 2] 之前这里有语法错误，现在补全了
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class EventCfg:
    """Configuration for events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-3.0, 3.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_posture = HeightPostureCommandCfg(
        asset_name="robot", # 现在 HeightPostureCommandCfg 有这个字段了，不会报错
        class_type=HeightPostureCommand,  # ✅ 手动修复这里的 None
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        height_range=(0.55, 0.78), 
        pitch_range=(-0.25, 0.25),
        roll_range=(-0.15, 0.15),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_posture"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_posture"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    track_height = RewTerm(
        func=track_height_command,
        weight=1.0,
        params={"command_name": "base_posture", "asset_cfg": SceneEntityCfg("robot")},
    )
    track_posture = RewTerm(
        func=track_posture_command,
        weight=4.0,
        params={"command_name": "base_posture", "asset_cfg": SceneEntityCfg("robot")},
    )
    stand_still_xy = RewTerm(
        func=penalize_horizontal_velocity,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_still_yaw = RewTerm(
        func=penalize_yaw_rate,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"]),
        },
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_posture"
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the posture control environment."""
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        }