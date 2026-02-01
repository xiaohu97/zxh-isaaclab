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
# 1. 自定义 Command 类：生成高度和姿态目标（使用偏移量设计）
# ==============================================================================
# 
# 命令设计说明：
# - 命令向量 [delta_height, delta_pitch, delta_roll] 
# - 实际目标 = 默认值 + 命令偏移
# - 当命令为 0 时，机器人保持默认站立姿态
# - 这样部署时手柄归零，机器人就站在默认高度

@dataclass
class HeightPostureCommandCfg(CommandTermCfg):
    """Configuration for the height and posture command generator.
    
    使用偏移量设计：command = 0 表示默认姿态
    - height_cmd = 0  -> 目标高度 = default_height (0.78m)
    - pitch_cmd = 0   -> 目标俯仰 = 0 rad
    - roll_cmd = 0    -> 目标侧倾 = 0 rad
    """
    class_type: type = None  # Will be set below
    
    asset_name: str = "robot" 
    
    # 默认站立参数
    default_height: float = 0.78          # 默认站立高度 (米)
    default_pitch: float = 0.0            # 默认俯仰角 (弧度)
    default_roll: float = 0.0             # 默认侧倾角 (弧度)
    
    # 命令偏移范围 (在默认值基础上的变化量)
    height_delta_range: tuple[float, float] = (-0.15, 0.05)  # 高度变化: 0.63m ~ 0.83m
    pitch_delta_range: tuple[float, float] = (-0.2, 0.2)     # 俯仰变化: ±0.2 rad
    roll_delta_range: tuple[float, float] = (-0.15, 0.15)    # 侧倾变化: ±0.15 rad
    
    # 训练时保持默认姿态的环境比例（用于学习稳定站立）
    rel_default_envs: float = 0.3         # 30% 的环境保持默认姿态
    
    resampling_time_range: tuple[float, float] = (3.0, 6.0)  # 重采样时间
    debug_vis: bool = True


class HeightPostureCommand(CommandTerm):
    """Generates commands for height, pitch, and roll using offset design.
    
    Command vector shape: (num_envs, 3) -> [delta_height, delta_pitch, delta_roll]
    
    实际目标计算:
    - target_height = default_height + delta_height
    - target_pitch = default_pitch + delta_pitch  
    - target_roll = default_roll + delta_roll
    
    部署时命令为0，机器人保持默认站姿。
    """
    cfg: HeightPostureCommandCfg

    def __init__(self, cfg: HeightPostureCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 命令向量存储偏移量 [delta_h, delta_pitch, delta_roll]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics = {}

    @property
    def command(self) -> torch.Tensor:
        """The current command tensor (offset values)."""
        return self._command

    def get_target_height(self) -> torch.Tensor:
        """获取实际目标高度 = 默认高度 + 命令偏移"""
        return self.cfg.default_height + self._command[:, 0]

    def get_target_pitch(self) -> torch.Tensor:
        """获取实际目标俯仰角 = 默认俯仰 + 命令偏移"""
        return self.cfg.default_pitch + self._command[:, 1]

    def get_target_roll(self) -> torch.Tensor:
        """获取实际目标侧倾角 = 默认侧倾 + 命令偏移"""
        return self.cfg.default_roll + self._command[:, 2]

    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        
        # 决定哪些环境保持默认姿态（命令为0）
        num_default = int(n * self.cfg.rel_default_envs)
        default_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        if num_default > 0:
            default_indices = torch.randperm(n, device=self.device)[:num_default]
            default_mask[default_indices] = True
        
        # 采样偏移量
        delta_h = torch.empty(n, device=self.device).uniform_(*self.cfg.height_delta_range)
        delta_pitch = torch.empty(n, device=self.device).uniform_(*self.cfg.pitch_delta_range)
        delta_roll = torch.empty(n, device=self.device).uniform_(*self.cfg.roll_delta_range)
        
        # 默认姿态环境的偏移量设为0
        delta_h[default_mask] = 0.0
        delta_pitch[default_mask] = 0.0
        delta_roll[default_mask] = 0.0
        
        self._command[env_ids, 0] = delta_h
        self._command[env_ids, 1] = delta_pitch
        self._command[env_ids, 2] = delta_roll

    def _update_command(self):
        # 部署时可以接入手柄映射
        pass 

    def _update_metrics(self):
        """Update metrics for logging."""
        self.metrics["height_delta"] = self._command[:, 0]
        self.metrics["target_height"] = self.get_target_height()


# 绑定实现类
HeightPostureCommandCfg.class_type = HeightPostureCommand


# ==============================================================================
# 2. 自定义 Reward 函数（使用偏移量设计）
# ==============================================================================

def track_height_command(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for tracking the target height.
    
    目标高度 = default_height + command[0]
    """
    cmd_term = env.command_manager.get_term(command_name)
    target_h = cmd_term.get_target_height()
    
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    
    error = torch.square(target_h - curr_h)
    return torch.exp(-error / 0.02)  # 更严格的高度跟踪


def track_posture_command(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for tracking target pitch and roll.
    
    目标姿态 = default + command偏移
    """
    cmd_term = env.command_manager.get_term(command_name)
    target_pitch = cmd_term.get_target_pitch()
    target_roll = cmd_term.get_target_roll()
    
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    error = torch.square(target_pitch - pitch) + torch.square(target_roll - roll)
    return torch.exp(-error / 0.03)  # 更严格的姿态跟踪


def penalize_horizontal_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize XY velocity to encourage standing in place."""
    asset = env.scene[asset_cfg.name]
    vel_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return -torch.square(vel_xy)


def penalize_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize Yaw rate to prevent spinning."""
    asset = env.scene[asset_cfg.name]
    return -torch.square(asset.data.root_ang_vel_w[:, 2])


def feet_flat_orientation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping both feet in contact with ground (stable standing)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 检测两脚都着地
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # Z方向力
    in_contact = forces > 1.0  # 接触阈值
    both_feet_contact = torch.all(in_contact, dim=1).float()
    return both_feet_contact

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
    """Configuration for events - 增强抗扰动训练."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 5.0),  # 更大的质量变化范围
            "operation": "add",
        },
    )
    # 重置时施加随机扰动力
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-10.0, 10.0),   # 增大扰动力
            "torque_range": (-5.0, 5.0),    # 增大扰动力矩
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
    # 频繁的推力扰动，增强抗扰动能力
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),  # 更频繁的推力
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 更强的推力
    )
    # 额外的持续外力扰动
    continuous_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-15.0, 15.0),
            "torque_range": (-8.0, 8.0),
        },
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP.
    
    使用偏移量设计：
    - 命令 = [delta_height, delta_pitch, delta_roll]
    - 当命令全为0时，机器人保持默认站姿 (高度0.78m, 姿态水平)
    """
    base_posture = HeightPostureCommandCfg(
        asset_name="robot",
        class_type=HeightPostureCommand,
        resampling_time_range=(4.0, 8.0),  # 较长的采样周期，让机器人有时间稳定
        debug_vis=True,
        # 默认站立参数
        default_height=0.78,
        default_pitch=0.0,
        default_roll=0.0,
        # 偏移量范围（训练时探索不同高度/姿态）
        height_delta_range=(-0.12, 0.02),  # 实际高度范围: 0.66m ~ 0.80m
        pitch_delta_range=(-0.15, 0.15),   # 俯仰变化: ±0.15 rad ≈ ±8.6°
        roll_delta_range=(-0.1, 0.1),      # 侧倾变化: ±0.1 rad ≈ ±5.7°
        rel_default_envs=0.4,              # 40%环境保持默认姿态（命令=0）
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    
    重要：必须使用单一的 action 配置，保持与部署代码一致的关节顺序！
    通过 joint_names=[".*"] 匹配所有关节，顺序由 URDF 决定。
    腰部和手臂的约束通过奖励函数实现，而不是分离动作。
    """
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # 匹配所有29个关节，保持原始顺序
        scale=0.25,
        use_default_offset=True
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
    """Reward terms for the MDP.
    
    奖励设计：
    1. 高度/姿态跟踪（命令=0时保持默认站姿）
    2. 抗扰动（原地站稳、双脚着地）
    3. 腰部和手臂保持默认位置
    """
    # === 主要任务奖励 ===
    track_height = RewTerm(
        func=track_height_command,
        weight=2.0,  # 高度跟踪权重
        params={"command_name": "base_posture", "asset_cfg": SceneEntityCfg("robot")},
    )
    track_posture = RewTerm(
        func=track_posture_command,
        weight=3.0,  # 姿态跟踪权重
        params={"command_name": "base_posture", "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 站稳奖励（抗扰动）===
    stand_still_xy = RewTerm(
        func=penalize_horizontal_velocity,
        weight=2.0,  # 增大权重，抑制水平移动
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_still_yaw = RewTerm(
        func=penalize_yaw_rate,
        weight=1.0,  # 抑制旋转
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_contact = RewTerm(
        func=feet_flat_orientation,
        weight=1.5,  # 双脚着地奖励
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    
    # === 存活奖励 ===
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # === 正则化惩罚 ===
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    
    # === 腰部和手臂保持默认位置（强约束）===
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,  # 强惩罚腰部偏离
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.0,  # 强惩罚手臂偏离
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]),
        },
    )
    
    # === 腿部轻微正则化（允许调整高度和姿态）===
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,  # 轻微惩罚，允许腿部调整
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )
    
    # === 平滑动作 ===
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})  # 降低阈值
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})  # 更严格


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the posture control environment.
    
    站立控制任务：
    - 命令为0时保持默认站姿（高度0.78m，姿态水平）
    - 命令非0时调整高度和姿态
    - 增强抗扰动能力
    - 只用腿部调整，腰部和手臂保持不动
    """
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
    """Play environment - 部署时命令默认为0，保持站姿."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        }
        # 部署时 100% 环境使用默认姿态
        self.commands.base_posture.rel_default_envs = 1.0