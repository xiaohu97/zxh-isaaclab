"""
G1 29DOF 站立策略配置（兼容部署版本）

核心设计原则：
1. 使用与行走策略相同的命令系统 (base_velocity)，确保部署兼容
2. 站立时速度命令为零，通过高比例 rel_standing_envs 训练
3. 观测结构与 velocity 策略完全一致，可直接复用 deploy.yaml
4. 通过奖励函数鼓励稳定站立、抗扰动
"""
from __future__ import annotations

import math
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


# ==============================================================================
# 自定义奖励函数
# ==============================================================================

def track_height_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                          target_height: float = 0.78) -> torch.Tensor:
    """追踪高度 - 使用更平缓的奖励避免抖动
    
    当高度很接近目标时，奖励较高；超出范围时惩罚
    这样可以让策略学会保持稳定而不是频繁调整
    """
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    
    # 允许的高度范围：±0.05m
    h_error = torch.abs(target_height - curr_h)
    
    # 使用分段函数避免剧烈变化
    # 在0.05m内：高奖励
    # 0.05-0.15m：线性衰减
    # 超过0.15m：惩罚
    reward = torch.where(
        h_error < 0.05,
        torch.ones_like(h_error),  # 在目标范围内给满分
        torch.where(
            h_error < 0.15,
            1.0 - (h_error - 0.05) / 0.1,  # 线性衰减
            -0.5 * torch.square(h_error - 0.15)  # 超出范围惩罚
        )
    )
    return reward


def track_height_command(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """根据速度命令来调整目标高度
    
    当有前进命令时，略微降低高度（俯身）
    当有后退命令时，略微提高高度（挺身）
    这提高了对高度命令的响应性
    """
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    
    # 获取速度命令
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd = cmd_term.command
    lin_vel_x = cmd[:, 0]
    
    # 基础高度 0.78m
    # 有前进命令时降低高度: 0.78 - 0.05*vx_norm
    # 有后退命令时提高高度: 0.78 + 0.03*(-vx_norm)
    vx_norm = torch.clamp(lin_vel_x, -1.0, 1.0)
    target_h = 0.78 - 0.04 * vx_norm  # 前进时降低，后退时提高
    target_h = torch.clamp(target_h, 0.70, 0.86)
    
    h_error = torch.abs(target_h - curr_h)
    reward = torch.where(
        h_error < 0.05,
        torch.ones_like(h_error),
        torch.where(
            h_error < 0.15,
            1.0 - (h_error - 0.05) / 0.1,
            -0.5 * torch.square(h_error - 0.15)
        )
    )
    return reward


def maintain_upright_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励保持直立姿态（pitch和roll接近0）
    
    使用较宽松的容差避免过度纠正导致抖动
    """
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # 容差范围：±0.1 rad (±5.7°)
    angle_error = torch.abs(pitch) + torch.abs(roll)
    
    reward = torch.where(
        angle_error < 0.1,
        torch.ones_like(angle_error),
        torch.where(
            angle_error < 0.3,
            1.0 - (angle_error - 0.1) / 0.2,
            torch.exp(-angle_error / 0.1)
        )
    )
    return reward


def penalize_horizontal_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚水平移动（站立应该不动）
    
    使用平方使得大幅度移动更加惩罚
    """
    asset = env.scene[asset_cfg.name]
    vel_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return -torch.square(vel_xy)


def penalize_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚旋转"""
    asset = env.scene[asset_cfg.name]
    return -torch.square(asset.data.root_ang_vel_w[:, 2])


def both_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励双脚同时着地"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    in_contact = forces > 1.0
    both_contact = torch.all(in_contact, dim=1).float()
    return both_contact


def penalize_joint_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚关节速度过大 - 这是解决抖动的关键
    
    抖动的根本原因是关节在快速往复运动
    通过惩罚高速运动可以鼓励平缓的控制
    """
    asset = env.scene[asset_cfg.name]
    # 特别对腿部关节进行惩罚 (关节 0-11)
    joint_vel = asset.data.joint_vel[:, :12]  # 只看腿部关节
    vel_penalty = torch.mean(torch.square(joint_vel), dim=1)
    return -vel_penalty


def penalize_waist_motion(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                          waist_indices: list = [12, 13, 14]) -> torch.Tensor:
    """惩罚腰部关节偏离默认位置"""
    asset = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    waist_deviation = torch.sum(torch.square(joint_pos_rel[:, waist_indices]), dim=1)
    return -waist_deviation


def penalize_arm_motion(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                        arm_indices: list = list(range(15, 29))) -> torch.Tensor:
    """惩罚手臂关节偏离默认位置"""
    asset = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    arm_deviation = torch.sum(torch.square(joint_pos_rel[:, arm_indices]), dim=1)
    return -arm_deviation


# ==============================================================================
# 场景配置
# ==============================================================================

FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
    """场景配置"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
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

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )


# ==============================================================================
# 事件配置
# ==============================================================================

@configclass
class EventCfg:
    """域随机化和事件配置"""
    # 物理属性随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-3.0, 5.0),
            "operation": "add",
        },
    )

    # 重置
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

    # 外部扰动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# ==============================================================================
# 命令配置（使用与行走相同的速度命令，确保部署兼容）
# ==============================================================================

@configclass
class CommandsCfg:
    """使用标准速度命令，高比例 standing_envs 训练站立
    
    关键调整：
    - ranges: 训练时探索的命令范围（较小）
    - limit_ranges: 部署时的最大命令范围（较大）
    - 更频繁的命令更新(8-12s)让策略适应动态变化
    """
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),  # 更频繁的命令更新，让高度响应更敏捷
        rel_standing_envs=0.75,  # 75% 环境速度命令为零（训练站立）
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.15, 0.15),  # 扩大训练范围，增强高度响应能力
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),  # 部署时的最大范围
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2)
        ),
    )


# ==============================================================================
# 动作配置
# ==============================================================================

@configclass
class ActionsCfg:
    """动作配置 - 必须保持与部署代码一致"""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True
    )


# ==============================================================================
# 观测配置（与 velocity 策略完全一致，确保部署兼容）
# ==============================================================================

@configclass
class ObservationsCfg:
    """观测配置 - 与行走策略结构完全一致"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测 - 顺序必须与 deploy.yaml 一致"""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ==============================================================================
# 奖励配置（专注站立稳定性）
# ==============================================================================

@configclass
class RewardsCfg:
    """站立奖励配置 - 平衡稳定性和命令响应性
    
    权重调整原则：
    1. 降低高度/姿态权重，避免过度纠正导致抖动
    2. 增加关节速度惩罚，平缓运动
    3. 增加关节位置惩罚，稳定站姿
    4. 增加动作率惩罚，避免急剧变化
    """
    
    # === 站立核心奖励 ===
    # 降低高度跟踪权重，使用新的平缓奖励函数
    height_tracking = RewTerm(
        func=track_height_velocity,
        weight=1.5,  # 从 3.0 降低到 1.5
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.78},
    )
    
    # 根据命令调整目标高度 - 增强高度控制性
    height_command_tracking = RewTerm(
        func=track_height_command,
        weight=1.5,  # 新增权重
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 降低姿态跟踪权重，使用更宽松的容差
    posture_tracking = RewTerm(
        func=maintain_upright_posture,
        weight=1.5,  # 从 3.0 降低到 1.5
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 速度跟踪（站立时命令为零）===
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,  # 从 2.0 降低到 1.5
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,  # 从 1.5 降低到 1.0
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    # === 稳定性奖励 ===
    stand_still_xy = RewTerm(
        func=penalize_horizontal_velocity,
        weight=2.0,  # 保持不变
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    stand_still_yaw = RewTerm(
        func=penalize_yaw_rate,
        weight=1.0,  # 保持不变
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    feet_contact = RewTerm(
        func=both_feet_contact,
        weight=1.0,  # 从 1.5 降低到 1.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    
    # === 关键：解决抖动的惩罚 ===
    # 惩罚高速关节运动 - 鼓励平缓控制
    joint_velocity_penalty = RewTerm(
        func=penalize_joint_velocity,
        weight=-0.5,  # 新增强的惩罚
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 关节约束 ===
    # 腰部保持在默认位置，避免不必要的摆动
    waist_penalty = RewTerm(
        func=penalize_waist_motion,
        weight=-2.0,  # 从 -3.0 降低到 -2.0
        params={"asset_cfg": SceneEntityCfg("robot"), "waist_indices": [12, 13, 14]},
    )
    
    arm_penalty = RewTerm(
        func=penalize_arm_motion,
        weight=-1.5,  # 从 -2.0 降低到 -1.5
        params={"asset_cfg": SceneEntityCfg("robot"), "arm_indices": list(range(15, 29))},
    )
    
    # === 正则化惩罚（关键：这些可以很好地抑制抖动）===
    # 惩罚扭矩 - 防止过度使用关节
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0005)  # 增强到 -0.0005
    
    # 惩罚动作变化率 - 这是抑制抖动的最重要参数！
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # 从 -0.01 增强到 -0.05
    
    # 惩罚加速度 - 平缓运动
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)  # 增强到 -1e-6


# ==============================================================================
# 终止条件
# ==============================================================================

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},  # 约45度
    )


# ==============================================================================
# 主环境配置
# ==============================================================================

@configclass
class G1StandEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29DOF 站立环境配置（兼容部署版本）"""
    
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 增加减速步数 - 这样策略输出会被平缓化，减少抖动
        self.decimation = 8  # 从 4 增加到 8，控制频率从 200Hz 降到 100Hz
        self.episode_length_s = 20.0
        
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


# 训练和评估配置
@configclass
class G1StandEnvCfg_TRAIN(G1StandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.observations.policy.enable_corruption = True


@configclass 
class G1StandEnvCfg_PLAY(G1StandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.rel_standing_envs = 1.0  # 播放时全部站立
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
