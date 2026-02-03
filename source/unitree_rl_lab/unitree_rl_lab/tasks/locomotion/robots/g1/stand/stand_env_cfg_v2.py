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
    
    前推摇杆(正vx): 降低高度（蹲下）
    后拉摇杆(负vx): 提高高度（站直）
    大幅度响应让控制更明显
    """
    asset = env.scene[asset_cfg.name]
    curr_h = asset.data.root_pos_w[:, 2]
    
    # 获取速度命令
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd = cmd_term.command
    lin_vel_x = cmd[:, 0]
    
    # 基础高度 0.78m
    # 前推摇杆时降低高度（蹲下），最低到0.55m
    # 后拉摇杆时保持或略提高高度，最高到0.85m
    # 使用更大的系数实现大幅高度变化
    vx_norm = torch.clamp(lin_vel_x, -1.0, 1.0)
    # 前推(正vx)时降低高度: 0.78 - 0.20*vx = 最低0.58m
    # 后拉(负vx)时提高高度: 0.78 + 0.07*(-vx) = 最高0.85m
    target_h = 0.78 - 0.20 * vx_norm  # 大幅高度变化
    target_h = torch.clamp(target_h, 0.55, 0.85)
    
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


def track_pitch_command(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """根据速度命令控制俯仰角(pitch)
    
    前推摇杆(正vx): 身体前倾(正pitch)
    后拉摇杆(负vx): 身体后仰(负pitch)
    左右摇杆(vy): 身体左右倾斜(roll)
    """
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # 获取速度命令
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd = cmd_term.command
    lin_vel_x = cmd[:, 0]
    lin_vel_y = cmd[:, 1]
    
    # 目标俯仰角: 前推时前倾，后拉时后仰
    # vx=1.0 -> pitch=0.3rad(约17度)，vx=-1.0 -> pitch=-0.2rad(约-11度)
    vx_norm = torch.clamp(lin_vel_x, -1.0, 1.0)
    target_pitch = 0.25 * vx_norm  # 大幅俯仰变化
    target_pitch = torch.clamp(target_pitch, -0.25, 0.35)  # 限制范围
    
    # 目标横滚角: 左推时左倾，右推时右倾
    vy_norm = torch.clamp(lin_vel_y, -1.0, 1.0)
    target_roll = 0.15 * vy_norm  # 适度横滚变化
    target_roll = torch.clamp(target_roll, -0.2, 0.2)
    
    # 计算误差
    pitch_error = torch.abs(target_pitch - pitch)
    roll_error = torch.abs(target_roll - roll)
    total_error = pitch_error + roll_error
    
    # 使用高斯奖励
    reward = torch.exp(-2.0 * total_error)
    return reward


def maintain_upright_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """当没有命令时保持直立姿态
    
    只有在速度命令接近零时才惩罚倾斜
    有命令时允许倾斜（由pitch/roll跟踪奖励控制）
    """
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # 获取速度命令
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd = cmd_term.command
    cmd_magnitude = torch.norm(cmd[:, :2], dim=1)  # vx, vy的幅度
    
    # 根据命令大小调整姿态容差
    # 命令大时允许大幅倾斜，命令小时要求直立
    base_tolerance = 0.1  # 基础容差
    cmd_tolerance = 0.4 * cmd_magnitude  # 命令越大容差越大
    tolerance = base_tolerance + cmd_tolerance
    
    angle_error = torch.abs(pitch) + torch.abs(roll)
    
    # 超过容差范围才惩罚
    reward = torch.where(
        angle_error < tolerance,
        torch.ones_like(angle_error),
        torch.exp(-(angle_error - tolerance) / 0.2)
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


def penalize_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """严厉惩罚双脚同时离地（跳跃）
    
    当双脚都离地时给予大惩罚，防止跳跃行为
    至少一只脚着地时不惩罚
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    in_contact = forces > 1.0  # shape: (num_envs, 2)
    
    # 检查是否有任何一只脚着地
    any_foot_contact = torch.any(in_contact, dim=1)  # 至少一只脚着地
    
    # 双脚都离地时惩罚
    both_feet_air = ~any_foot_contact
    penalty = -5.0 * both_feet_air.float()  # 强惩罚
    return penalty


def reward_single_foot_step(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, 
                            asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励单脚迈步稳定（受冲击时的恢复动作）
    
    当机器人速度较大（受到冲击）且只有一只脚着地时给予奖励
    这鼓励机器人在不稳定时迈出一步来恢复平衡
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    in_contact = forces > 1.0  # shape: (num_envs, 2)
    
    # 检查单脚着地状态
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    single_foot_contact = (left_contact ^ right_contact)  # 异或：只有一只脚着地
    
    # 检查机器人是否在移动/受冲击（速度较大）
    base_vel = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    is_moving = base_vel > 0.3  # 速度超过0.3m/s认为在移动/受冲击
    
    # 只有在移动状态下单脚着地才给奖励（迈步恢复）
    reward = (single_foot_contact & is_moving).float() * 1.0
    return reward


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
    """使用标准速度命令控制高度和姿态
    
    关键调整：
    - 扩大训练范围让策略学习大幅度姿态变化
    - 降低 standing_envs 比例，增加动态训练
    - 更频繁的命令更新(3-6s)让策略快速响应
    """
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),  # 更频繁的命令更新
        rel_standing_envs=0.3,  # 只有30%环境站立，70%训练动态响应
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.0),  # 大范围训练，让策略学习完整高度变化
            lin_vel_y=(-0.5, 0.5),  # 左右倾斜
            ang_vel_z=(-0.5, 0.5)   # 旋转
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),  # 部署时的最大范围
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-0.8, 0.8)
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
    """站立奖励配置 - 增强姿态和高度响应性
    
    设计原则：
    1. 高权重的姿态/高度跟踪让机器人响应摇杆命令
    2. 适度的稳定性惩罚防止抖动
    3. 降低水平速度惩罚，允许姿态变化时的轻微移动
    """
    
    # === 核心姿态控制奖励（高权重）===
    # 高度跟踪 - 根据命令大幅改变高度
    height_command_tracking = RewTerm(
        func=track_height_command,
        weight=5.0,  # 高权重确保高度响应
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 俯仰/横滚跟踪 - 根据命令改变身体姿态
    pitch_tracking = RewTerm(
        func=track_pitch_command,
        weight=4.0,  # 高权重确保姿态响应
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 无命令时保持直立（低权重，不干扰姿态响应）
    posture_tracking = RewTerm(
        func=maintain_upright_posture,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 稳定性惩罚（降低权重，允许动态响应）===
    stand_still_xy = RewTerm(
        func=penalize_horizontal_velocity,
        weight=0.5,  # 降低，允许姿态变化时的轻微移动
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    stand_still_yaw = RewTerm(
        func=penalize_yaw_rate,
        weight=1.0,  # 保持不变
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 脚部接触奖励 ===
    # 奖励双脚着地（稳定站立）
    feet_contact = RewTerm(
        func=both_feet_contact,
        weight=2.0,  # 增加权重鼓励双脚着地
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    
    # 严厉惩罚双脚同时离地（禁止跳跃）
    no_jump_penalty = RewTerm(
        func=penalize_air_time,
        weight=1.0,  # 函数内部已有-5.0惩罚
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    
    # 奖励受冲击时单脚迈步恢复平衡
    step_recovery = RewTerm(
        func=reward_single_foot_step,
        weight=1.0,  # 鼓励迈步恢复
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # === 关键：解决抖动的惩罚 ===
    # 惩罚高速关节运动 - 鼓励平缓控制
    joint_velocity_penalty = RewTerm(
        func=penalize_joint_velocity,
        weight=0.5,  # 函数返回负值，权重为正，相乘得负惩罚
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === 关节约束（降低权重，允许姿态变化）===
    # 腰部可以适度运动来实现俯仰变化
    waist_penalty = RewTerm(
        func=penalize_waist_motion,
        weight=0.5,  # 大幅降低，允许腰部运动实现姿态
        params={"asset_cfg": SceneEntityCfg("robot"), "waist_indices": [12, 13, 14]},
    )
    
    arm_penalty = RewTerm(
        func=penalize_arm_motion,
        weight=1.0,  # 略微降低
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
        params={"limit_angle": 1.2},  # 从0.8增加到1.2(约70度)，允许更大倾斜
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
