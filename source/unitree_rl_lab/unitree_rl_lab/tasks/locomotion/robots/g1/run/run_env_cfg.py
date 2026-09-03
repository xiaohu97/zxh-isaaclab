"""Unitree G1 29dof —— HugWBC 式可控步态跑步任务。

与 ``Unitree-G1-29dof-Velocity`` 的关键差别
-------------------------------------------
1. **指令空间**：除 (vx, vy, ωz) 外还暴露 5 个行为指令 —— 步频、支撑相比例、摆动腿
   高度、躯干高度、躯干俯仰。支撑相比例 <0.5 即产生腾空期，是"跑"而不是"快走"的
   分界线。原 Velocity 任务的 ``feet_gait`` 写死 ``threshold=0.55``（10% 双支撑），
   结构上就不可能腾空，速度上限提再高也只是快走。
2. **奖励放松**：躯干高度/俯仰从硬惩罚改为指令跟踪；``flat_orientation_l2`` 换成只压
   横滚；``lin_vel_z_l2`` 权重从 -2.0 降到 -0.3（腾空必然有竖直速度）。
3. **课程**：速度上限、步频上限、支撑相下限由同一个 progress 标量一起推进 ——
   先学会走，再逼出腾空期。
4. **地形**：平地。跑步先在平地把步态学出来，粗糙地形是后面的事。

⚠️ 观测维度与 Velocity 任务不同（多了 8 维指令 + 2 维步态时钟，去掉了原来的 3 维速度
指令），``deploy/robots/g1_29dof/config`` 那边要单独配一套才能上机。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from . import mdp

# 左右顺序在多处必须一致（GaitCommand.leg_phase 是 [左, 右]），所以显式列名 +
# preserve_order，不靠正则的字典序。
FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
SHOULDER_PITCH_JOINTS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
LEG_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """跑步场景：平地，无高度扫描（原 Velocity 任务里那个 height_scanner 观测本来就是
    注释掉的，这里直接去掉，省显存和一次 raycast）。"""

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
        # plane 地形只认 diffuse_color，给 MdlFileCfg 会在 import_ground_plane 里报错
        visual_material=None,
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """事件与域随机化。跑步的 sim2real gap 比走路大，扰动整体比 Velocity 任务加强。"""

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 跑步吃摩擦，下限比走路任务(0.3)提高，否则低摩擦 env 学出来的是打滑步态
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
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

    # 实机 PD 与仿真隐式执行器有偏差，高速下被放大。隐式执行器只能在 startup 改。
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # -- reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 给一点初速度，省得每个 episode 都要从静止起步（跑步起步阶段最难）
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """HugWBC 式命令空间。``ranges`` 是课程起点（走），``limit_ranges`` 是终点（跑）。"""

    base_velocity = mdp.GaitCommandCfg(
        asset_name="robot",
        # 跑步下 10s 一换太长，加减速要练得多一些
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.05,
        randomize_start_phase=True,
        foot_offset=0.035,
        max_stride_length=1.0,
        flight_speed_threshold=1.8,
        running_stance_ratio=0.48,
        debug_vis=True,
        ranges=mdp.GaitCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            gait_freq=(1.2, 1.8),
            stance_ratio=(0.50, 0.65),  # 全是双支撑步态：先学会走
            swing_height=(0.08, 0.15),
            body_height=(0.68, 0.78),
            body_pitch=(-0.1, 0.1),
        ),
        limit_ranges=mdp.GaitCommandCfg.Ranges(
            lin_vel_x=(-1.0, 3.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            gait_freq=(1.2, 3.0),
            stance_ratio=(0.30, 0.65),  # 下探到 0.30 -> 最多 40% 的周期处于腾空
            swing_height=(0.08, 0.25),
            body_height=(0.62, 0.78),
            body_pitch=(-0.1, 0.4),  # 允许前倾
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 8 维：速度(3, 原量纲) + 步态参数(5, 归一到 [-1,1])
        gait_commands = ObsTerm(func=mdp.gait_commands, params={"command_name": "base_velocity"})
        # 2 维步态时钟。没有它策略只能从触地历史反推相位，腾空期极难学出来
        gait_clock = ObsTerm(func=mdp.gait_clock, params={"command_name": "base_velocity"})
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
        gait_commands = ObsTerm(func=mdp.gait_commands, params={"command_name": "base_velocity"})
        gait_clock = ObsTerm(func=mdp.gait_clock, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- 任务：跟速度指令
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- 行为：跟步态指令。这四项决定"可控"，也是腾空期的来源
    gait_contact = RewTerm(
        func=mdp.gait_contact,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES, preserve_order=True),
        },
    )
    foot_height = RewTerm(
        func=mdp.foot_height_tracking,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES, preserve_order=True),
        },
    )
    body_height = RewTerm(
        func=mdp.body_height_tracking, weight=0.5, params={"command_name": "base_velocity", "std": 0.05}
    )
    body_pitch = RewTerm(
        func=mdp.body_pitch_tracking, weight=0.3, params={"command_name": "base_velocity", "std": 0.2}
    )
    arm_swing = RewTerm(
        func=mdp.arm_swing,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "std": 0.4,
            "max_amplitude": 0.5,
            "vel_scale": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=SHOULDER_PITCH_JOINTS, preserve_order=True),
        },
    )

    # -- 躯干姿态
    base_roll = RewTerm(func=mdp.base_roll_l2, weight=-3.0)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- 平滑 / 能耗。这些惩罚随速度平方增长，权重整体比 Velocity 任务低一档，
    #    否则 3 m/s 时会盖过跟踪项，策略宁可站着不动
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # 膝关节 velocity_limit_sim=20 rad/s，2.5 m/s 以上会持续顶上限，不压的话学出来的
    # 动作真机复现不了。已确认 soft_joint_vel_limits 正确继承了 velocity_limit_sim
    # （膝/髋roll 20、髋pitch 32、踝pitch 37 rad/s），soft_ratio=0.9 即超过 18 rad/s 起罚。
    # 注意不要换成 applied_torque_limits —— 隐式执行器下 applied==computed，那一项恒为 0。
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )

    # -- 关节姿态
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                # 必须排除 shoulder_pitch —— 那是 arm_swing 在控的自由度，两项会打架
                joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- 足端
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES, preserve_order=True),
        },
    )
    # G1 零髋滚转时两脚间距 0.237 m，跑步落脚天然向中线收，阈值 0.2 会逼出外八字；
    # 0.1 只挡绞腿
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-2.0,
        params={
            "threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES, preserve_order=True),
        },
    )

    # -- 站立
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation, weight=-0.5, params={"command_name": "base_velocity"}
    )

    # -- 其他
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    # 指令俯仰最大 0.4 rad，阈值留到 1.0 才不会把正常前倾误判成摔倒
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # 骨盆/躯干着地就是摔了，早点结束比让它在地上刷 alive 强
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]),
        },
    )


@configclass
class CurriculumCfg:
    gait_cmd_levels = CurrTerm(
        func=mdp.gait_cmd_levels,
        params={"command_name": "base_velocity", "reward_term_name": "track_lin_vel_xy", "delta": 0.02},
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29dof 可控步态跑步任务。"""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz 控制，200 Hz 物理
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # play 时直接给满量程：ranges == limit_ranges 后 progress 取任何值都等于跑步区间
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.curriculum.gait_cmd_levels = None
