"""Unitree G1 29dof —— 感知运控三档任务（2026-09-17）。

目标：在 ``velocitywithid_yaw_robust``（线上 walk 的训练配置）之上加复杂地形和外感知，
先用 G1 EDU 自带的 Mid-360 雷达（高程图），框架里同时预留 D435i 深度相机。三档任务共用
同一套场景 / 地形 / 奖励 / 终止，只有观测组不同，所以互相之间可以干净地比较：

======================================  ==================================  =============================
任务                                    actor 观测                          用途
======================================  ==================================  =============================
``Unitree-G1-29dof-PerceptiveBlind``    与线上 walk 完全相同（本体 ×5 帧）   rough 地形盲走基线；
                                                                            ONNX 可直接替换线上策略
``Unitree-G1-29dof-PerceptiveHeightScan`` 本体 ×5 帧 + 高程图 187 点 ×1 帧   雷达通路正式策略，
                                                                            也是深度学生的老师
``Unitree-G1-29dof-PerceptiveDepth``    本体 ×5 帧 + 深度图 64×36 ×1 帧      深度通路（蒸馏，框架预留）
======================================  ==================================  =============================

三个 critic 都拿高程图（无噪声）+ 基类的特权量。

相对 ``velocitywithid_yaw_robust`` 改了什么
------------------------------------------
* 地形：flat → ``G1_PERCEPTIVE_TERRAINS_CFG``（楼梯 / 斜坡 / 碎石 / 离散障碍 / 方块，带课程），
  ``max_init_terrain_level`` 5；``terrain_levels_vel`` 课程基类里本来就挂着。
* 场景：``height_scanner`` 换成带漂移噪声的版本；新增两只脚底小网格扫描（只给奖励用）。
* 奖励：三项在非平地上会算错的换成地形相对量，其余全部继承::

    base_height     加 sensor_cfg=height_scanner（Isaac Lab 自带的地形修正）
    feet_clearance  foot_clearance_reward → foot_clearance_terrain（脚高相对脚底地面）
    feet_stumble    新增 -1.0：脚的水平接触力 > 3× 竖直力（踢到台阶立面）

* 终止：``base_height`` 世界 z 版 → 地形相对版。世界 z 版在倒金字塔楼梯底部会把站立判成倒地。
* 观测历史：HeightScan / Depth 把组级 ``history_length=5`` 改成逐项设置（本体 5 帧、外感知 1 帧），
  Isaac Lab 的组级历史会覆盖所有项，不改的话高程图也会带 5 帧（935 维）。
  逐项历史的拼接布局和组级完全一样（都是按项分块），部署侧 C++ ObservationManager 的非 gym
  路径本来就按项处理，只需放开 ``walk_policy.h`` 里"六项 × 5 帧"的硬编码校验。

观测维度::

    本体六项每帧 96 → ×5 = 480
    height_scan 187 (17×11) ×1
    depth_image 2304 (64×36) ×1
    Blind actor 480 / HeightScan actor 667 / Depth 学生 2784

硬地形变体（``...Hard``，2026-09-18）
-----------------------------------
上面那份地形盲走也能过（400 个 episode 只摔 11 次），高程图只把摔倒率 9.5% 压到 2.2%，
期望回报差不到 1%，所以策略 3000 迭代就平台期。硬地形把台阶提到 0.20~0.30 m、坡度上限
提到 0.45（24°），并加入踏石和 gap，见 ``perception_cfg.G1_PERCEPTIVE_HARD_TERRAINS_CFG``。
奖励里只有 ``base_height`` 换成空洞安全版（见 ``HardRewardsCfg``），其余完全一致。

怎么训
------
::

    # 1. 盲走基线（顺便验证地形和奖励）
    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PerceptiveBlind --headless --max_iterations 15000
    # 2. 高程图进 actor（actor 维度变了，不能 --resume Blind）
    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PerceptiveHeightScan --headless --max_iterations 15000
    # 3. 深度学生（老师 = 第 2 步的 checkpoint，同一个实验目录）
    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PerceptiveDepth --headless \\
        --load_run <第 2 步 run 目录> --checkpoint model_15000.pt --num_envs 1024

验收看 ``Curriculum/terrain_levels``（ultra 盲走 rough 5 万迭代到 5.5/10，HeightScan 应明显更高）、
``Episode_Termination/bad_orientation`` 和 ``Episode_Reward/feet_stumble``。
"""
from __future__ import annotations

import copy
import importlib

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid_yaw.velocitywithid_yaw_env_cfg import (
    RewardsCfg as _YawRewardsCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid_yaw_robust.velocitywithid_yaw_robust_env_cfg import (
    RobotEnvCfg as _RobustEnvCfg,
)

from . import perceptive_mdp as pmdp
from .perception_cfg import (
    DEPTH_CAMERA_CFG,
    DEPTH_MAX_DISTANCE,
    G1_PERCEPTIVE_HARD_TERRAINS_CFG,
    G1_PERCEPTIVE_TERRAINS_CFG,
    HEIGHT_SCAN_OFFSET,
    HEIGHT_SCANNER_CFG,
    foot_scanner_cfg,
)

# velocity 任务的包名是 ``29dof``，数字开头不是合法标识符，写不成 import 语句
_velocity_env_cfg = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
_BaseSceneCfg = _velocity_env_cfg.RobotSceneCfg
_BaseObservationsCfg = _velocity_env_cfg.ObservationsCfg
_BaseTerminationsCfg = _velocity_env_cfg.TerminationsCfg

PROPRIO_HISTORY_LENGTH = 5
EXTEROCEPTIVE_TERMS = ("height_scan", "depth_image")


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
@configclass
class PerceptiveSceneCfg(_BaseSceneCfg):
    """velocity 场景 + 脚底扫描；地形和高程扫描在 ``__post_init__`` 里换成感知版。"""

    left_foot_scanner: RayCasterCfg = foot_scanner_cfg("left_ankle_roll_link")
    right_foot_scanner: RayCasterCfg = foot_scanner_cfg("right_ankle_roll_link")
    # Depth 任务才填；None 时 InteractiveScene 直接跳过
    depth_camera: RayCasterCameraCfg | None = None

    def __post_init__(self):
        # 模块级配置对象要 deepcopy：Play 配置会改 num_rows，不能改到共享对象上
        self.terrain.terrain_generator = copy.deepcopy(G1_PERCEPTIVE_TERRAINS_CFG)
        self.terrain.max_init_terrain_level = 5
        self.height_scanner = copy.deepcopy(HEIGHT_SCANNER_CFG)


@configclass
class DepthSceneCfg(PerceptiveSceneCfg):
    depth_camera: RayCasterCameraCfg = copy.deepcopy(DEPTH_CAMERA_CFG)


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------
def _use_per_term_history(group) -> None:
    """组级历史改成逐项：本体项 5 帧，外感知项保持各自的 history_length（1）。"""
    group.history_length = None
    for name, term in vars(group).items():
        if isinstance(term, ObsTerm) and name not in EXTEROCEPTIVE_TERMS:
            term.history_length = PROPRIO_HISTORY_LENGTH


_HEIGHT_SCAN_PARAMS = {"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": HEIGHT_SCAN_OFFSET}


@configclass
class HeightScanPolicyCfg(_BaseObservationsCfg.PolicyCfg):
    """本体六项（继承，含噪声）+ 带噪高程图。"""

    height_scan = ObsTerm(
        func=pmdp.height_scan,
        params=_HEIGHT_SCAN_PARAMS,
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
        history_length=1,
    )

    def __post_init__(self):
        super().__post_init__()
        _use_per_term_history(self)


@configclass
class HeightScanCriticCfg(_BaseObservationsCfg.CriticCfg):
    """基类特权量 + 无噪高程图。"""

    height_scan = ObsTerm(func=pmdp.height_scan, params=_HEIGHT_SCAN_PARAMS, clip=(-1.0, 1.0), history_length=1)

    def __post_init__(self):
        super().__post_init__()
        _use_per_term_history(self)


@configclass
class DepthStudentPolicyCfg(_BaseObservationsCfg.PolicyCfg):
    """本体六项 + 归一化深度图（展平）。"""

    depth_image = ObsTerm(
        func=pmdp.depth_image,
        params={"sensor_cfg": SceneEntityCfg("depth_camera"), "max_distance": DEPTH_MAX_DISTANCE},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(0.0, 1.0),
        history_length=1,
    )

    def __post_init__(self):
        super().__post_init__()
        _use_per_term_history(self)


@configclass
class BlindObservationsCfg(_BaseObservationsCfg):
    """actor 与线上 walk 完全一致；只有 critic 多看高程图。"""

    critic: HeightScanCriticCfg = HeightScanCriticCfg()


@configclass
class HeightScanObservationsCfg(_BaseObservationsCfg):
    policy: HeightScanPolicyCfg = HeightScanPolicyCfg()
    critic: HeightScanCriticCfg = HeightScanCriticCfg()


@configclass
class DepthObservationsCfg(_BaseObservationsCfg):
    """rsl_rl 的 Distillation 按组名找：``policy`` = 学生，``teacher`` = 老师。"""

    policy: DepthStudentPolicyCfg = DepthStudentPolicyCfg()
    teacher: HeightScanPolicyCfg = HeightScanPolicyCfg()
    critic: HeightScanCriticCfg = HeightScanCriticCfg()


# ---------------------------------------------------------------------------
# 奖励 / 终止
# ---------------------------------------------------------------------------
_FEET_CONTACT_CFG = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")


@configclass
class PerceptiveRewardsCfg(_YawRewardsCfg):
    """只换三项和地形相关的，其余（含 yaw 版的转向奖励）全部继承。"""

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    feet_clearance = RewTerm(
        func=pmdp.foot_clearance_terrain,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
        },
    )
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-1.0, params={"sensor_cfg": _FEET_CONTACT_CFG})


@configclass
class PerceptiveTerminationsCfg(_BaseTerminationsCfg):
    base_height = DoneTerm(
        func=pmdp.root_height_below_minimum_terrain,
        params={"minimum_height": 0.2, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
@configclass
class PerceptiveBlindEnvCfg(_RobustEnvCfg):
    """URDF / 动作 / 指令 / 事件 / 课程 / PPO 与 ``velocitywithid_yaw_robust`` 相同。"""

    scene: PerceptiveSceneCfg = PerceptiveSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BlindObservationsCfg = BlindObservationsCfg()
    rewards: PerceptiveRewardsCfg = PerceptiveRewardsCfg()
    terminations: PerceptiveTerminationsCfg = PerceptiveTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        step_dt = self.decimation * self.sim.dt
        self.scene.left_foot_scanner.update_period = step_dt
        self.scene.right_foot_scanner.update_period = step_dt
        if self.scene.depth_camera is not None:
            self.scene.depth_camera.update_period = step_dt
        # 楼梯 / 方块地形的碰撞对比平地多得多，ultra rough 用的就是这个值
        self.sim.physx.gpu_collision_stack_size = 2**29


@configclass
class PerceptiveHeightScanEnvCfg(PerceptiveBlindEnvCfg):
    observations: HeightScanObservationsCfg = HeightScanObservationsCfg()


@configclass
class PerceptiveDepthEnvCfg(PerceptiveBlindEnvCfg):
    scene: DepthSceneCfg = DepthSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DepthObservationsCfg = DepthObservationsCfg()


def _apply_play_settings(cfg: PerceptiveBlindEnvCfg) -> None:
    cfg.scene.num_envs = 32
    # 保留中高难度行，不然 play 看不到楼梯
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.num_cols = 8
    cfg.scene.terrain.max_init_terrain_level = 4
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges


# ---------------------------------------------------------------------------
# 硬地形变体（2026-09-18）
# ---------------------------------------------------------------------------
@configclass
class HardSceneCfg(PerceptiveSceneCfg):
    """地形换成 ``G1_PERCEPTIVE_HARD_TERRAINS_CFG``，其余（传感器、噪声）不变。"""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = copy.deepcopy(G1_PERCEPTIVE_HARD_TERRAINS_CFG)
        # 台阶 0.20 m 起步，热启动的策略也得从最低档重新爬
        self.terrain.max_init_terrain_level = 1


@configclass
class HardRewardsCfg(PerceptiveRewardsCfg):
    """``base_height`` 换成空洞安全版：gap 打空成 inf、踏石的洞在 −10 m，
    Isaac Lab 自带的 ``base_height_l2`` 对射线取均值且无防护，会直接把奖励打成 NaN。
    顺带修掉它在楼梯上目标偏 0.3 m 的问题（改用躯干正下方最近的有效地面）。"""

    base_height = RewTerm(
        func=pmdp.base_height_terrain,
        weight=-10,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class PerceptiveBlindHardEnvCfg(PerceptiveBlindEnvCfg):
    """硬地形上的盲走基线：只用来量"没有高程图能走多远"，不一定训得动。"""

    scene: HardSceneCfg = HardSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HardRewardsCfg = HardRewardsCfg()


@configclass
class PerceptiveHeightScanHardEnvCfg(PerceptiveHeightScanEnvCfg):
    scene: HardSceneCfg = HardSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HardRewardsCfg = HardRewardsCfg()


@configclass
class PerceptiveBlindPlayEnvCfg(PerceptiveBlindEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class PerceptiveHeightScanPlayEnvCfg(PerceptiveHeightScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class PerceptiveDepthPlayEnvCfg(PerceptiveDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class PerceptiveBlindHardPlayEnvCfg(PerceptiveBlindHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class PerceptiveHeightScanHardPlayEnvCfg(PerceptiveHeightScanHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)
