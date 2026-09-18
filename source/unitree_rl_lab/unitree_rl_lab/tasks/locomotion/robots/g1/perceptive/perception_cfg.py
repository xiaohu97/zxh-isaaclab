"""G1 感知运控共用的传感器 / 地形配置。

真机传感器就是 G1 EDU 头部自带的两个（数据来自 ``g1_29dof_rev_1_0.urdf`` 的固定关节）::

    mid360_joint   parent=torso_link  xyz=(0.0002835, 0.00003, 0.41618)   rpy=(0, 0.0401, 0)
    d435_joint     parent=torso_link  xyz=(0.0576235, 0.01753, 0.42987)   rpy=(0, 0.8308, 0)   # 俯仰 47.6°

Isaac Lab 的 URDF 转换默认 ``merge_fixed_joints=True``，``mid360_link`` / ``d435_link`` /
``head_link`` 都被并进 ``torso_link``，仿真里不存在这些 prim，所以两个传感器都挂在
``torso_link`` 上，用上面的位姿做偏置。

两条感知通路
------------
* **雷达通路（先做）**：实机 Mid-360 → 里程计 + elevation_mapping 出机器人中心高程图 →
  按 yaw 对齐采样成 17×11 网格。仿真里用 ``RayCaster`` 网格从 20 m 高处向下打射线得到
  同样布局的 ``height_scan``，策略输入两边完全一致，部署不需要任何网络改动。
  网格挂在 torso_link 而不是 mid360 安装点：高程图是世界系重建的，采样中心放哪都行，
  放躯干正下方最直观，也和 Isaac Lab 官方 G1 rough 任务一致。
* **深度相机通路（框架预留）**：``RayCasterCamera`` 放在 D435 的安装位姿，输出
  ``distance_to_image_plane``。真机 D435 深度 FOV 87°×58°，这里用 64×36 的低分辨率
  复现同样的视场；训练时先用雷达通路的策略当老师做蒸馏（见 ``perceptive_env_cfg.py``）。

RayCaster 的限制：本机 Isaac Lab 0.45 只支持对**一个静态 mesh** 打射线（``/World/ground``），
所以障碍物必须做进地形（``HfDiscreteObstacles`` / ``MeshRepeatedBoxes``），动态障碍物看不见。
"""
from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# 传感器安装位姿（torso_link 系，来自 URDF）
# ---------------------------------------------------------------------------
MID360_POS_IN_TORSO = (0.0002835, 0.00003, 0.41618)
D435_POS_IN_TORSO = (0.0576235, 0.01753, 0.42987)
D435_PITCH_IN_TORSO = 0.8307767239493009  # rad, 绕 y 轴向下俯仰


def _pitch_quat_wxyz(pitch: float) -> tuple[float, float, float, float]:
    """绕 y 轴旋转 ``pitch`` 的四元数 (w, x, y, z)。"""
    return (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)


# ---------------------------------------------------------------------------
# 高程扫描（雷达通路）
# ---------------------------------------------------------------------------
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_SIZE = (1.6, 1.0)  # x 前后 1.6 m, y 左右 1.0 m → 17×11 = 187 点
HEIGHT_SCAN_OFFSET = 0.5  # 观测 = torso_z - hit_z - offset；平地站立约 0.78 - 0.5 = 0.28
HEIGHT_SCAN_NUM_POINTS = (int(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION) + 1) * (
    int(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION) + 1
)

# 实机高程图的误差来源：里程计漂移（整张图平移）+ 每格的重建噪声。
# drift_range 作用在射线起点（整张图一起漂），ray_cast_drift_range 作用在投影点（逐格抖动）。
HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=list(HEIGHT_SCAN_SIZE)),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
    drift_range=(-0.05, 0.05),
    ray_cast_drift_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
)

# 脚底小网格：只给奖励 / 终止用（地形相对的抬脚高度），不进观测。
FOOT_SCANNER_PATTERN = patterns.GridPatternCfg(resolution=0.02, size=[0.12, 0.04])  # 7×3 = 21 条射线


def foot_scanner_cfg(body_name: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.02, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=FOOT_SCANNER_PATTERN,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


# ---------------------------------------------------------------------------
# 深度相机（D435 通路，框架预留）
# ---------------------------------------------------------------------------
DEPTH_IMAGE_WIDTH = 64
DEPTH_IMAGE_HEIGHT = 36
DEPTH_MAX_DISTANCE = 4.0  # m；D435 深度可靠范围约 0.3~3 m，超出按 max 截断
_D435_FOCAL_LENGTH_CM = 24.0
# FOV = 2·atan(aperture / (2·focal))：87° → 45.6 cm，58° → 26.6 cm（Isaac 的相机参数单位是 cm）
_D435_HORIZONTAL_APERTURE_CM = 2.0 * _D435_FOCAL_LENGTH_CM * math.tan(math.radians(87.0) / 2.0)
_D435_VERTICAL_APERTURE_CM = 2.0 * _D435_FOCAL_LENGTH_CM * math.tan(math.radians(58.0) / 2.0)

DEPTH_CAMERA_CFG = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    # convention="world"：偏置四元数按 前 +X / 上 +Z 解释，和 URDF 里 d435_link 的坐标系一致
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=D435_POS_IN_TORSO, rot=_pitch_quat_wxyz(D435_PITCH_IN_TORSO), convention="world"
    ),
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=_D435_FOCAL_LENGTH_CM,
        horizontal_aperture=_D435_HORIZONTAL_APERTURE_CM,
        vertical_aperture=_D435_VERTICAL_APERTURE_CM,
        width=DEPTH_IMAGE_WIDTH,
        height=DEPTH_IMAGE_HEIGHT,
    ),
    data_types=["distance_to_image_plane"],
    max_distance=DEPTH_MAX_DISTANCE,
    depth_clipping_behavior="max",
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


# ---------------------------------------------------------------------------
# 地形：楼梯 / 斜坡 / 碎石 / 离散障碍 / 方块，带课程
# ---------------------------------------------------------------------------


@configclass
class _RepeatedBoxesTerrainCfg(terrain_gen.MeshRepeatedBoxesTerrainCfg):
    """本机 Isaac Lab 0.45 的 ``repeated_objects_terrain`` 会读 ``cfg.platform_height``，
    但 ``MeshRepeatedObjectsTerrainCfg`` 没定义这个字段（官方 rough 地形不用 boxes 所以没人踩到）。
    -1.0 = 平台高度跟随物体高度，与楼梯配置的默认值一致。"""

    platform_height: float = -1.0


# 台阶高度上限 0.15 m 对应真实楼梯；先不放 gap / pit（射线打空会出 NaN，部署侧高程图也常缺失），
# 等雷达通路跑通再加。
G1_PERCEPTIVE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "inv_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0
        ),
        "inv_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0, inverted=True
        ),
        # 小障碍：策略要么跨过去要么绕开，这是"地形级避障"的训练场
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            obstacle_width_range=(0.4, 1.2),
            obstacle_height_range=(0.05, 0.20),
            num_obstacles=6,
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": _RepeatedBoxesTerrainCfg(
            proportion=0.1,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=4, height=0.05, size=(0.5, 0.5)
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=8, height=0.20, size=(0.8, 0.8)
            ),
            platform_width=2.0,
        ),
    },
)
