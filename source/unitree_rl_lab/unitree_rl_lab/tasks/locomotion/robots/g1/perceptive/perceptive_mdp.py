"""G1 感知任务专用的观测 / 奖励 / 终止项。

只放 ``velocity`` 系列在非平地上会出错、必须换成地形相对量的那几项；其余全部沿用
``unitree_rl_lab.tasks.locomotion.mdp``。
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, RayCasterCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------
def height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """``torso_z - hit_z - offset``，打空的射线（洞 / 缺失）按最深处理。

    和 ``isaaclab.envs.mdp.height_scan`` 同一公式，只多了 ``nan_to_num``：RayCaster 打空时
    ``ray_hits_w`` 是 inf，直接减会得到 -inf，进网络前 ObsTerm 的 clip 能兜住 inf 但兜不住 NaN。
    部署侧 C++ 的 ``height_scan`` 项只需把高程图节点发来的 187 维向量原样返回，
    scale / clip 由 ObservationManager 按 deploy.yaml 复现。
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.nan_to_num(scan, nan=-1.0e3, posinf=1.0e3, neginf=-1.0e3)


def depth_image(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float = 4.0) -> torch.Tensor:
    """深度图归一化到 [0, 1] 后展平，(N, H*W)。超量程 / 打空 = 1.0。"""
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"][..., 0]  # (N, H, W)
    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    depth = torch.clamp(depth, min=0.0, max=max_distance) / max_distance
    return depth.flatten(1)


# ---------------------------------------------------------------------------
# 地面高度：空洞安全的共用件
# ---------------------------------------------------------------------------
# 带空洞的地形有两种"没有地面"：
#   * ``MeshGapTerrain``：缺口处根本没有几何体，射线打空，``ray_hits_w`` 是 inf；
#   * ``HfSteppingStones``：洞是高度场的一部分，命中点在 ``holes_depth``（默认 −10 m）。
# 两者都必须排除，否则 inf 会污染奖励（Isaac Lab 自带的 ``base_height_l2`` 对 ray_hits_w 取
# 均值，没有任何防护，一条 inf 射线就能让整项奖励变 NaN），−10 m 也会把均值拖垮。
DEFAULT_MAX_DROP = 2.0
"""命中点比参考高度低超过这个值就当成"洞"，不作为地面 [m]。"""


def _valid_ground_mask(hits_z: torch.Tensor, ref_z: torch.Tensor, max_drop: float) -> torch.Tensor:
    return torch.isfinite(hits_z) & (hits_z > ref_z.unsqueeze(-1) - max_drop)


def ground_under_point(
    sensor: RayCaster, ref_xy: torch.Tensor, ref_z: torch.Tensor, max_drop: float = DEFAULT_MAX_DROP
) -> tuple[torch.Tensor, torch.Tensor]:
    """离 ``ref_xy`` 最近的那条有效射线的地面高度，以及"是否存在有效射线"。

    取最近点而不是整张图的均值：楼梯上均值能和脚下差 0.3 m 以上，踏石上还会被洞带偏。
    """
    hits = sensor.data.ray_hits_w
    z = hits[..., 2]
    valid = _valid_ground_mask(z, ref_z, max_drop)
    xy = torch.nan_to_num(hits[..., :2], nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
    dist = torch.norm(xy - ref_xy.unsqueeze(1), dim=-1)
    dist = torch.where(valid, dist, torch.full_like(dist, 1.0e6))
    idx = torch.argmin(dist, dim=1)
    rows = torch.arange(z.shape[0], device=z.device)
    ground = torch.nan_to_num(z[rows, idx], nan=0.0, posinf=0.0, neginf=0.0)
    return ground, valid.any(dim=1)


# ---------------------------------------------------------------------------
# 奖励
# ---------------------------------------------------------------------------
def _ground_height_under(sensor: RayCaster, fallback: torch.Tensor, max_drop: float = DEFAULT_MAX_DROP) -> torch.Tensor:
    """脚底小网格的平均命中高度；全部无效（悬在坑上 / 踏石的洞）时退回 ``fallback``。"""
    hits = sensor.data.ray_hits_w[..., 2]
    valid = _valid_ground_mask(hits, fallback, max_drop)
    count = valid.sum(dim=1)
    total = torch.where(valid, torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(hits)).sum(dim=1)
    return torch.where(count > 0, total / count.clamp(min=1), fallback)


def base_height_terrain(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_drop: float = DEFAULT_MAX_DROP,
) -> torch.Tensor:
    """``base_height_l2`` 的空洞安全版：目标高度按**躯干正下方最近的有效地面**抬升。

    Isaac Lab 自带版本用整张高程图的均值，在楼梯上目标能偏 0.3 m 以上（压制主动调姿），
    在 gap / 踏石上还会被 inf 或 −10 m 直接打成 NaN。没有任何有效射线时（整片悬空）返回 0，
    即不惩罚——那种状态交给倾倒终止去管。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    ground, has_ground = ground_under_point(sensor, asset.data.root_pos_w[:, :2], root_z, max_drop)
    error = torch.square(root_z - (ground + target_height))
    return torch.where(has_ground, error, torch.zeros_like(error))


def foot_clearance_terrain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """``foot_clearance_reward`` 的地形相对版：脚高 = 脚 z − 脚底地面 z，而不是世界 z。

    ``asset_cfg.body_ids`` 必须按 [左脚, 右脚] 顺序解析（``.*ankle_roll.*`` 在 G1 里就是这个顺序）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, 2)
    ground_z = torch.stack(
        [
            _ground_height_under(env.scene.sensors[left_sensor_cfg.name], foot_z[:, 0]),
            _ground_height_under(env.scene.sensors[right_sensor_cfg.name], foot_z[:, 1]),
        ],
        dim=1,
    )
    rel_height = torch.clamp(foot_z - ground_z, min=0.0, max=1.0)  # 洞已在 _ground_height_under 里排除
    foot_z_target_error = torch.square(rel_height - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


# ---------------------------------------------------------------------------
# 终止
# ---------------------------------------------------------------------------
def root_height_below_minimum_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_drop: float = DEFAULT_MAX_DROP,
) -> torch.Tensor:
    """``root_height_below_minimum`` 的地形相对版。

    地面高度取高程扫描里**离躯干投影最近的有效射线**（见 ``ground_under_point``），
    不用整张图的均值：楼梯上均值能和脚下差 0.3 m 以上，用均值会把倒地判成站着。
    世界 z 版在倒金字塔楼梯底部会直接把站立判成终止。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    ground, has_ground = ground_under_point(sensor, asset.data.root_pos_w[:, :2], root_z, max_drop)
    # 整片悬空（跨在 gap 上）时不按高度终止，交给 bad_orientation
    return has_ground & (root_z - ground < minimum_height)
