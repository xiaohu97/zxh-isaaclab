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
# 奖励
# ---------------------------------------------------------------------------
def _ground_height_under(sensor: RayCaster, fallback: torch.Tensor) -> torch.Tensor:
    """脚底小网格的平均命中高度；全部打空（悬在坑上）时退回 ``fallback``。"""
    hits = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    count = valid.sum(dim=1)
    mean = torch.where(valid, hits, torch.zeros_like(hits)).sum(dim=1) / count.clamp(min=1)
    return torch.where(count > 0, mean, fallback)


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
    rel_height = torch.clamp(foot_z - ground_z, min=0.0, max=1.0)
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
) -> torch.Tensor:
    """``root_height_below_minimum`` 的地形相对版。

    地面高度取高程扫描里**离躯干投影最近的那条射线**，不用整张图的均值：楼梯上均值能和脚下
    差 0.3 m 以上，用均值会把倒地判成站着。世界 z 版在倒金字塔楼梯底部会直接把站立判成终止。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    hits = sensor.data.ray_hits_w
    dist_xy = torch.norm(torch.nan_to_num(hits[..., :2], nan=1.0e6, posinf=1.0e6, neginf=-1.0e6) - root_xy[:, None, :], dim=-1)
    nearest = torch.argmin(dist_xy, dim=1)
    ground_z = hits[torch.arange(hits.shape[0], device=hits.device), nearest, 2]
    ground_z = torch.where(torch.isfinite(ground_z), ground_z, asset.data.root_pos_w[:, 2] - minimum_height - 1.0)
    return asset.data.root_pos_w[:, 2] - ground_z < minimum_height
