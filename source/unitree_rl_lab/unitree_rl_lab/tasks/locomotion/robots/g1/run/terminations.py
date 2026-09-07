"""Run 任务专用终止项：数值安全阀。

2026-09-07 的 run 2026-09-06_21-49-17 在第 30794 迭代崩于 ``normal expects all elements of std >= 0.0``。
TensorBoard 回溯：第 27404 迭代 ``action_rate`` 单项跳到 -1.7e4/s（正常 -1.1），mean_reward -5.5e5，
value loss 2e11；之后 30005 起反复出现，最终把网络参数打成 NaN。``action_rate_l2`` 是网络输出的
差分，不是物理量 —— 说明某个 env 的观测里出现了离群值（关节速度尖峰之类），策略输出了天文数字的
动作，再经 ``last_action`` 观测正反馈回去。物理本身并没有 NaN（回放 256 env x 20 s 零 NaN）。

对策分三层：动作裁剪（``RunPPORunnerCfg.clip_actions``）、观测裁剪（``ObsTerm.clip``）、这里的
状态安全阀 —— 任何 env 一旦出现非有限值或荒谬的关节速度，立刻结束该 episode，别让它进 batch。
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def state_not_finite(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """根状态或关节状态出现 NaN/Inf 就终止。

    注意 ``x > limit`` 这类比较对 NaN 恒为 False，普通的限位终止抓不到 NaN，必须单独用 isfinite。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    bad_root = ~torch.isfinite(asset.data.root_state_w).all(dim=1)
    bad_joint = ~(torch.isfinite(asset.data.joint_pos).all(dim=1) & torch.isfinite(asset.data.joint_vel).all(dim=1))
    return bad_root | bad_joint
