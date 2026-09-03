"""Run 任务课程：把指令区间从"走路"整体推向"跑步"。"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .gait_command import GaitCommand


def gait_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy",
    delta: float = 0.02,
    reward_ratio: float = 0.8,
) -> torch.Tensor:
    """速度跟踪达标就把课程进度推进一格（0 = 走路区间，1 = 跑步区间）。

    进度是**单个标量**，速度上限、步频上限、支撑相下限一起走：先学会走，再逐步逼出
    腾空期。比分别给每个量单独开课程简单得多，也不会出现"速度已经到 3 m/s 但支撑相
    还锁在 0.6"这种自相矛盾的中间状态。

    delta=0.02 时约需 50 次达标才推满。检查频率是每 ``max_episode_length`` 个 env step
    一次（≈40 个 PPO 迭代），所以推满大约 2000 个迭代。
    """
    command: GaitCommand = env.command_manager.get_term(command_name)
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * reward_ratio:
            command.progress = min(command.progress + delta, 1.0)

    return torch.tensor(command.progress, device=env.device)
