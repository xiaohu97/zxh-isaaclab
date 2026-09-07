"""Run 任务课程：把指令区间从"走路"整体推向"跑步"。

2026-09-03 第一次训练的教训
---------------------------
旧判据照抄仓库原 ``lin_vel_cmd_levels``：``episode_sums / max_episode_length_s > weight * 0.8``。
这个量 = 权重 × 每步平均核值 × episode 完成率。完成率 0.88 时上限只有 1.32，要过 1.2
需要平均核值 > 0.91（速度误差全程 < 0.15 m/s）。结果 13430 个迭代 progress 一次都没动，
全程在走路区间训练。原任务能推进只是因为它从 ±0.1 m/s 起步、几乎能完美跟踪。

现在的判据把两个量拆开、各自有物理含义，并且用 **区间内全部 reset 事件** 做统计而不是
"恰好在 1000 的整数倍那一步 reset 的 env"（那批样本偏向刚好超时的好 env）：

- ``mean_kernel``：存活期间每步平均跟踪核值（``episode_sums / (实际时长 × weight)``）
- ``survival``：以超时结束的 episode 占比（= 没摔）

两者都过线才推进；核值明显低于线则回退一格（防止推进过头后卡死在不可行区间）。
三个量都进 TensorBoard（``Curriculum/gait_cmd_levels/{progress,mean_kernel,survival}``）。
"""
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
    kernel_threshold: float = 0.65,
    survival_threshold: float = 0.6,
    regress_margin: float = 0.15,
    check_interval_steps: int | None = None,
) -> dict[str, float]:
    """跟踪够好且不摔就把课程进度推进一格（0 = 走路区间，1 = 跑步区间）。

    进度是**单个标量**，速度上限、步频上限、支撑相下限一起走：先学会走，再逐步逼出
    腾空期。

    Args:
        delta: 每次达标推进的进度量。0.02 -> 至少 50 次检查才推满。
        kernel_threshold: 平均跟踪核值门槛。exp(-e²/0.25) = 0.65 对应速度误差 ≈ 0.33 m/s。
        survival_threshold: 超时结束（没摔）的 episode 占比门槛。
        regress_margin: 核值低于 ``kernel_threshold - regress_margin`` 时回退一格。
        check_interval_steps: 每隔多少个 env step 检查一次；默认一个 episode 长度。
    """
    command: GaitCommand = env.command_manager.get_term(command_name)
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    stats = command.curriculum_stats

    # ---- 累积本批 reset env 的统计（每次 reset 都会进来，累到张量里，不做 GPU 同步）
    episode_len = env.episode_length_buf[env_ids]
    valid = episode_len > 0  # 刚建环境时的首次 reset，episode 长度为 0，没有信息
    if valid.any():
        episode_s = episode_len[valid] * env.step_dt
        sums = env.reward_manager._episode_sums[reward_term_name][env_ids][valid]
        kernel = sums / (episode_s * reward_term.weight)
        timed_out = env.termination_manager.time_outs[env_ids][valid].float()
        stats["kernel_sum"] += kernel.sum()
        stats["timeout_sum"] += timed_out.sum()
        stats["count"] += float(kernel.numel())

    # ---- 到检查点：算区间内的均值，决定推进 / 回退 / 不动
    interval = check_interval_steps or env.max_episode_length
    if env.common_step_counter - stats["last_check"] >= interval and stats["count"].item() > 0:
        mean_kernel = (stats["kernel_sum"] / stats["count"]).item()
        survival = (stats["timeout_sum"] / stats["count"]).item()
        stats["mean_kernel"], stats["survival"] = mean_kernel, survival

        if mean_kernel > kernel_threshold and survival > survival_threshold:
            command.progress = min(command.progress + delta, 1.0)
        elif mean_kernel < kernel_threshold - regress_margin:
            command.progress = max(command.progress - delta, 0.0)

        stats["kernel_sum"].zero_()
        stats["timeout_sum"].zero_()
        stats["count"].zero_()
        stats["last_check"] = env.common_step_counter

    return {
        "progress": command.progress,
        "mean_kernel": stats["mean_kernel"],
        "survival": stats["survival"],
    }
