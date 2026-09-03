"""Run 任务专用观测。"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .gait_command import GaitCommand


def gait_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """8 维指令观测：速度保物理量纲，5 个步态参数按 limit_ranges 归一到 [-1, 1]。

    注意不要用 ``mdp.generated_commands`` 代替 —— 那个返回未归一化的原始指令，
    躯干高度(~0.75)、步频(~2)这类分量量纲差异会拖慢训练。
    """
    command: GaitCommand = env.command_manager.get_term(command_name)
    return command.command_obs


def gait_clock(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """步态时钟 (sin, cos)。

    相位从命令项取（那里是逐步积分的），**不能**像 locomotion.mdp.gait_phase 那样用
    ``episode_length_buf * dt % period`` 重算：本任务步频逐 env 可变，重算会得到
    与奖励项不一致的相位。
    """
    command: GaitCommand = env.command_manager.get_term(command_name)
    phase = command.phase * 2.0 * math.pi
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
