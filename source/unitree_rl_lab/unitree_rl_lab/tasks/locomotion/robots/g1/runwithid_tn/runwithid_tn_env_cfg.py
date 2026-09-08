"""Unitree G1 29dof —— 辨识动力学 + 真实电机力矩-转速模型的跑步任务。

``RunWithId`` 的平行对照任务：辨识版 URDF、奖励、观测、指令空间、课程、PPO 超参全部继承，
**唯一差别是执行器模型**，所以两者的结果差异可以干净地归因到电机模型本身。

    Run          原厂 URDF (33.341 kg) + ImplicitActuatorCfg（常数力矩上限）
    RunWithId    辨识 URDF (37.341 kg) + ImplicitActuatorCfg（常数力矩上限）
    RunWithIdTN  辨识 URDF (37.341 kg) + UnitreeActuatorCfg（T-N 曲线 + 摩擦）   <- 本任务

隐式执行器的力矩上限是常数：膝关节在 0 rad/s 和 19 rad/s 都能出满 139 N·m。真实电机接近空载
转速时力矩掉到 0。跑步高速档恰恰是膝关节转速最高的时候，所以隐式模型会系统性高估可达速度。
换成 T-N 曲线后：膝低速段 111 N·m（比原来低 20%），14.5 rad/s 之后开始衰减，22.7 rad/s 归零。
详见 ``UNITREE_G1_29DOF_TN_CFG`` 的注释。

预期这个任务学出来的速度低于 ``RunWithId``。两者跑同样迭代数后用同一组指令回放对比，
差值就是"仿真高估了多少"。注意课程会自己停在硬件真正够得着的地方，所以更该看
``Metrics/base_velocity/cmd_max_lin_vel_x`` 停在哪，而不是只看最终回报。

训练会比 RunWithId 慢：显式执行器每步要在 Python 里算 PD + 限幅 + 摩擦，隐式的这些都在 PhysX 内部。

动作延迟没有打开（``min_delay``/``max_delay`` 保持 0）。``UnitreeActuator`` 继承自
``DelayedPDActuator``，支持延迟，但那是另一个 sim2real 维度，一起改会让对比无法归因。
"""
from __future__ import annotations

import copy

from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_TN_CFG
from unitree_rl_lab.tasks.locomotion.robots.g1.runwithid.runwithid_env_cfg import (
    RobotEnvCfg as BaseRunWithIdEnvCfg,
    RobotPlayEnvCfg as BaseRunWithIdPlayEnvCfg,
)


def _apply_tn_actuators(scene_robot) -> None:
    """把执行器换成带 T-N 曲线的真实电机模型。

    deepcopy 是必须的：``UNITREE_G1_29DOF_TN_CFG.actuators`` 是模块级共享对象，直接赋值的话
    Isaac Lab 在 ``_process_actuators_cfg`` 里对 cfg 的就地修改会污染它，同进程里再创建环境
    （比如 play 脚本先建训练 env 再建 play env）就会拿到被改过的配置。
    """
    scene_robot.actuators = copy.deepcopy(UNITREE_G1_29DOF_TN_CFG.actuators)


@configclass
class RobotEnvCfg(BaseRunWithIdEnvCfg):
    """除执行器模型外与 ``RunWithId`` 完全一致。"""

    def __post_init__(self):
        super().__post_init__()  # RunWithId: 设置辨识版 URDF
        _apply_tn_actuators(self.scene.robot)


@configclass
class RobotPlayEnvCfg(BaseRunWithIdPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_tn_actuators(self.scene.robot)
