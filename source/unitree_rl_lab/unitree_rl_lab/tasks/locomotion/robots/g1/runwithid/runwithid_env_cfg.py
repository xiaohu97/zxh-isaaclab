"""Unitree G1 29dof —— 辨识动力学版的可控步态跑步任务。

与 ``Unitree-G1-29dof-Run`` 的**唯一**差别是机器人 URDF：换成系统辨识得到的
``g1_29dof_rev_1_0_identified0907.urdf``。奖励、观测、指令空间、课程、PPO 超参全部复用
``run``，所以两个任务的结果差异可以干净地归因到动力学本身。改 run 的配置会自动同步到这里。

辨识版相对宇树原版 ``g1_29dof_rev_1_0.urdf`` 只改了两处质量（碰撞体与关节限位逐项比对完全一致）::

    torso_link        6.78 -> 8.28 kg   (+1.5,  躯干负载)
    left_rubber_hand  0.17 -> 2.67 kg   (+2.5,  左手负载)
    整机              33.341 -> 37.341 kg  (+12%)

两点需要留意：

1. **整机重 12%**，腿部力矩需求同比上升。膝/髋 roll 的 ``effort_limit_sim`` 是 139 N·m，
   高速档更容易顶到上限，``joint_vel_limits`` 和课程推进速度都可能比 run 差一截。
2. **负载只加在左手**，机器人左右不再对称。``run`` 的 ``arm_swing`` 奖励要求两臂等幅反相
   摆动（左肩 ``+A·cos``、右肩 ``-A·cos``），左臂现在末端多 2.5 kg，同样幅度所需力矩大得多，
   而肩关节 ``effort_limit_sim`` 只有 25 N·m。如果训练中 ``Episode_Reward/arm_swing`` 明显
   低于 run，先调 ``rewards.arm_swing.params`` 里的 ``vel_scale``/``max_amplitude`` 降幅度，
   而不是加权重。

URDF 不在本目录，而是直接引用 mimic 的 jumpwithid 任务里那一份：同一次系统辨识的结果只应
有一个副本，否则重新辨识后两个任务会在不同的动力学上训练而不自知。
"""
from __future__ import annotations

import os
import shutil

from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_ROS_DIR
from unitree_rl_lab.tasks.locomotion.robots.g1.run.run_env_cfg import (
    RobotEnvCfg as BaseRunEnvCfg,
    RobotPlayEnvCfg as BaseRunPlayEnvCfg,
)

# tasks/locomotion/robots/g1/runwithid -> tasks
_TASKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_URDF_PATH = os.path.join(
    _TASKS_DIR, "mimic", "robots", "g1_29dof", "jumpwithid", "g1_29dof_rev_1_0_identified0907.urdf"
)
# 任务专属的 stage 目录：和 jumpwithid 共用会在两个任务并行训练时互相覆盖
_STAGE_DIR = "/tmp/IsaacLab/unitree_rl_lab/runwithid"


def _stage_robot_urdf() -> str:
    """把辨识版 URDF 和 unitree_ros 的 meshes 拼到一个临时目录，再交给 Isaac Lab 转换。

    URDF 里的 mesh 路径写成 ``meshes/xxx.STL``，按 URDF 自身所在目录解析；meshes 有 100MB
    不进 git，所以用软链指回 unitree_ros。每次都重新拷贝 URDF，改完仓库里的文件直接生效
    （Isaac Lab 的转换缓存按文件内容 hash，内容一变会自动重新转换）。

    与 ``jumpwithid.tracking_env_cfg._stage_robot_urdf`` 同构，但用各自的 stage 目录。
    """
    if not UNITREE_ROS_DIR:
        raise RuntimeError(
            "UNITREE_ROS_DIR is empty. It is exported by the conda activate hook, so run "
            "`conda activate ustc_isaaclab` (or source "
            "envs/ustc_isaaclab/etc/conda/activate.d/unitree_paths.sh) before launching."
        )
    if not os.path.isfile(_URDF_PATH):
        raise RuntimeError(f"identified URDF not found: {_URDF_PATH}")
    meshes_src = os.path.join(UNITREE_ROS_DIR, "robots", "g1_description", "meshes")
    if not os.path.isdir(meshes_src):
        raise RuntimeError(f"G1 meshes directory not found: {meshes_src}")

    os.makedirs(_STAGE_DIR, exist_ok=True)
    meshes_link = os.path.join(_STAGE_DIR, "meshes")
    # islink 要放在前面：悬空软链的 os.path.exists() 是 False，但 symlink() 仍会报 FileExistsError
    if os.path.islink(meshes_link) or os.path.exists(meshes_link):
        os.remove(meshes_link)
    os.symlink(meshes_src, meshes_link)

    staged_urdf = os.path.join(_STAGE_DIR, os.path.basename(_URDF_PATH))
    shutil.copyfile(_URDF_PATH, staged_urdf)
    return staged_urdf


@configclass
class RobotEnvCfg(BaseRunEnvCfg):
    """除机器人 URDF 外与 ``run`` 完全一致。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = _stage_robot_urdf()


@configclass
class RobotPlayEnvCfg(BaseRunPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = _stage_robot_urdf()
