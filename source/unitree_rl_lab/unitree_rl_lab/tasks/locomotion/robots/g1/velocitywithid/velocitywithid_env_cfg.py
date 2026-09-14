"""Unitree G1 29dof —— 带负载动力学版的速度跟踪任务。

与 ``Unitree-G1-29dof-Velocity`` 的**唯一**差别是机器人 URDF：换成
``g1_29dof_rev_1_0_identified0907.urdf``。奖励、观测、指令空间、课程、PPO 超参全部复用
``29dof/velocity_env_cfg.py``，所以两个任务的结果差异可以干净地归因到动力学本身。
改 velocity 的配置会自动同步到这里。

0907 版相对宇树原版 ``g1_29dof_rev_1_0.urdf`` 只改了两个 link（35 个 link 里其余 33 个的
质量/质心/惯量张量逐项完全一致，碰撞体与关节限位也一致）::

    torso_link         6.78 -> 8.28 kg   (+1.5,  躯干负载)
    left_rubber_hand   0.17 -> 2.67 kg   (+2.5,  左手负载)
    整机              33.341 -> 37.341 kg  (+12%)

所以这个任务的名字里虽然带 "Id"，实际换的是**负载**而不是重新辨识的惯量；真机空手行走时
应该训 ``Unitree-G1-29dof-Velocity``（它指向的全局 URDF 已经是原厂 33.341kg 版），
只有左手确实拎着 2.5kg 时才用这个任务。

三点需要留意：

1. **整机重 12%**，腿部力矩需求同比上升。膝/髋 roll 的 ``effort_limit_sim`` 是 139 N·m，
   ``lin_vel_cmd_levels`` 课程推进到高速档会比 velocity 慢一截，最终能跟踪的
   ``lin_vel_x`` 上限也可能达不到 limit_ranges 的 1.5 m/s。
2. **负载只加在左手，机器人左右不再对称**。质心向左偏，策略必须学会用髋 roll / 踝 roll
   常态性地补一个侧倾力矩。``joint_deviation_legs`` (-1.0, 作用在 hip_roll / hip_yaw) 和
   ``flat_orientation_l2`` (-5.0) 会和这个补偿直接冲突，如果训练中这两项的
   ``Episode_Reward`` 明显比 velocity 差、且 ``track_lin_vel_xy`` 上不去，先降
   ``joint_deviation_legs`` 的权重，而不是加 orientation 的权重。
3. ``events.add_base_mass`` 仍然在 torso_link 上随机 (-1.0, 3.0) kg，是叠加在已经 +1.5 的
   8.28 kg 之上的域随机化，不用改。

URDF 不在本目录，而是直接引用 mimic 的 jumpwithid 任务里那一份（runwithid 也引用同一份）：
同一份负载模型只应有一个副本，否则改了以后各任务会在不同的动力学上训练而不自知。
"""
from __future__ import annotations

import importlib
import os
import shutil

from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_ROS_DIR

# velocity 任务的包名是 ``29dof``，以数字开头不是合法的 Python 标识符，写不成 import 语句
# （gym.register 里那个 entry_point 字符串也是同样的原因才能工作）。importlib 按字符串导入没这个限制。
_velocity_env_cfg = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
_BaseVelocityEnvCfg = _velocity_env_cfg.RobotEnvCfg
_BaseVelocityPlayEnvCfg = _velocity_env_cfg.RobotPlayEnvCfg

# tasks/locomotion/robots/g1/velocitywithid -> tasks
_TASKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_URDF_PATH = os.path.join(
    _TASKS_DIR, "mimic", "robots", "g1_29dof", "jumpwithid", "g1_29dof_rev_1_0_identified0907.urdf"
)
# 任务专属的 stage 目录：和 jumpwithid / runwithid 共用会在并行训练时互相覆盖
_STAGE_DIR = "/tmp/IsaacLab/unitree_rl_lab/velocitywithid"


def _stage_robot_urdf() -> str:
    """把带负载的 URDF 和 unitree_ros 的 meshes 拼到一个临时目录，再交给 Isaac Lab 转换。

    URDF 里的 mesh 路径写成 ``meshes/xxx.STL``，按 URDF 自身所在目录解析；meshes 有 100MB
    不进 git，所以用软链指回 unitree_ros。每次都重新拷贝 URDF，改完仓库里的文件直接生效
    （Isaac Lab 的转换缓存按文件内容 hash，内容一变会自动重新转换）。

    与 ``runwithid._stage_robot_urdf`` 同构，但用各自的 stage 目录。
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
class RobotEnvCfg(_BaseVelocityEnvCfg):
    """除机器人 URDF 外与 ``velocity`` 完全一致。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = _stage_robot_urdf()


@configclass
class RobotPlayEnvCfg(_BaseVelocityPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.asset_path = _stage_robot_urdf()
