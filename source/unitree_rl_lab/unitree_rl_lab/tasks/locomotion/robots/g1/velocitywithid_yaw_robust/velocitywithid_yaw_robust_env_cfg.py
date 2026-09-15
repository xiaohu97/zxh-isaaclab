"""``VelocityWithIdYaw`` + 落地态抗扰随机化（2026-09-15）。

为什么
------
线上 walk（0914_yawfix，任务 ``VelocityWithIdYaw``）训练时的初始状态和扰动只有::

    reset_base.velocity_range   六个分量全 0
    reset_robot_joints          reset_joints_by_scale，默认关节速度为 0，等于没加
    push_robot                  每 5 s 一次，只有水平线速度 ±0.5 m/s

它只见过"直立、静止起步、偶尔被水平推一下"。jump3 落地后交给它的状态是骨盆倾角 0.5~0.6 rad、
机身角速度 1~2.2 rad/s、腿部关节速度 1~2 rad/s，完全在分布之外；仿真 17/17 接住是靠余量，
实机一次侧倾 −1.87 rad/s 就触发倾倒保护。

改了什么（只动 events，观测/动作/奖励/URDF 与 yaw 版完全一致，导出的 ONNX 可直接替换）
------------------------------------------------------------------------------------
* ``reset_base``：初始 roll/pitch ±0.15 rad；初始线速度 x/y ±0.5、z ±0.2 m/s，
  角速度 roll/pitch ±1.0、yaw ±0.8 rad/s。
* ``reset_robot_joints``：改用 ``reset_joints_by_offset``，关节位置 ±0.1 rad、关节速度 ±1.5 rad/s。
* ``push_robot``：间隔 3~6 s；线速度 x/y ±0.7、z ±0.3 m/s，角速度 roll/pitch ±1.0、yaw ±0.6 rad/s。

范围取自仿真/实机交接日志的量级再留余量；``bad_orientation`` 终止仍是 0.8 rad，倾角推过头会被判终止，
策略学的是"从这种状态里站回来"，不是硬扛。这只增加 walk 的余量，**不能替代** jump 的站稳尾段：
38° 躯干倾角 + 回弹的接手状态不是靠 walk 抗扰能可靠解决的。

怎么训
------
从 0914_yawfix 的 checkpoint 热启动（experiment_name 与 yaw 版相同，否则 ``--load_run`` 找不到）::

    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-VelocityWithIdYawRobust --headless \
        --resume --load_run 2026-09-13_21-25-15_clip10 --checkpoint model_14999.pt --max_iterations 5000

验收：``Episode_Termination/bad_orientation`` 不高于 yaw 版的 0.005 量级、``track_lin_vel_xy`` 不掉；
再用 artifacts/jump3_walk_handoff_20260915/simulation 的 walk→jump→walk 闭环跑一遍，
切换后倾角峰值与恢复时间应不劣于 0914_yawfix。
"""

from __future__ import annotations

import importlib

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid_yaw.rsl_rl_ppo_cfg import VelocityYawPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid_yaw.velocitywithid_yaw_env_cfg import (
    RobotEnvCfg as _YawEnvCfg,
    RobotPlayEnvCfg as _YawPlayEnvCfg,
)

# velocity 任务的包名是 ``29dof``，数字开头不是合法标识符，写不成 import 语句
_velocity_env_cfg = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")

# 落地态量级（见 deploy/robots/g1_29dof/log 与 artifacts/jump3_walk_handoff_20260915）：
#   仿真切换：倾角 0.44~0.51 rad，|ω| ≈ 1.1 rad/s，腿 |dq| 1.9 rad/s，根速 0.4~0.65 m/s
#   实机切换：倾角 0.59 rad，|ω| = 2.19 rad/s（roll −1.87），腿 |dq| 0.8 rad/s
RESET_POSE_RANGE = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-3.14, 3.14)}
RESET_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-1.0, 1.0),
    "pitch": (-1.0, 1.0),
    "yaw": (-0.8, 0.8),
}
PUSH_VELOCITY_RANGE = {
    "x": (-0.7, 0.7),
    "y": (-0.7, 0.7),
    "z": (-0.3, 0.3),
    "roll": (-1.0, 1.0),
    "pitch": (-1.0, 1.0),
    "yaw": (-0.6, 0.6),
}


@configclass
class EventCfg(_velocity_env_cfg.EventCfg):
    """其余项（摩擦、质量、执行器增益随机化）全部继承 ``velocity``。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": RESET_POSE_RANGE, "velocity_range": RESET_VELOCITY_RANGE},
    )

    # by_scale 是按默认值成比例缩放，默认关节速度为 0 所以缩放后仍是 0；by_offset 才是真的加速度扰动
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-1.5, 1.5)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class RobotEnvCfg(_YawEnvCfg):
    """URDF、观测、动作、奖励与 ``velocitywithid_yaw`` 相同，只换 events。"""

    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(_YawPlayEnvCfg):
    events: EventCfg = EventCfg()


@configclass
class VelocityYawRobustPPORunnerCfg(VelocityYawPPORunnerCfg):
    # 与 yaw 版共用实验目录，train.py 的 resume 才找得到 2026-09-13_21-25-15_clip10/model_14999.pt
    experiment_name = "unitree_g1_29dof_velocitywithidyaw"
    run_name = "robust"
