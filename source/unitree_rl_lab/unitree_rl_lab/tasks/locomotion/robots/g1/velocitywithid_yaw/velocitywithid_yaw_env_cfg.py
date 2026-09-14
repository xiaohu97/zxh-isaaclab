"""Unitree G1 29dof —— 带负载 + 解开转向约束的速度跟踪任务。

机器人 URDF 沿用 ``velocitywithid``（``g1_29dof_rev_1_0_identified0907.urdf``，37.341kg，
左手 2.5kg + 躯干 1.5kg 负载），**只动奖励里和转向相关的那几项**。

诊断依据（2026-09-13 实测，三个 run 在 iteration 32000 的 ``Episode_Reward``）::

    run                         track_lin_vel_xy   track_ang_vel_z   error_vel_yaw
    velocitywithid  (0907)        0.861 / 1.0        0.231 / 0.5         1.078
    velocity        (原厂)         0.862 / 1.0        0.244 / 0.5         1.011
    velocity 05-22  (id0521)      0.845 / 1.0        0.216 / 0.5         1.156

线速度跟踪拿到满分的 86%，角速度只拿到 47%；``error_vel_yaw`` 是按
``/ max_command_step`` (=500) 累加而 episode 有 1000 步，所以实际平均 yaw 误差约
``1.07 / 2 = 0.53 rad/s``，而指令范围只有 ±0.8 rad/s。三个 run 数值一致，**换 URDF 完全没动它**，
而且 10000 迭代后就进平台期（旧 run 跑到 49999 也没变好），所以这是奖励配置的平衡问题。

原因可以从奖励数值上直接算出来：humanoid 转向必须靠 hip_yaw 和 waist_yaw 出偏移，
而 ``velocity`` 把这两组都用 -1.0 罚着::

    joint_deviation_legs    -1.0  [.*_hip_roll_joint, .*_hip_yaw_joint]  ->  -0.152
    joint_deviation_waists  -1.0  [waist.*]                              ->  -0.068
                                                                 罚项合计  -0.220
    track_ang_vel_z          0.5                                         ->  +0.231

罚项几乎正好抵消整个转向奖励，所以"少转"就是当前配置下的最优解。

本任务相对 ``velocitywithid`` 改了 5 处，都是为了解掉这个抵消::

    track_ang_vel_z            weight  0.5  -> 1.0      转向和线速度同等重要
    joint_deviation_legs       joints  去掉 .*_hip_yaw_joint，只留 .*_hip_roll_joint（权重仍 -1.0）
    joint_deviation_hip_yaw    新增    -0.25  [.*_hip_yaw_joint]
    joint_deviation_waists     joints  waist.* -> 只留 waist_roll_joint / waist_pitch_joint（仍 -1.0）
    joint_deviation_waist_yaw  新增    -0.25  [waist_yaw_joint]

两个 yaw 自由度是**降权到 -0.25 而不是删掉**：完全不罚的话 hip_yaw 会变成廉价的"免费"自由度，
直行时也会外八／内扣，落脚点变差。hip_roll 和 waist_roll/pitch 保持 -1.0 不动——它们管的是
腿不要左右劈开、躯干不要歪，和转向无关，是站稳的主力。

要留意的地方：

1. 这是**三项耦合改动**（抬奖励 + 解两处罚），训出来只能说明"这组配置更会转"，不能归因到单独某一项。
   如果要干净的 ablation，先只改 ``track_ang_vel_z`` 跑一版。
2. ``track_ang_vel_z`` 抬到 1.0 后总奖励量级变了，``Train/mean_reward`` 不能直接和前两个 run 比，
   要比就比 ``Episode_Reward/track_ang_vel_z / weight`` 这个归一化值和 ``Metrics/.../error_vel_yaw``。
3. 预期副作用：转向变积极以后 ``feet_slide`` 和 ``action_rate`` 可能变差（转身时脚要搓地）。
   如果 ``error_vel_yaw`` 降了但 ``bad_orientation`` 终止率明显上升，说明转太猛，把两个 yaw 罚项
   从 -0.25 往 -0.5 收。
4. 按三个 run 的平台期证据，**15000 迭代就够了**，不用跑满 50000（起训时用 ``--max_iterations``
   指定，不改共享的 ``BasePPORunnerCfg``）。
5. 本任务用自己的 ``rsl_rl_ppo_cfg:VelocityYawPPORunnerCfg``（``BasePPORunnerCfg`` +
   ``clip_actions=10.0``）。第一次 run ``2026-09-13_16-32-08_yawfix`` 没有这条限幅，在 13900 迭代
   崩于 std NaN；转向改动本身当时已经收敛且明显更好（``error_vel_yaw`` 0.74 vs 基线 1.01~1.08），
   崩溃与奖励改动无关，是 velocity 系列一直缺 ``clip_actions`` 的老问题。详见那个文件的注释。

实测结果（yawfix run @13000，崩溃前的平台期）::

    指标                          本任务      velocitywithid   velocity 原厂
    error_vel_yaw                 0.738          1.078           1.011
    track_ang_vel_z / weight      0.607          0.462           0.487
    track_lin_vel_xy              0.877          0.861           0.862
    bad_orientation               0.0048         0.0047          0.0045

转向误差降 31%，线速度跟踪和稳定性均无代价。
"""
from __future__ import annotations

import importlib

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid.velocitywithid_env_cfg import (
    RobotEnvCfg as _BaseWithIdEnvCfg,
    RobotPlayEnvCfg as _BaseWithIdPlayEnvCfg,
)

# velocity 任务的包名是 ``29dof``，数字开头不是合法标识符，写不成 import 语句
_velocity_env_cfg = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
_BaseRewardsCfg = _velocity_env_cfg.RewardsCfg


@configclass
class RewardsCfg(_BaseRewardsCfg):
    """只覆盖／新增和转向相关的 5 项，其余全部继承 ``velocity``。"""

    # 转向和线速度同等重要（原来 0.5，被罚项抵消掉了）
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},  # math.sqrt(0.25)
    )

    # hip_yaw 从这里摘出去，只留 hip_roll（管腿不左右劈开，和转向无关，保持 -1.0）
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    # 降权而不是删掉：完全不罚的话直行时也会外八／内扣
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    # waist_yaw 同理摘出去，roll/pitch（管躯干不歪）保持 -1.0
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint", "waist_pitch_joint"])},
    )
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )


@configclass
class RobotEnvCfg(_BaseWithIdEnvCfg):
    """URDF 与 ``velocitywithid`` 相同（0907 带载版），只换奖励。"""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(_BaseWithIdPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
