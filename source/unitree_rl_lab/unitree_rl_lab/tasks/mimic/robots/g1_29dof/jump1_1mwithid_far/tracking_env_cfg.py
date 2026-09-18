"""``jump1_1mwithid_recover`` + 弹道自洽的起跳/腾空段（2026-09-16）。

为什么
------
线上参考 ``jump1_1m_waist15_recover.npz`` 的腾空段物理上不可实现，实测：

* 腾空竖直加速度 -5.31 m/s²（弹道应 -9.81），水平速度 1.06→2.67 m/s 变化（应恒定）
* 蹬伸只有 0.08 s、膝角 87.7°→85.6°，**没有蹬地动作**；根部竖直速度峰值出现在离地后

策略因此学不到蹬地。干净测量（1024 次，第 0 帧起跳、关闭随机化）：实际跳 0.973 m，
而参考是 1.206 m（同口径），差 19%；髋俯仰在蹬伸窗口 26% 时间顶满 88 N·m，
膝只用到 74/139 N·m、13.4/20 rad/s —— 短板在髋不在膝。

改了什么
--------
1. 参考换成 ``jump1_1m_ballistic130.npz``（``scripts/mimic/retime_jump_ballistic_g1.py`` 生成）：
   蓄力段再蹲深 6 cm（蹬伸行程 0.19→0.268 m，``a=v²/(2h)`` 才放得开），蹬伸 0.18 s 把膝从
   88° 蹬到 2.2°，腾空按真弹道积分。同口径距离 **1.206 → 1.401 m**，腾空竖直加速度 -9.81、
   水平速度恒定 2.05 m/s。落地+恢复尾段 113 帧原样继承。
2. ``torque_headroom`` 奖励：落地窗口内只罚超过力矩上限 80% 的部分。落地竖直速度从 0.88
   提到 2.23 m/s（动能 6.4 倍），而腰俯仰 25 N·m 在旧参考落地时已经饱和 67% —— 不加约束
   会把"落地前倾过大"重新放大。逼策略用髋膝屈曲吸能，而不是靠腰硬撑。
3. （v1 曾换 T-N 电机模型，v2 撤回，见下方失败记录。）
4. ``torso_tilt_landing`` 窗口起点按新帧号改到 56（腾空段 48..69，触地 70）。参考自身在
   56..末帧的最大躯干倾角 44.6°，阈值取 0.95 rad(54.4°) 留 10°，且仍低于部署保护 57.3°。

第一轮(far, 2026-09-16 14:32 / 16:57)失败记录
--------------------------------------------
两个 far checkpoint **起跳率 0 %**（位移 0.45-0.55 m，足端最高 4.5-6 cm），而训练曲线全程向好
（motion_end 98.7 %、error_body_pos 0.044）。消融：recover 策略换 T-N 仍跳（87 %，距离 -21 %）；
far 策略换回隐式仍不跳 —— 是策略学成了"贴地挪完整段"，不是电机。根因是同时换了参考和电机模型，
热启动策略第一步就蹬不起来，而 MDP 里没有任何东西惩罚"不跳"（见 ``feet_grounded_in_motion_window``
的说明）。v2 改法：

* 加 ``must_jump`` 终止：各级参考"离地 ≥0.15 m"帧段内脚底离地低于门槛即终止（门槛 C1/C2/C3 = 0.05/0.08/0.10 m）。
* 课程三级 C1(1.00 m) → C2(1.15 m) → C3(1.30 m)：recover 在 1.30 m 参考上起跳率仅 1 %、must_jump 触发 91 %，
  直接热启动没有成功样本；每级从上一级 checkpoint 热启动。
* 执行器回到隐式（与 recover 相同），T-N 留到第二阶段单独切、从本任务训好的策略热启动，
  预期距离再掉 ~20 %（recover 消融实测 1.218 → 0.958 m），那是真机数字。
* ``ee_body_pos`` 恢复基类 0.55（放宽到 0.75 的方向是错的）。

怎么训
------
从 recover 的 checkpoint 热启动（experiment_name 与 jump1_1mwithid 相同，否则 --load_run 找不到）::

    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar --headless \
        --num_envs 8192 --resume --load_run 2026-09-15_18-28-51_recover --checkpoint model_34500.pt \
        --max_iterations 8000

验收：干净测量（frame_zero_probability=1、关随机化）距离 > 1.15 m；落地窗口 torque_headroom
接近 0；收尾 |ω|<0.3 rad/s、骨盆倾角<10°、腿 |dq|<1 rad/s 的达标率不低于 recover 的 66%。
"""

from __future__ import annotations

import json
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.jump1_1mwithid_recover.tracking_env_cfg import (
    CommandsCfg as _RecoverCommandsCfg,
    RobotEnvCfg as _RecoverEnvCfg,
    TerminationsCfg as _RecoverTerminationsCfg,
)
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.jump1_1mwithid.tracking_env_cfg import RewardsCfg as _WithIdRewardsCfg

_TASK_DIR = os.path.dirname(__file__)

# 课程三级：recover 策略在 1.30 m 参考上起跳率只有 1 %（must_jump 触发 91 %），直接热启动没有成功样本。
# 中间参考由 retime_jump_ballistic_g1.py 生成，蹬伸段从接近原动作逐级过渡到 1.30 m 版。
# must_jump 的窗口取各参考"离地 ≥0.15 m"的帧段内缩 1 帧，门槛逐级抬。
# recover 策略在 C1(1.20 m) 上热启动实测：起跳率 0 %，must_jump 触发 94 % —— 硬终止在第一级拿不到
# 成功样本。所以 C1 不用硬终止，改用稠密的 flight_clearance 奖励把脚"托"起来；C2/C3 再上硬终止。
#
# lift_w 必须按"满分在总回报里的占比"反推，不能凭感觉给：奖励只在 12~14 帧的腾空窗口生效，
# 而跟踪奖励全程 174 帧累积。第一次给 3.0，满分记录值 = 3.0*12*0.02/30 = 0.024，占 mean_reward
# (6.75) 仅 0.36 % —— 跑了 4400 迭代 flight_clearance 从 0.0025 到 0.0027 纹丝不动。
# 60 对应约 7 %，30/20 用于 C2/C3（那时已会跳，硬终止兜底，不需要这么强的诱导）。
STAGES = {
    "c1": dict(npz="jump1_1m_ballistic100.npz", window=None, clearance=0.05, hard=False, lift_w=60.0, run_name="far_c1"),
    # c1u：同 c1 但起跳骨盆前倾 35°->~20°（--takeoff-pitch-deg 18）。C1/C2 实测蹬伸窗口腰俯仰 91% 时间
    # 顶死 25 N·m（膝只用到 57~66 %、速度 40~64 %），腰是动力链短板；躯干更直 -> 腰的力矩臂更短。
    "c1u": dict(npz="jump1_1m_ballistic100_p18.npz", window=None, clearance=0.05, hard=False, lift_w=60.0, run_name="far_c1u"),
    # c1f：躯干直(18°) + 目标距离 1.25。C1U 证明改直躯干把腰俯仰蹬伸饱和从 91.5% 压到 72.6%、
    # 起跳 vz 从 0.62 提到 1.16、足端高 0.503，但位移没涨——因为 c1u 的参考只要求 0.99 m。
    # 膝仍剩 22~37% 力矩、46% 速度余量，所以这一级把距离要求提上去，让腿把余量用掉。
    "c1f": dict(npz="jump1_1m_ball125_p18.npz", window=None, clearance=0.05, hard=False, lift_w=60.0, run_name="far_c1f"),
    "c2": dict(npz="jump1_1m_ballistic115.npz", window=None, clearance=0.08, hard=True, lift_w=30.0, run_name="far_c2"),
    "c3": dict(npz="jump1_1m_ballistic130.npz", window=(53, 65), clearance=0.10, hard=True, lift_w=20.0, run_name="far_v2"),
}


def _push_frames(npz_path: str) -> tuple[int, int]:
    """(蓄力最低帧, 离地首帧)：定向重置窗口用。"""
    try:
        import numpy as np

        P = np.load(npz_path)["body_pos_w"]
        fz = np.minimum(P[:, 18, 2], P[:, 19, 2])
        lift = int(np.where(fz - fz[:10].min() > 0.02)[0].min())
        return int(P[:lift, 0, 2].argmin()), lift
    except Exception:  # noqa: BLE001
        return (33, 39)


def _flight_window(npz_path: str, clearance: float = 0.15) -> tuple[int, int]:
    """参考里双脚离地 ≥clearance 的帧段（内缩 1 帧），作为 must_jump 的检查窗口。"""
    try:
        import numpy as np

        P = np.load(npz_path)["body_pos_w"]
        fz = np.minimum(P[:, 18, 2], P[:, 19, 2])
        idx = np.where(fz - fz[:10].min() >= clearance)[0]
        return int(idx.min()) + 1, int(idx.max()) - 1
    except Exception:  # noqa: BLE001
        return (53, 65)


def _num_frames(npz_path: str, fallback: int = 184) -> int:
    try:
        import numpy as np

        return int(np.load(npz_path)["joint_pos"].shape[0])
    except Exception:  # noqa: BLE001
        return fallback


def make_stage(stage: str, prefix: str):
    st = STAGES[stage]
    MOTION_FILE = os.path.join(_TASK_DIR, st["npz"])
    LAST_FRAME = _num_frames(MOTION_FILE) - 1
    JUMP_WINDOW = st["window"] or _flight_window(MOTION_FILE)
    st = dict(st)
    st["crouch"], st["lift"] = _push_frames(MOTION_FILE)
    LANDING_START_FRAME = JUMP_WINDOW[0] + 3      # 腾空后半段起（此前躯干还在收腿大俯仰）
    TORQUE_WINDOW = (JUMP_WINDOW[1] + 2, LAST_FRAME)
    # 定向重置指向"蓄力最低点前 4 帧 ~ 离地首帧"：要学的是蹬地，就得多采到正处在蹬地的样本。
    # 之前指向落地段(48,173)、概率 0.3，等于 70 % 的样本落在站着不动的尾段，蹬伸段几乎采不到。
    PUSH_WINDOW = (max(st["crouch"] - 4, 0), st["lift"])

    @configclass
    class CommandsCfg(_RecoverCommandsCfg):
        motion = _RecoverCommandsCfg().motion.replace(
            motion_file=MOTION_FILE,
            targeted_frame_range=PUSH_WINDOW,
            targeted_frame_probability=0.5,
        )


    @configclass
    class TerminationsCfg(_RecoverTerminationsCfg):
        must_jump = None if not st["hard"] else DoneTerm(
            func=mdp.feet_grounded_in_motion_window,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                # 参考在 52~66 帧离地 ≥0.20 m；窗口取 53~65，门槛 0.10 m：
                # 跳到参考六成高度(recover 在旧参考上的水平)就能过，贴地挪(4.5~6 cm)必死
                "frame_range": JUMP_WINDOW,
                "min_clearance": st["clearance"],
            },
        )
        torso_tilt_landing = DoneTerm(
            func=mdp.bad_body_orientation_in_motion_window,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                # 参考自身在该窗口最大 44.6°，0.95 rad = 54.4° 留 10°，仍低于部署保护 57.3°
                "threshold": 0.95,
                "frame_range": (LANDING_START_FRAME, LAST_FRAME),
            },
        )


    @configclass
    class RewardsCfg(_WithIdRewardsCfg):
        flight_clearance = RewTerm(
            func=mdp.flight_clearance_in_motion_window,
            weight=st["lift_w"],
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                "frame_range": JUMP_WINDOW,
                "target_clearance": 0.20,
            },
        )
        torque_headroom = RewTerm(
            func=mdp.torque_headroom_in_motion_window,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_pitch_joint", ".*_ankle_pitch_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
                ),
                "command_name": "motion",
                "frame_range": TORQUE_WINDOW,
                "start_ratio": 0.8,
            },
        )


    @configclass
    class RobotEnvCfg(_RecoverEnvCfg):
        commands: CommandsCfg = CommandsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        rewards: RewardsCfg = RewardsCfg()

        # v2：执行器沿用 recover 的隐式模型；T-N 作为第二阶段单独切换（见文件头）


    class RobotPlayEnvCfg(RobotEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.episode_length_s = 1e9
            self.terminations.motion_end = None


    @configclass
    class RunnerCfg(BasePPORunnerCfg):
        # 与 jump1_1mwithid 共用实验目录，resume 才找得到 recover / 上一级的 checkpoint
        experiment_name = "unitree_g1_29dof_mimic_jump1_1mwithid"
        run_name = st["run_name"]

    # train.py 会 pickle env_cfg（params/env.pkl）：闭包里定义的类 pickle 不了
    # （"Can't pickle local object make_stage.<locals>.RobotEnvCfg"）。把每个类的 __qualname__
    # 改成模块级名字并在下面绑定同名属性，pickle 就能按 module.qualname 找到它。
    for cls, name in ((CommandsCfg, "CommandsCfg"), (TerminationsCfg, "TerminationsCfg"),
                      (RewardsCfg, "RewardsCfg"), (RobotEnvCfg, "EnvCfg"),
                      (RobotPlayEnvCfg, "PlayEnvCfg"), (RunnerCfg, "RunnerCfg")):
        cls.__qualname__ = f"{prefix}_{name}"
        cls.__module__ = __name__
        globals()[cls.__qualname__] = cls
    return RobotEnvCfg, RobotPlayEnvCfg, RunnerCfg


C1_EnvCfg, C1_PlayEnvCfg, C1_RunnerCfg = make_stage("c1", "C1")
C1U_EnvCfg, C1U_PlayEnvCfg, C1U_RunnerCfg = make_stage("c1u", "C1U")
C1F_EnvCfg, C1F_PlayEnvCfg, C1F_RunnerCfg = make_stage("c1f", "C1F")
C2_EnvCfg, C2_PlayEnvCfg, C2_RunnerCfg = make_stage("c2", "C2")
RobotEnvCfg, RobotPlayEnvCfg, Jump1_1mWithIdFarPPORunnerCfg = make_stage("c3", "C3")
