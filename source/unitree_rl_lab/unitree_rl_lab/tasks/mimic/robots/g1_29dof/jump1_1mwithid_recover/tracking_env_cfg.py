"""``jump1_1mwithid`` + 落地站稳尾段（2026-09-15）。

为什么要这个任务
----------------
线上 Mimic_Jump3（checkpoint 0915_28000，任务 ``jump1_1mwithid``）的参考 ``jump1_1m_waist15.npz``
在落地回弹中途就结束了：末帧根部速度 (−0.02, −0.51, +0.41) m/s、骨盆前倾 23.6°、躯干倾角 38.8°、
腿部关节速度 3.6 rad/s。训练用默认 ``motion_end_behavior="resample"``，到末帧直接重置，策略从没被
要求把身体停下来；部署侧 State_Mimic 又只按时长退出，walk 接手的就是这个回弹状态（实机切换瞬间
骨盆 33.9°、角速度 2.19 rad/s，1 s 内触发倾倒保护）。

改了什么
--------
1. 参考动作换成 ``jump1_1m_waist15_recover.npz``：前 90 帧与原参考逐帧相同，之后接
   0.16 s 刹车（关节从末帧速度减到零，脚不动）+ 0.64 s 收姿（到 walk 的 default_joint_pos、腰俯仰 0、
   双脚原地放平）+ 1.0 s 站立保持。根部位姿由"双脚位姿 + 关节角"经正运动学反推，脚不滑不穿地。
   生成链见同目录 README.md；``*.summary.json`` 记录帧数和一致性检查结果。
2. ``motion_end_behavior="hold"`` + ``motion_end`` 终止：episode 在最后一帧结束，而不是中途 resample。
   站立保持段因此成了策略必须完成的任务。
3. ``targeted_frame_range=(52, 末帧)``、``targeted_frame_probability=0.3``：30% 的 reset 落在
   落地 + 恢复段，其余仍走基类的自适应采样，起跳段不会被忘掉。
4. ``torso_tilt_landing`` 的窗口延长到末帧。

奖励、动作 clip、URDF（左手 2.5 kg 球）全部沿用 ``jump1_1mwithid``。

怎么训
------
从产出线上策略的 checkpoint 热启动（experiment_name 与 ``jump1_1mwithid`` 相同，否则 ``--load_run``
找不到那次 run）::

    python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Jump1-1mWithIdRecover --headless \
        --resume --load_run <产出 0915_28000 的 run 目录名> --checkpoint model_28000.pt \
        --max_iterations 6000

验收（play）：末帧之后 |ω| < 0.3 rad/s、骨盆倾角 < 10°、腿部 |dq| < 1 rad/s，且起跳高度/距离不退化
（``scripts/mimic/compare_mimic_runs.py`` 与原 run 对比）。
"""

from __future__ import annotations

import json
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import VELOCITY_RANGE
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.jump1_1mwithid.tracking_env_cfg import (
    RobotEnvCfg as _WithIdEnvCfg,
    TerminationsCfg as _WithIdTerminationsCfg,
)

_TASK_DIR = os.path.dirname(__file__)
MOTION_FILE = os.path.join(_TASK_DIR, "jump1_1m_waist15_recover.npz")
SUMMARY_FILE = os.path.join(_TASK_DIR, "jump1_1m_waist15_recover.summary.json")


def _summary() -> dict:
    try:
        with open(SUMMARY_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _num_frames() -> int:
    """优先读 NPZ 的真实帧数；NPZ 还没生成时退回 summary 的预期值，保证包能 import。"""
    try:
        import numpy as np

        return int(np.load(MOTION_FILE)["joint_pos"].shape[0])
    except Exception:  # noqa: BLE001 - 文件缺失/损坏都退回预期值
        return int(_S.get("expected_npz_frames_after_csv_to_npz", 180))


_S = _summary()
NUM_FRAMES = _num_frames()
TAIL_START_FRAME = int(_S.get("tail_start_frame", 90))
HOLD_START_FRAME = int(_S.get("hold_start_frame", 130))
LANDING_START_FRAME = 52  # 与 jump1_1mwithid 的 torso_tilt_landing 一致：落地段从第 52 帧起
LAST_FRAME = NUM_FRAMES - 1


@configclass
class CommandsCfg:
    """与 jump1_1mwithid 相同的跟踪 body、噪声范围，只换参考并改成 hold + 定向重置。"""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=MOTION_FILE,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        motion_end_behavior="hold",
        targeted_frame_range=(LANDING_START_FRAME, LAST_FRAME),
        targeted_frame_probability=0.3,
    )


@configclass
class TerminationsCfg(_WithIdTerminationsCfg):
    """落地倾角窗口延长到末帧；片段播完即结束 episode（time_out 语义，不当失败算）。"""

    torso_tilt_landing = DoneTerm(
        func=mdp.bad_body_orientation_in_motion_window,
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "threshold": 0.9,
            "frame_range": (LANDING_START_FRAME, LAST_FRAME),
        },
    )
    motion_end = DoneTerm(
        func=mdp.motion_clip_finished,
        params={"command_name": "motion"},
        time_out=True,
    )


@configclass
class RobotEnvCfg(_WithIdEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        # play 时停在最后一帧看它站不站得住，不结束 episode
        self.terminations.motion_end = None


@configclass
class Jump1_1mWithIdRecoverPPORunnerCfg(BasePPORunnerCfg):
    # 与 jump1_1mwithid 共用实验目录：train.py 的 resume 在 logs/rsl_rl/<experiment_name>/ 里
    # 按 --load_run 找 checkpoint，名字不同就热启动不到 0915_28000。
    experiment_name = "unitree_g1_29dof_mimic_jump1_1mwithid"
    run_name = "recover"
