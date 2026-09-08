from __future__ import annotations

import os
import shutil

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_ROS_DIR
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import (
    VELOCITY_RANGE,
    EventCfg as BaseEventCfg,
    RewardsCfg as BaseRewardsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    TerminationsCfg as BaseTerminationsCfg,
)


_TASK_DIR = os.path.dirname(__file__)
_URDF_NAME = "g1_29dof_rev_1_0_identified0907.urdf"  # 0521 版仍保留在本目录, 改这里即可切回
_STAGE_DIR = "/tmp/IsaacLab/unitree_rl_lab/jumpwithid"


def _stage_robot_urdf() -> str:
    """把本目录的 URDF 和 unitree_ros 的 meshes 拼到一个临时目录，再交给 Isaac Lab 转换。

    URDF 里的 mesh 写成 ``meshes/xxx.STL``，按 URDF 自身所在目录解析。meshes 有 100MB，
    不适合进 git，所以只把 URDF 放仓库里（方便 git 同步和改动），meshes 用软链指回
    unitree_ros。每次都重新拷贝 URDF，改完仓库里的文件直接生效；Isaac Lab 的转换缓存
    按文件字节做 hash，内容一变会自动重新转换。

    用任务专属的 stage 目录，而不是 ``UnitreeUrdfFileCfg.replace_asset()`` 里那个共享的
    ``/tmp/IsaacLab/unitree_rl_lab/robot.urdf``，否则两个任务并行训练时会互相覆盖。
    """
    if not UNITREE_ROS_DIR:
        raise RuntimeError(
            "UNITREE_ROS_DIR is empty. It is exported by the conda activate hook, so run "
            "`conda activate ustc_isaaclab` (or source "
            "envs/ustc_isaaclab/etc/conda/activate.d/unitree_paths.sh) before launching."
        )
    meshes_src = os.path.join(UNITREE_ROS_DIR, "robots", "g1_description", "meshes")
    if not os.path.isdir(meshes_src):
        raise RuntimeError(f"G1 meshes directory not found: {meshes_src}")

    os.makedirs(_STAGE_DIR, exist_ok=True)
    meshes_link = os.path.join(_STAGE_DIR, "meshes")
    # islink 要放在前面：悬空软链的 os.path.exists() 是 False，但 symlink() 仍会报 FileExistsError
    if os.path.islink(meshes_link) or os.path.exists(meshes_link):
        os.remove(meshes_link)
    os.symlink(meshes_src, meshes_link)

    staged_urdf = os.path.join(_STAGE_DIR, _URDF_NAME)
    shutil.copyfile(os.path.join(_TASK_DIR, _URDF_NAME), staged_urdf)
    return staged_urdf


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/../jump1/jump1.npz",
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
    )


@configclass
class EventCfg(BaseEventCfg):
    """Reduced randomization for the first jumpwithid training stage."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 32,
        },
    )
    add_joint_default_pos = None
    base_com = None
    push_robot = None


@configclass
class TerminationsCfg(BaseTerminationsCfg):
    """Looser terminations so the policy can see longer jumpwithid segments early."""

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.8},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.5},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 1.2,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        },
    )


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Softer regularization for jumpwithid warmup."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5e-2)


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 用本任务目录里的 0907 辨识版 URDF（整机 37.341kg = 原厂 33.341 + 左手负载 2.5 + 躯干 +1.5；
        # 腿/骨盆/右臂保持原厂对称值，见文件头注释）。0521 版（37.662kg，左右腿不对称，
        # jump1 那批 run 实际训练用的动力学）仍在本目录，切回改 _URDF_NAME 即可；
        # 不走 assets/robots/unitree.py 全局指向的 g1_29dof_rev_1_0.urdf（宇树原版 33.341kg）
        self.scene.robot.spawn.asset_path = _stage_robot_urdf()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


@configclass
class JumpWithIdPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_mimic_jumpwithid"
    run_name = "identified"
