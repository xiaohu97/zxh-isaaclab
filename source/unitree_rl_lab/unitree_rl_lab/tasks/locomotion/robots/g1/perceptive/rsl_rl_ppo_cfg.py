"""G1 感知任务的 runner 配置。

* ``PerceptiveBlindPPORunnerCfg`` / ``PerceptiveHeightScanPPORunnerCfg``：
  ``VelocityYawPPORunnerCfg``（``BasePPORunnerCfg`` + ``clip_actions=10``）只改实验目录名。
  三个 run 的经验是 1 万迭代后进平台期，起训时用 ``--max_iterations 15000``，不改共享基类。
* ``PerceptiveDepthDistillationRunnerCfg``：rsl_rl 2.3.3 自带的 ``Distillation`` 算法，
  老师 = HeightScan 任务训好的 actor（``teacher_hidden_dims`` 必须和 ``BasePPORunnerCfg``
  的 ``actor_hidden_dims`` 一样，``load_state_dict`` 才能把 ``actor.*`` 权重装进 teacher）。
  实验目录和 HeightScan 共用，``train.py`` 才能用 ``--load_run/--checkpoint`` 找到老师。

  学生目前是把 64×36 深度图展平后接 MLP（rsl_rl 自带的 ``StudentTeacher`` 只有 MLP），
  只够验证流程；真要靠深度图走楼梯得给学生加 CNN 编码器，那需要自定义 policy 类，
  留到雷达通路上机之后。
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.velocitywithid_yaw.rsl_rl_ppo_cfg import VelocityYawPPORunnerCfg

PERCEPTIVE_EXPERIMENT_NAME = "unitree_g1_29dof_perceptive"


@configclass
class PerceptiveBlindPPORunnerCfg(VelocityYawPPORunnerCfg):
    """盲走 rough：观测和线上 walk 完全一样，单独一个实验目录。"""

    experiment_name = PERCEPTIVE_EXPERIMENT_NAME + "_blind"


@configclass
class PerceptiveHeightScanPPORunnerCfg(VelocityYawPPORunnerCfg):
    """高程图进 actor。actor 第一层维度变了，不能直接从 Blind 的 checkpoint ``--resume``
    （要先做补零列的热启动 checkpoint，见 memory / 2026-09-18 的 warmstart run）。

    ``noise_std_type="log"``（2026-09-18）：Blind 12723、HeightScan 3161、yaw 13900 三次崩溃都是
    ``normal expects all elements of std >= 0.0``。scalar 参数化的 std 没有下界，某一维（都是 dim 2）
    一路缩到 0.13，adaptive 学习率冲到 1e-2 时一次迭代 20 步 Adam 就能把它推过零。log 参数化
    std = exp(log_std) 不可能为负。从 scalar 的 checkpoint 续训要把 ``std`` 换成 ``log_std = log(std)``。
    """

    experiment_name = PERCEPTIVE_EXPERIMENT_NAME
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class PerceptiveDepthDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """深度图学生 ← 高程图老师。用法::

        python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PerceptiveDepth --headless \\
            --load_run <HeightScan 的 run 目录名> --checkpoint model_15000.pt --num_envs 1024
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = PERCEPTIVE_EXPERIMENT_NAME
    run_name = "depth_student"
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        student_hidden_dims=[512, 256, 128],
        # configclass 字段是 default_factory，类上取不到，要实例化后再读
        teacher_hidden_dims=list(BasePPORunnerCfg().policy.actor_hidden_dims),
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )
    clip_actions = 10.0
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
