from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RunPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """跑步任务的 PPO 配置。相对 locomotion 的 BasePPORunnerCfg 只动了两处：

    - ``num_steps_per_env`` 24 -> 32：一个步态周期在 3 Hz 下是 ~17 步，24 步的 rollout
      勉强覆盖一个周期，加长后优势估计能跨越完整的支撑-腾空循环。
    - ``gamma`` 0.99 -> 0.995：0.99 在 50 Hz 下的有效视野只有 1 s，跑步的信用分配
      （蹬地 -> 腾空 -> 落地）跨度更长。
    """

    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # 留空 -> cli_args 自动填成 unitree_g1_29dof_run
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
