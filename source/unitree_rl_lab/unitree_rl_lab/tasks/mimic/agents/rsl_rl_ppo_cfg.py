# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""  # same as task name
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LowEntropyPPORunnerCfg(BasePPORunnerCfg):
    """Warm-start fine-tuning config that lets the action noise collapse.

    ``entropy_coef = 0.005`` pays the policy to keep its action noise wide, and
    on the houtaitui line it settled at std 0.57 -- with ACTION_SCALE 0.25 that
    is 0.14 rad of exploration noise per joint per step.  PPO then optimises the
    return of ``N(mu, std)`` while deployment runs ``mu`` alone, and nothing
    constrains where mu sits inside that distribution.

    Measured consequence: across five checkpoints of one run, evaluating mu gave
    fall rates of 4/66/41/57/10 per 100 while evaluating the full distribution
    on the same seeds gave 23/26/48/39/23 -- the spread collapses from a range
    of 62 to 25 (std 24.8 -> 10.0).  So "adjacent checkpoints differ 5x" was
    mostly an artefact of scoring an unconstrained mean, and the smooth
    TensorBoard curves (which are stochastic-policy returns) were never in
    conflict with it.

    Zeroing the entropy bonus removes the pressure keeping std wide, so the
    distribution contracts onto mu as the policy converges and the thing being
    optimised becomes the thing being deployed.  Intended for fine-tuning from
    an already-good checkpoint, where exploration is no longer the priority.
    Watch ``Policy/mean_noise_std``: it should fall well below 0.57.
    """

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
