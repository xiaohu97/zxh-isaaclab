import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-VelocityWithIdYaw",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocitywithid_yaw_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocitywithid_yaw_env_cfg:RobotPlayEnvCfg",
        # BasePPORunnerCfg + clip_actions=10.0（防动作尖峰导致 std NaN，见 rsl_rl_ppo_cfg.py）。
        # experiment_name 仍为空，cli_args 按任务名填成 unitree_g1_29dof_velocitywithidyaw
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:VelocityYawPPORunnerCfg",
    },
)
