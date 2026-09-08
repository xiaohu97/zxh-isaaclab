import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-RunWithId",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.runwithid_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.runwithid_env_cfg:RobotPlayEnvCfg",
        # 直接复用 run 的 PPO 配置：experiment_name 是空字符串，cli_args 会按任务名填成
        # unitree_g1_29dof_runwithid，日志目录自动分开
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.g1.run.rsl_rl_ppo_cfg:RunPPORunnerCfg",
    },
)
