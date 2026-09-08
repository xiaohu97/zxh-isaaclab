import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-RunWithIdTN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.runwithid_tn_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.runwithid_tn_env_cfg:RobotPlayEnvCfg",
        # 复用 run 的 PPO 配置：experiment_name 为空，cli_args 按任务名填成
        # unitree_g1_29dof_runwithidtn，日志目录自动分开
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.g1.run.rsl_rl_ppo_cfg:RunPPORunnerCfg",
    },
)
