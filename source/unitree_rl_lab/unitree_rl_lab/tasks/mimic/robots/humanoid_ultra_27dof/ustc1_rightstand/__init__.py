import gymnasium as gym

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-2-5kg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiLeftArm2P5kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiLeftArm2P5kgPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
