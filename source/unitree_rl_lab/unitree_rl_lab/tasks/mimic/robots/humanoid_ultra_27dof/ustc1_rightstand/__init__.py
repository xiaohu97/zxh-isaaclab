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
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaituiEMA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiEmaEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiEmaPlayEnvCfg",
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

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-yaw",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiYawEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiYawPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-yawarm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiYawArmEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiYawArmPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-yawarm-2-5kg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiYawArmLeftArm2P5kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiYawArmLeftArm2P5kgPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:LowEntropyPPORunnerCfg",
    },
)
