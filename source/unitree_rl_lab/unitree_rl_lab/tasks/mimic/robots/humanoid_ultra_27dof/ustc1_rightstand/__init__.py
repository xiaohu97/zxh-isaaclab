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

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-0808base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaitui0808EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaitui0808PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-ankle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiAnkleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiAnklePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-tightroll",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiTightRollEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiTightRollPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-newpd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotHoutaituiNewPDEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiNewPDPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-rawroll",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiRawBoundedRollEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiRawBoundedRollPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-rawroll-2-5kg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiRawBoundedRollLeftArm2P5kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:RobotHoutaituiRawBoundedRollLeftArm2P5kgPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
