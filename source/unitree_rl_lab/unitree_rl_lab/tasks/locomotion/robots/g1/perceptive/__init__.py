import gymnasium as gym

# 三档任务共用场景 / 地形 / 奖励，只有观测组不同，见 perceptive_env_cfg.py 的模块注释。
gym.register(
    id="Unitree-G1-29dof-PerceptiveBlind",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveBlindEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveBlindPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveBlindPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-PerceptiveHeightScan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveHeightScanEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveHeightScanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveHeightScanPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-PerceptiveDepth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveDepthEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:PerceptiveDepthPlayEnvCfg",
        # 蒸馏：--load_run/--checkpoint 指向 HeightScan 的 run（同一实验目录）
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveDepthDistillationRunnerCfg",
    },
)
