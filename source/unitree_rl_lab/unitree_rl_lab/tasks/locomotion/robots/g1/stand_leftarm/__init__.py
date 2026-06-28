import gymnasium as gym

# 站立 + 左臂轨迹跟踪（独立任务，不影响 Stand-v2）
gym.register(
    id="Unitree-G1-29dof-Stand-LeftArmTrack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_leftarm_env_cfg:G1StandLeftArmEnvCfg_TRAIN",
        "play_env_cfg_entry_point": f"{__name__}.stand_leftarm_env_cfg:G1StandLeftArmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.stand_leftarm_env_cfg:G1StandLeftArmPPORunnerCfg",
    },
)
