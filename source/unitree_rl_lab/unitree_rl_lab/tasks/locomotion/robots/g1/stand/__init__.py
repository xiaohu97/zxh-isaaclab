import gymnasium as gym

# 使用标准速度命令，兼容现有部署代码
gym.register(
    id="Unitree-G1-29dof-Stand-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg_v2:G1StandEnvCfg_TRAIN",
        "play_env_cfg_entry_point": f"{__name__}.stand_env_cfg_v2:G1StandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
