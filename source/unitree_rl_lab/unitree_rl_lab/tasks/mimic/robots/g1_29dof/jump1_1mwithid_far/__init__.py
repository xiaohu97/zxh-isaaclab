import gymnasium as gym

# 三级课程共享同一套 cfg（tracking_env_cfg.make_stage），只换参考/must_jump 门槛/run_name。
# 训练顺序: C1(1.00m, 从 recover 热启动) -> C2(1.15m, 从 C1) -> Far(1.30m, 从 C2)。
for _id, _env, _play, _runner in (
    ("Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar-C1", "C1_EnvCfg", "C1_PlayEnvCfg", "C1_RunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar-C1U", "C1U_EnvCfg", "C1U_PlayEnvCfg", "C1U_RunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar-C1F", "C1F_EnvCfg", "C1F_PlayEnvCfg", "C1F_RunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar-C2", "C2_EnvCfg", "C2_PlayEnvCfg", "C2_RunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Jump1-1mWithIdFar", "RobotEnvCfg", "RobotPlayEnvCfg", "Jump1_1mWithIdFarPPORunnerCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:{_env}",
            "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:{_play}",
            "rsl_rl_cfg_entry_point": f"{__name__}.tracking_env_cfg:{_runner}",
        },
    )
