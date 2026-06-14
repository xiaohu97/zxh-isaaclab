"""Register the Humanoid Ultra 27-DoF stand task."""

import gymnasium as gym


gym.register(
    id="USTC-Humanoid-Ultra-27dof-Stand",
    entry_point=f"{__name__}.stand_env:HumanoidUltra27dofStandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stand_env_cfg:HumanoidUltra27dofStandTrainEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.stand_env_cfg:HumanoidUltra27dofStandPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.stand_agent_cfg:HumanoidUltra27dofStandAgentCfg"
        ),
    },
)
