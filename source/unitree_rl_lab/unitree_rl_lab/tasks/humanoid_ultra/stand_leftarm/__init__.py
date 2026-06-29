"""Register the Humanoid Ultra 27-DoF stand + left-arm trajectory tracking task."""

import gymnasium as gym


gym.register(
    id="USTC-Humanoid-Ultra-27dof-Stand-LeftArmTrack",
    entry_point=f"{__name__}.stand_leftarm_env:HumanoidUltra27dofStandLeftArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stand_leftarm_env_cfg:HumanoidUltra27dofStandLeftArmTrainEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.stand_leftarm_env_cfg:HumanoidUltra27dofStandLeftArmPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.stand_leftarm_agent_cfg:HumanoidUltra27dofStandLeftArmAgentCfg"
        ),
    },
)
