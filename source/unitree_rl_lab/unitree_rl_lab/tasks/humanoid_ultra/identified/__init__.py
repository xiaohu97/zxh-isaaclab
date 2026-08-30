"""Register Humanoid Ultra tasks that use the identified robot parameters."""

import gymnasium as gym


gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Flat",
    entry_point="unitree_rl_lab.tasks.humanoid_ultra.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:HumanoidUltra27dofIdentifiedFlatEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:HumanoidUltra27dofIdentifiedFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedFlatAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Flat-newpd",
    entry_point="unitree_rl_lab.tasks.humanoid_ultra.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatNewPDEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatNewPDEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedFlatNewPDAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm2kg",
    entry_point="unitree_rl_lab.tasks.humanoid_ultra.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm2kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm2kgEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm2kgAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm4kg",
    entry_point="unitree_rl_lab.tasks.humanoid_ultra.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm4kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm4kgEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm4kgAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm5kg",
    entry_point="unitree_rl_lab.tasks.humanoid_ultra.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm5kgEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm5kgEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedFlatLeftArm5kgAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Identified-Stand",
    entry_point=(
        "unitree_rl_lab.tasks.humanoid_ultra.stand_leftarm.stand_leftarm_env:"
        "HumanoidUltra27dofStandLeftArmEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedStandTrainEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.identified_env_cfg:"
            "HumanoidUltra27dofIdentifiedStandPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.identified_agent_cfg:"
            "HumanoidUltra27dofIdentifiedStandAgentCfg"
        ),
    },
)
