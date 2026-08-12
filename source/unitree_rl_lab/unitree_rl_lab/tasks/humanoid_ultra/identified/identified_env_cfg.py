"""Walk and arm-excitation stand configs using identified robot parameters."""

from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.humanoid_ultra import (
    HUMANOIDULTRA27DOF_IDENTIFIED_CFG,
    HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM2KG_CFG,
    HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM4KG_CFG,
    HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM5KG_CFG,
)
from unitree_rl_lab.tasks.humanoid_ultra.base.humanoidultra27dof_env_cfg import (
    Humanoidultra27dofFlatEnvCfg,
)
from unitree_rl_lab.tasks.humanoid_ultra.base.scene_cfg import SceneCfg
from unitree_rl_lab.tasks.humanoid_ultra.stand_leftarm.stand_leftarm_env_cfg import (
    HumanoidUltra27dofStandLeftArmPlayEnvCfg,
    HumanoidUltra27dofStandLeftArmTrainEnvCfg,
)


def _apply_identified_robot(env_cfg, robot_cfg=HUMANOIDULTRA27DOF_IDENTIFIED_CFG) -> None:
    """Replace the nominal asset and rebuild the scene around that asset."""
    env_cfg.scene_context.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene = SceneCfg(
        config=env_cfg.scene_context,
        physics_dt=env_cfg.sim.dt,
        step_dt=env_cfg.decimation * env_cfg.sim.dt,
    )


@configclass
class HumanoidUltra27dofIdentifiedFlatEnvCfg(Humanoidultra27dofFlatEnvCfg):
    """Flat-ground walking with identified URDF inertias and mimic armatures."""

    def __post_init__(self):
        super().__post_init__()
        _apply_identified_robot(self)


@configclass
class HumanoidUltra27dofIdentifiedFlatLeftArm2kgEnvCfg(Humanoidultra27dofFlatEnvCfg):
    """Flat walking with the identified left-arm 2 kg payload rigid-body model."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.debug_vis = True
        _apply_identified_robot(self, HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM2KG_CFG)


@configclass
class HumanoidUltra27dofIdentifiedFlatLeftArm4kgEnvCfg(Humanoidultra27dofFlatEnvCfg):
    """Flat walking with the identified left-arm 4 kg payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.debug_vis = True
        _apply_identified_robot(self, HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM4KG_CFG)


@configclass
class HumanoidUltra27dofIdentifiedFlatLeftArm5kgEnvCfg(Humanoidultra27dofFlatEnvCfg):
    """Flat walking with the identified left-arm 5 kg payload model."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.debug_vis = True
        _apply_identified_robot(self, HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM5KG_CFG)


@configclass
class HumanoidUltra27dofIdentifiedStandTrainEnvCfg(
    HumanoidUltra27dofStandLeftArmTrainEnvCfg
):
    """Left-arm excitation training with identified robot parameters."""

    def __post_init__(self):
        super().__post_init__()
        _apply_identified_robot(self)


@configclass
class HumanoidUltra27dofIdentifiedStandPlayEnvCfg(
    HumanoidUltra27dofStandLeftArmPlayEnvCfg
):
    """Left-arm excitation play config using the same identified robot."""

    def __post_init__(self):
        super().__post_init__()
        _apply_identified_robot(self)
