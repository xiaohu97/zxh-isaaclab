"""RSL-RL configurations for identified Humanoid Ultra tasks."""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.humanoid_ultra.base.agents.humanoidultra27dof_agent_cfg import (
    Humanoidultra27dofFlatAgentCfg,
)
from unitree_rl_lab.tasks.humanoid_ultra.stand_leftarm.agents.stand_leftarm_agent_cfg import (
    HumanoidUltra27dofStandLeftArmAgentCfg,
)


@configclass
class HumanoidUltra27dofIdentifiedFlatAgentCfg(Humanoidultra27dofFlatAgentCfg):
    """Keep identified-walk logs separate from nominal Flat checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "humanoidultra27dof_identified_flat"
        self.wandb_project = "humanoidultra27dof_identified_flat"


@configclass
class HumanoidUltra27dofIdentifiedFlatLeftArm2kgAgentCfg(Humanoidultra27dofFlatAgentCfg):
    """Keep left-arm payload Flat runs separate from other identified runs."""

    def __post_init__(self):
        super().__post_init__()
        # A left-arm payload breaks the bilateral dynamics assumed by the
        # Flat task's mirror augmentation and mirror loss.
        self.algorithm.symmetry_cfg = None
        self.experiment_name = "humanoidultra27dof_identified_flat_leftarm2kg"
        self.wandb_project = "humanoidultra27dof_identified_flat_leftarm2kg"


@configclass
class HumanoidUltra27dofIdentifiedStandAgentCfg(
    HumanoidUltra27dofStandLeftArmAgentCfg
):
    """Identified left-arm excitation agent with task-local symmetry disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "humanoidultra27dof_identified_stand_leftarm"
        self.wandb_project = "humanoidultra27dof_identified_stand_leftarm"
