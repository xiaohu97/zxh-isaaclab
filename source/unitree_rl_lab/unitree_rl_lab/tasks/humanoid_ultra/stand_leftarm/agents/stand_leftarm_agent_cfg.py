"""RSL-RL configuration for Humanoid Ultra 27-DoF stand + left-arm tracking."""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.humanoid_ultra.base.agents.humanoidultra27dof_agent_cfg import (
    Humanoidultra27dofFlatAgentCfg,
)


@configclass
class HumanoidUltra27dofStandLeftArmAgentCfg(Humanoidultra27dofFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        # This task tracks only the left arm, so the base left/right symmetry
        # transform is both dimensionally incompatible with the extra command
        # observations and semantically invalid for this asymmetric objective.
        self.algorithm.symmetry_cfg = None
        self.experiment_name = "humanoidultra27dof_stand_leftarm"
        self.wandb_project = "humanoidultra27dof_stand_leftarm"
        self.max_iterations = 30001
