"""RSL-RL configuration for Humanoid Ultra 27-DoF stand training."""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.humanoid_ultra.base.agents.humanoidultra27dof_agent_cfg import (
    Humanoidultra27dofFlatAgentCfg,
)


@configclass
class HumanoidUltra27dofStandAgentCfg(Humanoidultra27dofFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "humanoidultra27dof_stand"
        self.wandb_project = "humanoidultra27dof_stand"
        self.max_iterations = 30001
