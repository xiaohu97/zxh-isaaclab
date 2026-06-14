"""Humanoid Ultra stand environment with an explicit policy joint-order check."""

from unitree_rl_lab.tasks.humanoid_ultra.base.base_env import BaseEnv

from .joint_order import validate_humanoid_ultra_27dof_joint_order


class HumanoidUltra27dofStandEnv(BaseEnv):
    """Use the existing Humanoid Ultra policy interface and reject bad mappings."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        validate_humanoid_ultra_27dof_joint_order(self.robot.joint_names)
        print("[INFO] Humanoid Ultra 27-DoF stand joint order verified.")
