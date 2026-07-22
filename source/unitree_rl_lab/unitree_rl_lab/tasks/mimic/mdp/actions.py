from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class DeploymentLimitedJointPositionAction(JointPositionAction):
    """Joint-position action with deployment target limits and a control-rate slew limit.

    Position clipping is inherited from :class:`JointPositionAction` and is
    applied to the processed position target.  The additional slew limit is
    evaluated once per policy step, matching the deployment controller.
    """

    cfg: DeploymentLimitedJointPositionActionCfg

    def __init__(self, cfg: DeploymentLimitedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if cfg.max_target_velocity <= 0.0:
            raise ValueError("max_target_velocity must be positive")

        self._previous_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._applied_actions = torch.zeros_like(self._previous_target)
        self._needs_target_initialization = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_target_step = float(cfg.max_target_velocity) * float(env.step_dt)

    @property
    def applied_actions(self) -> torch.Tensor:
        """Normalized action corresponding to the position target actually sent to the PD actuator."""
        return self._applied_actions

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        if torch.any(self._needs_target_initialization):
            env_ids = torch.where(self._needs_target_initialization)[0]
            current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
            self._previous_target[env_ids] = current_joint_pos[env_ids]
            self._needs_target_initialization[env_ids] = False

        target_delta = torch.clamp(
            self._processed_actions - self._previous_target,
            min=-self._max_target_step,
            max=self._max_target_step,
        )
        self._processed_actions = self._previous_target + target_delta
        self._previous_target.copy_(self._processed_actions)
        self._applied_actions.copy_((self._processed_actions - self._offset) / self._scale)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._applied_actions[env_ids] = 0.0
        self._needs_target_initialization[env_ids] = True


@configclass
class DeploymentLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for deployment-matched joint-position targets."""

    class_type: type[ActionTerm] = DeploymentLimitedJointPositionAction

    max_target_velocity: float = MISSING
    """Maximum position-target change in rad/s, evaluated at the policy control rate."""


def last_applied_action(env, action_name: str) -> torch.Tensor:
    """Return the normalized action that survived target clipping and slew limiting."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, DeploymentLimitedJointPositionAction):
        raise TypeError(
            f"Action term '{action_name}' must be DeploymentLimitedJointPositionAction, "
            f"got {type(action_term).__name__}."
        )
    return action_term.applied_actions
