from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class USTCActuator(DelayedPDActuator):
    """Delayed PD actuator with an USTC motor torque-speed envelope.

    ``Y1`` is used while torque and velocity have the same direction
    (motoring), and ``Y2`` is used while they have opposite directions
    (braking). The selected torque limit is constant up to ``X1`` and then
    decreases linearly to zero at the no-load speed ``X2``.

    Motor-specific parameter sets are defined below so they can be replaced
    directly when new test-bench results are available.
    """
    """USTCActuatorCfg class that implements a torque-speed curve for the actuators.
    
        The torque-speed curve is defined as follows:
    
                Torque Limit, N·m
                    ^
        Y2──────────|
                    |──────────────Y1
                    |              │\
                    |              │ \
                    |              │  \
                    |              |   \
        ------------+--------------|------> velocity: rad/s
                                  X1   X2
    
        - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
        - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
        - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
        - X2: No-Load Speed Test
    
        - Fs: Static friction coefficient
        - Fd: Dynamic friction coefficient
        - Va: Velocity at which the friction is fully activated
        """

    cfg: USTCActuatorCfg

    def __init__(self, cfg: USTCActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, None)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, self._effort_y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, None)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, None)

        if torch.any(self._velocity_x1 < 0.0).item():
            raise ValueError("USTCActuator requires X1 >= 0 rad/s.")
        if torch.any(self._velocity_x2 <= self._velocity_x1).item():
            raise ValueError("USTCActuator requires X2 > X1.")
        if torch.any(self._effort_y1 <= 0.0).item() or torch.any(self._effort_y2 <= 0.0).item():
            raise ValueError("USTCActuator requires Y1 > 0 Nm and Y2 > 0 Nm.")

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # The delay applies to the command, while the motor envelope always uses
        # the current physical joint velocity.
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        motoring = (self._joint_vel * effort) >= 0.0
        zero_speed_limit = torch.where(motoring, self._effort_y1, self._effort_y2)
        speed_scale = (
            (self._velocity_x2 - self._joint_vel.abs()) / (self._velocity_x2 - self._velocity_x1)
        ).clamp(min=0.0, max=1.0)
        effort_limit = zero_speed_limit * speed_scale
        return torch.clamp(effort, min=-effort_limit, max=effort_limit)


@configclass
class USTCActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for an :class:`USTCActuator`."""

    class_type: type = USTCActuator

    X1: float | dict[str, float] = MISSING
    """Maximum output speed at the full configured torque, in rad/s."""

    X2: float | dict[str, float] = MISSING
    """No-load output speed, in rad/s."""

    Y1: float | dict[str, float] = MISSING
    """Motoring torque limit, in N*m."""

    Y2: float | dict[str, float] | None = None
    """Braking torque limit, in N*m. Defaults to ``Y1``."""


# X1, X2, and Y1 come from the supplied torque-speed knee measurements. Y2
# retains the previous per-joint simulation maximum as the first braking limit.


@configclass
class USTCActuatorCfg_E8112(USTCActuatorCfg):
    X1 = 16.24
    X2 = 16.42
    Y1 = 45.3
    Y2 = 90.0

    effort_limit_sim = 90.0
    velocity_limit_sim = 1.1 * X2


@configclass
class USTCActuatorCfg_E10020_P24(USTCActuatorCfg):
    X1 = 12.33
    X2 = 13.05
    Y1 = 119.8
    Y2 = 300.0

    effort_limit_sim = 300.0
    velocity_limit_sim = 1.1 * X2


@configclass
class USTCActuatorCfg_E13715(USTCActuatorCfg):
    X1 = 10.50
    X2 = 14.18
    Y1 = 170.1
    Y2 = 300.0

    effort_limit_sim = 300.0
    velocity_limit_sim = 1.1 * X2


@configclass
class USTCActuatorCfg_E10020_P12(USTCActuatorCfg):
    X1 = 11.98
    X2 = 15.05
    Y1 = 62.2
    Y2 = 150.0

    effort_limit_sim = 150.0
    velocity_limit_sim = 1.1 * X2


@configclass
class USTCActuatorCfg_E4315_P36(USTCActuatorCfg):
    X1 = 9.34
    X2 = 12.23
    Y1 = 40.4
    Y2 = 60.0

    effort_limit_sim = 60.0
    velocity_limit_sim = 1.1 * X2


@configclass
class USTCActuatorCfg_E4315_P36_Ankle(USTCActuatorCfg_E4315_P36):
    """First-pass joint-space approximation for the coupled ankle motors."""

    # This uses the measured E4315 motoring curve, but retains the previous
    # 27 Nm ankle maximum for braking. It still operates on virtual ankle joint
    # velocity rather than the two motor speeds and is not an exact coupled model.
    Y2 = 27.0

    effort_limit_sim = 40.4


@configclass
class USTCActuatorCfg_E4310_P36(USTCActuatorCfg):
    X1 = 6.19
    X2 = 9.12
    Y1 = 25.1
    Y2 = 24.0

    effort_limit_sim = 25.1
    velocity_limit_sim = 1.1 * X2
