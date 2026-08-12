"""Plant randomization shared by every mimic task.

The mimic ``EventCfg`` inherited from the upstream g1 tasks randomized only the
ground material, the joint default positions, the anchor-body CoM and push
disturbances. The plant itself — link inertias, PD gains, rotor inertia — was
byte-identical in every environment, so a mimic policy could fit one exact set
of actuator and rigid-body parameters. That shows up on hardware as leg
vibration, most visibly in single-support motions where the ankle carries the
whole robot and its rotor inertia dominates the joint-space inertia.

Ranges follow the ``humanoid_ultra`` flat tasks, except that the armature range
is widened to 0.6-1.4x: identification from standing data cannot pin the rotor
inertia down (the estimates collapse back to the prior under a 10x stronger
rotor regularizer), so it has to be treated as genuinely uncertain.

Mix into a task's ``EventCfg`` by inheriting from it::

    @configclass
    class EventCfg(PlantRandomizationEventCfg):
        ...
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp


@configclass
class PlantRandomizationEventCfg:
    """Rigid-body and actuator randomization applied to every mimic task."""

    # startup
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Limb links only; the torso/pelvis payload is covered by each task's own
            # CoM term. recompute_inertia defaults to True, so the inertia tensor is
            # rescaled with the mass under a uniform-density assumption.
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )
