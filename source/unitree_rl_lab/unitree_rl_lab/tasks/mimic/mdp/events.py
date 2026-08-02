from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class phase_targeted_velocity_push(ManagerTermBase):
    """Apply at most one heading-frame root-velocity kick in a motion window.

    A target frame and an enable flag are sampled independently for every
    episode.  The interval event may poll this term every policy step; the kick
    is applied only once when the reference reaches the sampled frame.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        frame_start, frame_end = cfg.params["frame_range"]
        probability = cfg.params["probability"]
        if not 0 <= frame_start <= frame_end:
            raise ValueError(f"Invalid phase-targeted push frame range: {(frame_start, frame_end)}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Phase-targeted push probability must be in [0, 1], got {probability}.")

        self._target_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        frame_start, frame_end = self.cfg.params["frame_range"]
        probability = self.cfg.params["probability"]
        self._target_frames[env_ids] = torch.randint(
            frame_start,
            frame_end + 1,
            (len(env_ids),),
            device=self.device,
        )
        self._pending[env_ids] = torch.rand(len(env_ids), device=self.device) < probability

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        command_name: str,
        frame_range: tuple[int, int],
        probability: float,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        command = env.command_manager.get_term(command_name)
        _, frame_end = frame_range
        motion_frames = command.time_steps[env_ids]
        eligible = (
            self._pending[env_ids]
            & (motion_frames >= self._target_frames[env_ids])
            & (motion_frames <= frame_end)
        )
        if not torch.any(eligible):
            return

        push_env_ids = env_ids[eligible]
        asset: Articulation = env.scene[asset_cfg.name]
        velocity = asset.data.root_vel_w[push_env_ids].clone()
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        local_delta = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], velocity.shape, device=asset.device
        )
        # Interpret the configured direction in the robot heading frame.  This
        # keeps, for example, a rightward kick rightward while the reference is
        # turning instead of accidentally binding it to world -Y.
        anchor_quat_w = command.robot_anchor_quat_w[push_env_ids]
        velocity[:, :3] += math_utils.quat_apply_yaw(anchor_quat_w, local_delta[:, :3])
        velocity[:, 3:] += math_utils.quat_apply_yaw(anchor_quat_w, local_delta[:, 3:])
        asset.write_root_velocity_to_sim(velocity, env_ids=push_env_ids)
        self._pending[push_env_ids] = False


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("JointPositionAction")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
