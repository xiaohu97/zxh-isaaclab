import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    robot_cfg = getattr(getattr(env.cfg, "scene", None), "robot", None)
    if (
        robot_cfg is None
        or not hasattr(robot_cfg, "joint_sdk_names")
        or not hasattr(env, "action_manager")
        or not hasattr(env, "observation_manager")
    ):
        print("[INFO] Skipping deployment configuration export for environments without Unitree SDK mappings.")
        return

    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- left-arm trajectory command: stash Fourier data to inject at FULL precision ---
    # The C++ `arm_command` observation reproduces q_ref_rel / dq_ref from a Fourier series;
    # coefficients must NOT go through the 3-sig-fig `format_value` rounding, so we inject
    # them after formatting. Coeffs are exported RELATIVE to the default pose (fold default
    # into the DC term) so the C++ side evaluates q_ref_rel directly with no extra data.
    arm_injections = {}
    try:
        from unitree_rl_lab.tasks.locomotion.mdp.commands.left_arm_command import (
            LeftArmJointTrajectoryCommand,
        )
    except Exception:
        LeftArmJointTrajectoryCommand = None
    if LeftArmJointTrajectoryCommand is not None:
        for obs_name, obs_cfg in zip(obs_names, env.observation_manager._group_obs_term_cfgs["policy"]):
            cmd_name = (obs_cfg.params or {}).get("command_name")
            if cmd_name is None:
                continue
            try:
                cmd = env.command_manager.get_term(cmd_name)
            except Exception:
                continue
            if not isinstance(cmd, LeftArmJointTrajectoryCommand):
                continue
            a = cmd.fa.detach().cpu().numpy().copy()  # (K, n) cos coeffs of q_traj
            b = cmd.fb.detach().cpu().numpy().copy()  # (K, n) sin coeffs
            default = cmd.default_q[0].detach().cpu().numpy()  # (n,)
            a[0] = a[0] - default  # fold default -> coeffs of (q_traj - default) = q_ref_rel
            arm_injections[obs_name] = {
                "n_joints": int(cmd.num_joints),
                "period": float(cmd.period),
                "blend_time_s": float(cmd.blend_time),
                "ref_vel_scale": float(cmd.cfg.ref_vel_scale),
                "toggle": str(getattr(cmd.cfg, "toggle_button", "RB + A.on_pressed")),
                "omega": cmd.omega.detach().cpu().numpy().tolist(),  # (K,)
                "a_rel": a.tolist(),  # (K, n)
                "b_rel": b.tolist(),  # (K, n)
            }

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    # inject full-precision trajectory data AFTER rounding
    for obs_name, extra in arm_injections.items():
        cfg["observations"][obs_name].setdefault("params", {})
        cfg["observations"][obs_name]["params"].update(extra)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
