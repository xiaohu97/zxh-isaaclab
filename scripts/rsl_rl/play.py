# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--measure_root_displacement",
    action="store_true",
    default=False,
    help="Measure full frame-0 policy rollouts over a mimic motion with an optional stress profile.",
)
parser.add_argument(
    "--measurement_rollouts",
    type=int,
    default=20,
    help="Number of parallel frame-0 rollouts to measure (use 1 for the previous single-rollout behavior).",
)
parser.add_argument(
    "--measurement_profile",
    choices=("nominal", "noise", "push", "push_noise"),
    default="nominal",
    help="Measurement stress profile: nominal, policy observation noise, training-style pushes, or both.",
)
parser.add_argument(
    "--measurement_seed",
    type=int,
    default=42,
    help="Environment seed used to reproduce observation noise and push sampling.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Export the loaded policy as JIT/ONNX and exit without running the simulation loop.",
)
parser.add_argument(
    "--skip_export",
    action="store_true",
    default=False,
    help="Skip JIT/ONNX export (useful for repeated checkpoint measurements).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.measurement_rollouts < 1:
    parser.error("--measurement_rollouts must be at least 1")
if args_cli.export_only and args_cli.skip_export:
    parser.error("--export_only and --skip_export cannot be used together")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


_PUSH_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")
_CONTROLLED_PUSH_INTERVAL_S = 2.0
_CONTROLLED_PUSH_TABLE_LENGTH = 32


def _build_measurement_push_table(num_envs, velocity_range, seed):
    """Pre-sample reproducible velocity kicks without consuming the environment RNG."""
    ranges = torch.tensor(
        [velocity_range.get(component, (0.0, 0.0)) for component in _PUSH_COMPONENTS],
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unit_samples = torch.rand(
        (_CONTROLLED_PUSH_TABLE_LENGTH, num_envs, len(_PUSH_COMPONENTS)),
        generator=generator,
    )
    return (ranges[:, 0] + unit_samples * (ranges[:, 1] - ranges[:, 0])).tolist()


def _apply_measurement_velocity_kick(env, env_ids, velocity_delta_table):
    """Apply and record one controlled training-range root-velocity perturbation."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)

    event_index = getattr(env, "_measurement_push_event_index", 0)
    if event_index >= len(velocity_delta_table):
        raise RuntimeError("Measurement push table was exhausted.")
    all_deltas = torch.tensor(velocity_delta_table[event_index], device=robot.device)
    deltas = all_deltas[env_ids]
    velocity_before = robot.data.root_vel_w[env_ids].clone()
    velocity_after = velocity_before + deltas
    robot.write_root_velocity_to_sim(velocity_after, env_ids=env_ids)

    command = env.command_manager.get_term("motion")
    env._measurement_push_records.append(
        {
            "motion_frame": int(command.time_steps[0].item()),
            "env_ids": env_ids.detach().cpu().tolist(),
            "velocity_delta": deltas.detach().cpu().tolist(),
            "velocity_before": velocity_before.detach().cpu().tolist(),
        }
    )
    env._measurement_push_event_index = event_index + 1


def _reset_measurement_rollout(env, command, policy_nn):
    """Reset all measurement environments to the exact first reference frame."""
    env.reset()
    base_env = env.unwrapped
    base_env._measurement_push_event_index = 0
    base_env._measurement_push_records = []

    command.time_steps.zero_()
    robot = command.robot
    robot.write_joint_state_to_sim(command.joint_pos, command.joint_vel)
    robot.write_root_state_to_sim(
        torch.cat(
            [
                command.body_pos_w[:, 0],
                command.body_quat_w[:, 0],
                command.body_lin_vel_w[:, 0],
                command.body_ang_vel_w[:, 0],
            ],
            dim=-1,
        )
    )

    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.scene.update(dt=base_env.physics_dt)

    # Recompute the relative reference after the exact robot state is in the simulator.
    command.time_steps.fill_(-1)
    command._update_command()

    # Clear history left by env.reset() and seed every history slot from frame 0.
    base_env.observation_manager.reset()
    obs_dict = base_env.observation_manager.compute(update_history=True)
    base_env.obs_buf = obs_dict

    # This is a no-op for the current feed-forward policy and resets recurrent policies.
    policy_nn.reset(torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device))
    return obs_dict["policy"], robot.data.root_pos_w.clone()


def _compute_tracking_errors(base_env, command, motion_frame, ee_body_indexes):
    """Compute the same tracking errors used by the disabled failure terms."""
    env_origins = base_env.scene.env_origins
    ref_anchor_pos = (
        command.motion.body_pos_w[motion_frame, command.motion_anchor_body_index].unsqueeze(0) + env_origins
    )
    anchor_pos_z = torch.abs(ref_anchor_pos[:, 2] - command.robot_anchor_pos_w[:, 2])
    anchor_pos_xy = torch.linalg.norm(ref_anchor_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)

    ref_anchor_quat = command.motion.body_quat_w[
        motion_frame, command.motion_anchor_body_index
    ].unsqueeze(0).expand(base_env.num_envs, -1)
    motion_projected_gravity_b = quat_apply_inverse(ref_anchor_quat, command.robot.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, command.robot.data.GRAVITY_VEC_W)
    anchor_ori = torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2])

    ref_ee_z = command.motion.body_pos_w[motion_frame, ee_body_indexes, 2].unsqueeze(0)
    ref_ee_z = ref_ee_z + env_origins[:, None, 2]
    ee_body_pos_z = torch.abs(ref_ee_z - command.robot_body_pos_w[:, ee_body_indexes, 2])
    return {
        "anchor_pos": anchor_pos_z,
        "anchor_pos_xy": anchor_pos_xy,
        "anchor_ori": anchor_ori,
        "ee_body_pos": ee_body_pos_z,
    }


def _run_root_displacement_measurements(env, policy, policy_nn, dt, limits, settings):
    """Run parallel mimic rollouts from reference frame 0 under one stress profile."""
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("motion")
    robot = command.robot
    num_envs = base_env.num_envs
    # Execute frames 0..N-1.  The final action matches the step on which
    # motion_end would be evaluated during training.
    rollout_steps = command.motion.time_step_total
    diagnostic_frame_range = command.cfg.targeted_frame_range or (87, 130)
    diagnostic_frame_start, diagnostic_frame_end = diagnostic_frame_range
    reference_start = command.motion.body_pos_w[0, 0].clone()
    reference_end = command.motion.body_pos_w[-1, 0].clone()
    reference_displacement = reference_end - reference_start
    ee_body_names = limits["ee_body_pos"]["body_names"]
    ee_body_indexes = [
        index for index, body_name in enumerate(command.cfg.body_names) if body_name in ee_body_names
    ]
    ee_body_names = [command.cfg.body_names[index] for index in ee_body_indexes]
    if not ee_body_indexes:
        raise RuntimeError("Measurement requires at least one configured ee_body_pos body.")

    term_names = ("anchor_pos", "anchor_pos_xy", "anchor_ori")
    first_term_frames = {
        name: torch.full((num_envs,), -1, dtype=torch.long, device=base_env.device) for name in term_names
    }
    first_term_values = {
        name: torch.zeros(num_envs, dtype=torch.float, device=base_env.device) for name in term_names
    }
    first_ee_frames = torch.full(
        (num_envs, len(ee_body_indexes)), -1, dtype=torch.long, device=base_env.device
    )
    first_ee_values = torch.zeros((num_envs, len(ee_body_indexes)), device=base_env.device)
    hard_window_first_frames = torch.full_like(first_ee_frames, -1)
    hard_window_max_errors = torch.zeros_like(first_ee_values)
    hard_window_max_frames = torch.full_like(first_ee_frames, -1)
    ee_error_by_frame = torch.zeros(
        (command.motion.time_step_total, num_envs, len(ee_body_indexes)), device=base_env.device
    )

    print(
        f"[INFO] Measuring {num_envs} parallel frame-0 rollouts over {rollout_steps} policy steps "
        f"with profile={settings['profile']!r}, seed={settings['seed']}."
    )
    print(
        f"[INFO] Observation corruption={'on' if settings['observation_noise'] else 'off'}, "
        f"controlled training-range velocity kicks={'on' if settings['push'] else 'off'}; "
        "startup randomization, failure terminations, and debug visualization are off."
    )
    if settings["push"]:
        print(
            f"[INFO] Controlled push interval_s={settings['push_interval_range_s']}; "
            f"training interval_s={settings['training_push_interval_range_s']}, "
            f"velocity_delta_range={settings['push_velocity_range']}."
        )
    obs, starts = _reset_measurement_rollout(env, command, policy_nn)
    completed_steps = 0
    for rollout_step in range(1, rollout_steps + 1):
        if not simulation_app.is_running():
            break

        start_time = time.time()
        # Terminations would evaluate this frame after physics and before command update.
        motion_frame = int(command.time_steps[0].item())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if torch.any(dones):
            done_env_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Unexpected active termination during measurement for environments: {done_env_ids}")

        errors = _compute_tracking_errors(base_env, command, motion_frame, ee_body_indexes)
        for term_name in term_names:
            crossed = (errors[term_name] > limits[term_name]["threshold"]) & (first_term_frames[term_name] < 0)
            first_term_frames[term_name][crossed] = motion_frame
            first_term_values[term_name][crossed] = errors[term_name][crossed]

        ee_crossed = errors["ee_body_pos"] > limits["ee_body_pos"]["threshold"]
        ee_error_by_frame[motion_frame] = errors["ee_body_pos"]
        first_ee_crossed = ee_crossed & (first_ee_frames < 0)
        first_ee_frames[first_ee_crossed] = motion_frame
        first_ee_values[first_ee_crossed] = errors["ee_body_pos"][first_ee_crossed]

        if diagnostic_frame_start <= motion_frame <= diagnostic_frame_end:
            first_hard_crossed = ee_crossed & (hard_window_first_frames < 0)
            hard_window_first_frames[first_hard_crossed] = motion_frame
            new_max = errors["ee_body_pos"] > hard_window_max_errors
            hard_window_max_errors[new_max] = errors["ee_body_pos"][new_max]
            hard_window_max_frames[new_max] = motion_frame

        completed_steps = rollout_step
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if completed_steps != rollout_steps:
        print(f"[WARNING] Measurement stopped after {completed_steps}/{rollout_steps} policy steps.")
        return

    ends = robot.data.root_pos_w.clone()
    displacements = ends - starts
    horizontal_displacements = torch.linalg.norm(displacements[:, :2], dim=-1)
    ee_term_frames = torch.where(first_ee_frames >= 0, first_ee_frames, rollout_steps + 1).min(dim=1).values
    ee_term_frames[ee_term_frames > rollout_steps] = -1
    all_first_frames = torch.stack([*first_term_frames.values(), ee_term_frames], dim=1)
    valid_first_frames = torch.where(all_first_frames >= 0, all_first_frames, rollout_steps + 1)
    overall_first_frames = valid_first_frames.min(dim=1).values
    successful = overall_first_frames > rollout_steps
    overall_first_frames[successful] = -1

    print("[SUMMARY] Parallel frame-0 rollout measurement")
    print(f"  profile: {settings['profile']}")
    print(f"  seed: {settings['seed']}")
    print(f"  observation_noise: {settings['observation_noise']}")
    print(f"  push: {settings['push']}")
    print(f"  rollouts: {num_envs}")
    print(f"  policy_steps: {rollout_steps}")
    print(f"  duration_s: {rollout_steps * dt:.6f}")
    print(f"  successful_rollouts: {successful.count_nonzero().item()}")
    print(f"  failed_rollouts: {(~successful).count_nonzero().item()}")
    print(f"  full_rollout_success_rate: {successful.float().mean().item():.6f}")
    print(
        "  thresholds: "
        + ", ".join(f"{name}={limits[name]['threshold']:.6f}" for name in (*term_names, "ee_body_pos"))
    )
    print(f"  reference_delta_xyz: {reference_displacement.tolist()}")
    print(f"  horizontal_displacement_mean: {horizontal_displacements.mean().item():.6f}")
    print(f"  horizontal_displacement_min: {horizontal_displacements.min().item():.6f}")
    print(f"  horizontal_displacement_max: {horizontal_displacements.max().item():.6f}")

    if settings["push"]:
        push_records = getattr(base_env, "_measurement_push_records", [])
        print(f"  controlled_push_events: {len(push_records)}")
        print("[DETAIL] Controlled root-velocity perturbations")
        for event_index, record in enumerate(push_records):
            delta = torch.tensor(record["velocity_delta"])
            delta_norm = torch.linalg.norm(delta, dim=1)
            rounded_delta = [[round(value, 6) for value in row] for row in record["velocity_delta"]]
            print(
                f"  event_{event_index:02d}: motion_frame={record['motion_frame']}, "
                f"env_ids={record['env_ids']}, delta_components={list(_PUSH_COMPONENTS)}, "
                f"delta_norm_mean={delta_norm.mean().item():.6f}, "
                f"delta_norm_max={delta_norm.max().item():.6f}, velocity_delta={rounded_delta}"
            )

    print("[DETAIL] Per-environment first threshold crossings")
    for env_id in range(num_envs):
        if successful[env_id]:
            print(f"  env_{env_id:02d}: success")
            continue
        first_frame = int(overall_first_frames[env_id].item())
        first_reasons = [
            name for name in term_names if int(first_term_frames[name][env_id].item()) == first_frame
        ]
        if int(ee_term_frames[env_id].item()) == first_frame:
            first_reasons.append("ee_body_pos")
        term_crossings = {
            name: {
                "frame": int(first_term_frames[name][env_id].item()),
                "error": round(first_term_values[name][env_id].item(), 6),
            }
            for name in term_names
            if first_term_frames[name][env_id] >= 0
        }
        ee_crossings = {
            ee_body_names[index]: {"frame": int(frame), "error_m": round(first_ee_values[env_id, index].item(), 6)}
            for index, frame in enumerate(first_ee_frames[env_id].tolist())
            if frame >= 0
        }
        hard_crossings = {
            ee_body_names[index]: int(frame)
            for index, frame in enumerate(hard_window_first_frames[env_id].tolist())
            if frame >= 0
        }
        ee_errors_at_first_failure = ee_error_by_frame[first_frame, env_id]
        max_ee_index = int(torch.argmax(ee_errors_at_first_failure).item())
        ee_errors_at_first_failure_text = {
            ee_body_names[index]: round(error, 6)
            for index, error in enumerate(ee_errors_at_first_failure.tolist())
        }
        print(
            f"  env_{env_id:02d}: first_failure_frame={first_frame}, reasons={first_reasons}, "
            f"term_crossings={term_crossings}, ee_first_crossings={ee_crossings}, "
            f"ee_z_at_first_failure={ee_errors_at_first_failure_text}, "
            f"max_ee={ee_body_names[max_ee_index]}, "
            f"ee_crossings_{diagnostic_frame_start}_{diagnostic_frame_end}={hard_crossings}"
        )

    print(
        "[DETAIL] End-effector z tracking in motion frames "
        f"{diagnostic_frame_start}-{diagnostic_frame_end}"
    )
    for ee_index, ee_body_name in enumerate(ee_body_names):
        exceeded = hard_window_first_frames[:, ee_index] >= 0
        env_ids = exceeded.nonzero(as_tuple=False).flatten().tolist()
        frames = hard_window_first_frames[exceeded, ee_index].tolist()
        max_error_index = int(torch.argmax(hard_window_max_errors[:, ee_index]).item())
        max_error = hard_window_max_errors[max_error_index, ee_index].item()
        max_error_frame = int(hard_window_max_frames[max_error_index, ee_index].item())
        print(
            f"  {ee_body_name}: exceeded={len(env_ids)}/{num_envs}, env_ids={env_ids}, first_frames={frames}, "
            f"max_error={max_error:.6f} m at frame={max_error_frame} env={max_error_index}"
        )


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    measurement_limits = None
    measurement_settings = None
    if args_cli.measure_root_displacement:
        use_observation_noise = args_cli.measurement_profile in ("noise", "push_noise")
        use_push = args_cli.measurement_profile in ("push", "push_noise")
        env_cfg.scene.num_envs = args_cli.measurement_rollouts
        env_cfg.seed = args_cli.measurement_seed
        env_cfg.observations.policy.enable_corruption = use_observation_noise
        env_cfg.commands.motion.debug_vis = False
        if hasattr(env_cfg.scene, "contact_forces"):
            env_cfg.scene.contact_forces.debug_vis = False

        push_cfg = getattr(env_cfg.events, "push_robot", None)
        if use_push and push_cfg is None:
            raise RuntimeError("The requested measurement profile requires a configured push_robot event.")
        training_push_interval_range_s = tuple(push_cfg.interval_range_s) if use_push else None
        push_velocity_range = dict(push_cfg.params["velocity_range"]) if use_push else None
        measurement_settings = {
            "profile": args_cli.measurement_profile,
            "seed": args_cli.measurement_seed,
            "observation_noise": use_observation_noise,
            "push": use_push,
            "push_interval_range_s": (
                (_CONTROLLED_PUSH_INTERVAL_S, _CONTROLLED_PUSH_INTERVAL_S) if use_push else None
            ),
            "training_push_interval_range_s": training_push_interval_range_s,
            "push_velocity_range": push_velocity_range,
        }

        if use_push:
            push_cfg.func = _apply_measurement_velocity_kick
            push_cfg.interval_range_s = (_CONTROLLED_PUSH_INTERVAL_S, _CONTROLLED_PUSH_INTERVAL_S)
            push_cfg.params = {
                "velocity_delta_table": _build_measurement_push_table(
                    args_cli.measurement_rollouts,
                    push_velocity_range,
                    args_cli.measurement_seed,
                )
            }

        measurement_limits = {}
        for term_name in ("anchor_pos", "anchor_pos_xy", "anchor_ori", "ee_body_pos"):
            term_cfg = getattr(env_cfg.terminations, term_name, None)
            if term_cfg is None or "threshold" not in term_cfg.params:
                raise RuntimeError(f"Measurement requires a configured {term_name!r} threshold termination.")
            measurement_limits[term_name] = {
                "threshold": float(term_cfg.params["threshold"]),
                "body_names": list(term_cfg.params.get("body_names", [])),
            }
            # Measure the same threshold manually so Isaac Lab cannot auto-reset
            # and overwrite the state that caused the first crossing.
            setattr(env_cfg.terminations, term_name, None)
        if hasattr(env_cfg.terminations, "motion_end"):
            env_cfg.terminations.motion_end = None

        # The measurement harness supplies its own deterministic push table.
        # Disable training-only randomizers, including the phase-targeted push,
        # so nominal/noise profiles stay clean and push profiles contain only
        # the recorded controlled kicks.
        for event_name in (
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "targeted_push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        if not use_push and hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.measure_root_displacement:
        # Also disable at runtime so no callback survives until environment teardown.
        env.unwrapped.command_manager.set_debug_vis(False)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # Export unless this process is one entry in a repeated checkpoint
    # measurement matrix.
    if not args_cli.skip_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        if args_cli.export_only:
            print(f"[INFO] Exported policy to: {export_model_dir}")
            env.close()
            return

    dt = env.unwrapped.step_dt

    try:
        if args_cli.measure_root_displacement:
            _run_root_displacement_measurements(
                env, policy, policy_nn, dt, measurement_limits, measurement_settings
            )
            # Isaac Sim may terminate before fully flushing a redirected stdout
            # stream, so persist the final aggregate diagnostics explicitly.
            sys.stdout.flush()
            sys.stderr.flush()
        else:
            # reset environment
            obs = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                obs, _ = env.get_observations()
            timestep = 0
            # simulate environment
            while simulation_app.is_running():
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, _, _ = env.step(actions)
                if args_cli.video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if args_cli.measure_root_displacement:
            env.unwrapped.command_manager.set_debug_vis(False)
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
