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
    help="Measure the policy root displacement over one nominal mimic motion.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Export the loaded policy as JIT/ONNX and exit without running the simulation loop.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
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

    if args_cli.measure_root_displacement:
        env_cfg.scene.num_envs = 1
        for event_name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    if args_cli.export_only:
        print(f"[INFO] Exported policy to: {export_model_dir}")
        env.close()
        return

    dt = env.unwrapped.step_dt

    measurement = None
    if args_cli.measure_root_displacement:
        command = env.unwrapped.command_manager.get_term("motion")
        robot = command.robot
        command.time_steps.fill_(-1)
        command._update_command()
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
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.forward()
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
        measurement = {
            "command": command,
            "robot": robot,
            "steps": command.motion.time_step_total - 1,
            "start": command.body_pos_w[0, 0].clone(),
            "reference_start": command.motion.body_pos_w[0, 0].clone(),
            "reference_end": command.motion.body_pos_w[-1, 0].clone(),
        }
        print(f"[INFO] Measuring nominal root displacement over {measurement['steps']} policy steps.")

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
            obs, _, dones, _ = env.step(actions)
        if args_cli.video or measurement is not None:
            timestep += 1
        if measurement is not None:
            if dones[0]:
                termination_manager = env.unwrapped.termination_manager
                termination_reasons = [
                    name for name in termination_manager.active_terms if termination_manager.get_term(name)[0]
                ]
                print(
                    f"[ERROR] Measurement rollout terminated early at policy step {timestep}: "
                    f"{termination_reasons}"
                )
                break
            if timestep == measurement["steps"]:
                end = measurement["robot"].data.root_pos_w[0].clone()
                displacement = end - measurement["start"]
                reference_displacement = measurement["reference_end"] - measurement["reference_start"]
                print("[RESULT] Root displacement measurement")
                print(f"  policy_steps: {timestep}")
                print(f"  duration_s: {timestep * dt:.6f}")
                print(f"  actual_start_xyz: {measurement['start'].tolist()}")
                print(f"  actual_end_xyz: {end.tolist()}")
                print(f"  actual_delta_xyz: {displacement.tolist()}")
                print(f"  actual_horizontal_displacement: {torch.linalg.norm(displacement[:2]).item():.6f}")
                print(f"  reference_delta_xyz: {reference_displacement.tolist()}")
                print(
                    "  reference_horizontal_displacement: "
                    f"{torch.linalg.norm(reference_displacement[:2]).item():.6f}"
                )
                break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
