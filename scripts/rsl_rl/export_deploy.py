"""把训练好的策略打包成部署目录，并可选地导出一段参考轨迹供部署侧 C++ 离线对拍。

用法::

    python scripts/rsl_rl/export_deploy.py --task Unitree-G1-29dof-Run \\
        --load_run 2026-09-07_19-28-31 --checkpoint model_49400.pt \\
        --out deploy/robots/g1_29dof/config/policy/run/0908 --reference_steps 40

产物::

    <out>/exported/policy.onnx      纯 actor(normalizer(x))，不含动作裁剪
    <out>/params/deploy.yaml        含 commands.base_velocity.gait 与 actions.*.raw_clip
    <out>/params/source.txt         来源 checkpoint
    <out>/params/reference.json     --reference_steps > 0 时；供 deploy/robots/g1_29dof/test 对拍

与 train.py 启动时那份 deploy.yaml 的区别：这里关掉了执行器增益随机化，导出的 kp/kd 是标称值；
训练启动时的那份是 env 0 抽到的随机增益（±10%），不能直接上机。
"""
import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a trained policy for deployment.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Run")
parser.add_argument("--out", type=str, required=True, help="部署目录，如 deploy/robots/g1_29dof/config/policy/run/0908")
parser.add_argument("--reference_steps", type=int, default=0, help="导出多少步参考轨迹（0 = 不导出）")
parser.add_argument("--reference_vx", type=float, default=1.2, help="参考轨迹的前向速度指令 [m/s]")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.g1.run.gait_command import (
    IDX_ANG_VEL_Z,
    IDX_BODY_HEIGHT,
    IDX_BODY_PITCH,
    IDX_GAIT_FREQ,
    IDX_LIN_VEL_X,
    IDX_LIN_VEL_Y,
    IDX_STANCE_RATIO,
    IDX_SWING_HEIGHT,
)
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

# 参考轨迹用的固定指令：走路区间内、不触发可行性约束，方便看清链路本身
REFERENCE_GAIT = {IDX_GAIT_FREQ: 1.8, IDX_STANCE_RATIO: 0.5, IDX_SWING_HEIGHT: 0.12, IDX_BODY_HEIGHT: 0.74, IDX_BODY_PITCH: 0.0}


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.events.push_robot = None
    env_cfg.events.actuator_gains = None  # 导出标称 kp/kd
    env_cfg.curriculum.gait_cmd_levels = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.base_velocity.randomize_start_phase = False
    # 参考轨迹要和 C++ 逐位对拍：观测不能带噪声
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.reference_steps > 0:
        # 让 reset 时采样到的指令恰好是参考指令（progress=0 -> 用 ranges）；limit_ranges 保持不动，
        # 它决定观测归一化和 deploy.yaml 里的区间
        r = env_cfg.commands.base_velocity.ranges
        r.lin_vel_x = (args_cli.reference_vx, args_cli.reference_vx)
        r.lin_vel_y = (0.0, 0.0)
        r.ang_vel_z = (0.0, 0.0)
        r.gait_freq = (REFERENCE_GAIT[IDX_GAIT_FREQ],) * 2
        r.stance_ratio = (REFERENCE_GAIT[IDX_STANCE_RATIO],) * 2
        r.swing_height = (REFERENCE_GAIT[IDX_SWING_HEIGHT],) * 2
        r.body_height = (REFERENCE_GAIT[IDX_BODY_HEIGHT],) * 2
        r.body_pitch = (REFERENCE_GAIT[IDX_BODY_PITCH],) * 2

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.join(out, "params"), exist_ok=True)
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=os.path.join(out, "exported"), filename="policy.onnx")
    export_deploy_cfg(env.unwrapped, out, clip_actions=agent_cfg.clip_actions)
    with open(os.path.join(out, "params", "source.txt"), "w") as f:
        f.write(f"task: {args_cli.task}\ncheckpoint: {resume_path}\nclip_actions: {agent_cfg.clip_actions}\n")
    print(f"[INFO] exported policy.onnx + deploy.yaml to {out}")

    if args_cli.reference_steps > 0:
        dump_reference(env, wrapped, policy, out, agent_cfg.clip_actions)

    env.close()


def dump_reference(env, wrapped, policy, out, clip_actions):
    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("base_velocity")
    robot = uenv.scene["robot"]

    def snapshot(obs, action):
        return {
            "joint_pos": robot.data.joint_pos[0].tolist(),
            "joint_vel": robot.data.joint_vel[0].tolist(),
            "root_ang_vel_b": robot.data.root_ang_vel_b[0].tolist(),
            "projected_gravity_b": robot.data.projected_gravity_b[0].tolist(),
            "command": cmd.command_b[0].tolist(),
            "phase": float(cmd.phase[0]),
            "obs": obs[0].tolist(),
            "action": action[0].tolist(),
        }

    steps = []
    with torch.inference_mode():
        obs, _ = wrapped.get_observations()
        action = policy(obs)
        steps.append(snapshot(obs, action))
        for _ in range(args_cli.reference_steps):
            obs, _, dones, _ = wrapped.step(action)
            action = policy(obs)
            steps.append(snapshot(obs, action))
            if bool(dones[0]):
                print("[WARN] reference episode terminated early; truncating")
                break

    ref = {
        "task": args_cli.task,
        "joint_names": list(robot.joint_names),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(action.shape[1]),
        "clip_actions": clip_actions,
        "step_dt": float(uenv.step_dt),
        "steps": steps,
    }
    path = os.path.join(out, "params", "reference.json")
    with open(path, "w") as f:
        json.dump(ref, f)
    print(f"[INFO] reference trajectory ({len(steps)} steps) -> {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
