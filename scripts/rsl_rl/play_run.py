"""可视化 Unitree-G1-29dof-Run 策略：多档速度并排跑，键盘实时调指令。

用法（GUI）::

    python scripts/rsl_rl/play_run.py --speeds 1.0,2.0,2.5,3.0            # 4 个机器人分别跑 4 档速度
    python scripts/rsl_rl/play_run.py --speeds 2.5 --envs_per_speed 8      # 8 个机器人同跑 2.5 m/s
    python scripts/rsl_rl/play_run.py --load_run 2026-09-07_19-28-31 --checkpoint model_5000.pt

步态参数（步频/支撑相/摆高/躯干高/俯仰）默认按速度插值给一组合理预设，也可用 --freq --stance --swing --body_height --pitch 覆盖。

键盘（窗口有焦点时，对所有机器人同时生效）::

    ↑ / ↓      前向速度 ±0.25 m/s        ← / →   偏航角速度 ±0.25 rad/s
    A / D      侧向速度 ±0.1 m/s          Q / E   支撑相比例 ±0.05（<0.5 有腾空）
    Z / X      步频 ±0.2 Hz               H / N   躯干高 ±0.02 m
    P / L      俯仰 ±0.05 rad             空格    速度归零（看它怎么停、站不站得住）
    K          清掉所有键盘增量            R       重置所有机器人

终端每 2 s 打印各组的指令速度 / 实际速度 / 腾空占比 / 摔倒次数。
"""
import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play the G1 Run policy at chosen speeds.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Run")
parser.add_argument("--speeds", type=str, default="1.0,2.0,2.5,3.0", help="每组的前向速度指令 [m/s]，逗号分隔")
parser.add_argument("--envs_per_speed", type=int, default=1, help="每档速度几个机器人")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--freq", type=float, default=None, help="步频 [Hz]，默认按速度预设")
parser.add_argument("--stance", type=float, default=None, help="支撑相比例，默认按速度预设")
parser.add_argument("--swing", type=float, default=None, help="摆动足高 [m]")
parser.add_argument("--body_height", type=float, default=None, help="躯干高 [m]（--height 是 Isaac Sim 的窗口高度，不能用）")
parser.add_argument("--pitch", type=float, default=None, help="躯干俯仰 [rad]")
parser.add_argument("--push", action="store_true", help="保留训练时的随机推力")
parser.add_argument("--steps", type=int, default=0, help="跑多少步后退出，0 = 一直跑")
parser.add_argument("--real-time", action="store_true", default=False, help="按真实时间播放")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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

# 按速度插值的步态预设（与训练时的可行性约束一致：高速 -> 高步频、低支撑相）
_KNOTS = np.array([0.0, 1.0, 2.0, 2.5, 3.0])
_PRESET = {
    IDX_GAIT_FREQ: np.array([2.0, 1.5, 2.4, 2.8, 3.0]),
    IDX_STANCE_RATIO: np.array([0.60, 0.55, 0.40, 0.35, 0.32]),
    IDX_SWING_HEIGHT: np.array([0.08, 0.12, 0.15, 0.18, 0.20]),
    IDX_BODY_HEIGHT: np.array([0.74, 0.74, 0.72, 0.70, 0.70]),
    IDX_BODY_PITCH: np.array([0.0, 0.0, 0.10, 0.15, 0.20]),
}
_OVERRIDE = {
    IDX_GAIT_FREQ: args_cli.freq,
    IDX_STANCE_RATIO: args_cli.stance,
    IDX_SWING_HEIGHT: args_cli.swing,
    IDX_BODY_HEIGHT: args_cli.body_height,
    IDX_BODY_PITCH: args_cli.pitch,
}


def gait_params(vx: float) -> dict[int, float]:
    out = {}
    for idx, table in _PRESET.items():
        out[idx] = _OVERRIDE[idx] if _OVERRIDE[idx] is not None else float(np.interp(abs(vx), _KNOTS, table))
    return out


def main():
    speeds = [float(s) for s in args_cli.speeds.split(",")]
    per = max(args_cli.envs_per_speed, 1)
    num_envs = len(speeds) * per

    env_cfg = load_cfg_from_registry(args_cli.task, "play_env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = num_envs
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)  # 指令由本脚本控制
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = True
    if not args_cli.push:
        env_cfg.events.push_robot = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("base_velocity")
    robot = uenv.scene["robot"]
    sensor = uenv.scene.sensors["contact_forces"]
    feet, _ = sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
    dt = uenv.step_dt
    device = uenv.device

    # 键盘增量（对所有组生效）
    delta = {"vx": 0.0, "vy": 0.0, "wz": 0.0, IDX_STANCE_RATIO: 0.0, IDX_GAIT_FREQ: 0.0, IDX_BODY_HEIGHT: 0.0, IDX_BODY_PITCH: 0.0}
    state = {"stop": False, "reset": False}

    def apply_targets(snap: bool = False):
        t = cmd.target_b
        for g, vx in enumerate(speeds):
            sl = slice(g * per, (g + 1) * per)
            v = 0.0 if state["stop"] else vx + delta["vx"]
            t[sl, IDX_LIN_VEL_X] = v
            t[sl, IDX_LIN_VEL_Y] = 0.0 if state["stop"] else args_cli.vy + delta["vy"]
            t[sl, IDX_ANG_VEL_Z] = 0.0 if state["stop"] else args_cli.wz + delta["wz"]
            for idx, val in gait_params(v).items():
                t[sl, idx] = val + delta.get(idx, 0.0)
        t[:, IDX_STANCE_RATIO].clamp_(0.25, 1.0)
        t[:, IDX_GAIT_FREQ].clamp_(0.8, 4.0)
        cmd.is_standing_env[:] = False
        cmd.time_left[:] = 1e9
        if snap:
            cmd.command_b[:] = cmd.target_b

    if not args_cli.headless:
        try:
            import carb
            import omni.appwindow

            keymap = {
                "UP": ("vx", 0.25), "DOWN": ("vx", -0.25), "LEFT": ("wz", 0.25), "RIGHT": ("wz", -0.25),
                "A": ("vy", 0.1), "D": ("vy", -0.1), "Q": (IDX_STANCE_RATIO, -0.05), "E": (IDX_STANCE_RATIO, 0.05),
                "Z": (IDX_GAIT_FREQ, -0.2), "X": (IDX_GAIT_FREQ, 0.2), "H": (IDX_BODY_HEIGHT, 0.02), "N": (IDX_BODY_HEIGHT, -0.02),
                "P": (IDX_BODY_PITCH, 0.05), "L": (IDX_BODY_PITCH, -0.05),
            }

            def on_key(event, *_):
                if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                    return True
                name = event.input.name
                if name in keymap:
                    k, dv = keymap[name]
                    delta[k] += dv
                elif name == "SPACE":
                    state["stop"] = not state["stop"]
                elif name == "K":
                    for k in delta:
                        delta[k] = 0.0
                elif name == "R":
                    state["reset"] = True
                else:
                    return True
                shown = {("vx" if k == "vx" else "vy" if k == "vy" else "wz" if k == "wz" else f"idx{k}"): round(v, 2) for k, v in delta.items() if v}
                print(f"[key {name}] stop={state['stop']} 增量={shown}")
                return True

            appwindow = omni.appwindow.get_default_app_window()
            inp = carb.input.acquire_input_interface()
            _sub = inp.subscribe_to_keyboard_events(appwindow.get_keyboard(), on_key)  # noqa: F841
            print(__doc__)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 键盘控制不可用: {e}")

    obs, _ = wrapped.get_observations()
    apply_targets(snap=True)

    falls = torch.zeros(num_envs, device=device)
    win_v = []; win_flight = []; win_footz = []
    t = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            t0 = time.time()
            if state["reset"]:
                uenv.reset(); obs, _ = wrapped.get_observations(); apply_targets(snap=True); falls.zero_(); state["reset"] = False
            act = policy(obs)
            obs, _, dones, _ = wrapped.step(act)
            apply_targets()
            d = dones.bool() & ~uenv.termination_manager.time_outs
            falls += d.float()
            cmd.command_b[d] = cmd.target_b[d]  # 摔倒重置后直接给目标指令，不从随机值滑

            win_v.append(robot.data.root_lin_vel_b[:, 0].clone())
            contact = sensor.data.current_contact_time[:, feet] > 0
            win_flight.append((~contact).all(-1).float())
            win_footz.append(robot.data.body_pos_w[:, feet, 2].max(-1)[0].clone())

            t += 1
            if t % int(2.0 / dt) == 0:
                V = torch.stack(win_v); F = torch.stack(win_flight); Z = torch.stack(win_footz)
                line = []
                for g, vx in enumerate(speeds):
                    sl = slice(g * per, (g + 1) * per)
                    line.append(
                        f"[组{g} 指令vx={cmd.command_b[sl, IDX_LIN_VEL_X].mean():.2f} θ={cmd.command_b[sl, IDX_STANCE_RATIO].mean():.2f} "
                        f"f={cmd.command_b[sl, IDX_GAIT_FREQ].mean():.1f} | 实际 {V[:, sl].mean():.2f} m/s 腾空 {F[:, sl].mean():.0%} "
                        f"足峰 {Z[:, sl].flatten().quantile(0.95):.2f} 摔 {int(falls[sl].sum())}]"
                    )
                print(f"t={t*dt:6.1f}s " + " ".join(line))
                win_v.clear(); win_flight.clear(); win_footz.clear()
            if args_cli.steps and t >= args_cli.steps:
                break
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
