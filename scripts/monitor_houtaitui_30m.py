#!/usr/bin/env python3
"""Read-only 30-minute monitor for the two houtaitui training runs.

The script intentionally reads TensorBoard event files only; it never opens or
modifies a simulator/training process.  It writes one snapshot immediately and
then repeats every 30 minutes, which makes the monitor useful even when the
training process itself is attached to another terminal.

IsaacLab's ``Episode_Reward/*`` scalars are already episodic sums divided by
``max_episode_length_s`` (a fixed 30 seconds in this task).  They are therefore
reported as comparable per-second reward rates; they must not be divided again
by the observed episode length.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "houtaitui": Path(
        "logs/rsl_rl/ustc_humanoid_ultra_27dof_mimic_houtaitui/2026-08-14_01-08-04"
    ),
    "houtaituiEMA": Path(
        "logs/rsl_rl/ustc_humanoid_ultra_27dof_mimic_houtaituiema/2026-08-14_03-55-28_impact10"
    ),
}
# ``Episode_Reward/feet_impact_velocity`` is the physical touchdown-speed
# proxy multiplied by the task's reward weight.  Keep this mapping explicit so
# comparisons remain valid if the two task weights diverge in a later run.
IMPACT_REWARD_WEIGHT = {
    "houtaitui": -1.0,
    "houtaituiEMA": -1.0,
}
TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/time_out",
    "Episode_Termination/ee_body_pos",
    "Episode_Reward/single_support_stability",
    "Episode_Reward/feet_impact_velocity",
    "Episode_Reward/action_rate_l2",
    "Policy/mean_noise_std",
)


def moving_mean(values: list[float], count: int = 50) -> float:
    window = values[-min(count, len(values)) :]
    return sum(window) / len(window)


def read_events(path: Path) -> dict[str, list[tuple[int, float]]] | None:
    event_files = list(path.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = accumulator.Tags().get("scalars", [])
    result: dict[str, list[tuple[int, float]]] = {}
    for tag in TAGS:
        if tag not in scalars:
            continue
        events = accumulator.Scalars(tag)
        finite = [(event.step, event.value) for event in events if math.isfinite(event.value)]
        if not finite:
            continue
        result[tag] = finite
    return result


def read_run(path: Path) -> dict[str, tuple[int, float, float]] | None:
    events = read_events(path)
    if not events:
        return None
    return {
        tag: (values[-1][0], values[-1][1], moving_mean([value for _, value in values]))
        for tag, values in events.items()
    }


def read_run_at_step(path: Path, target_step: int) -> dict[str, tuple[int, float, float]] | None:
    """Return 50-step means using only samples at or before ``target_step``."""
    events = read_events(path)
    if not events:
        return None
    result: dict[str, tuple[int, float, float]] = {}
    for tag, values in events.items():
        eligible = [(step, value) for step, value in values if step <= target_step]
        if eligible:
            result[tag] = (
                eligible[-1][0],
                eligible[-1][1],
                moving_mean([value for _, value in eligible]),
            )
    return result


def format_run(name: str, data: dict[str, tuple[int, float, float]]) -> str:
    step = max(item[0] for item in data.values())

    def mean(tag: str) -> float:
        return data.get(tag, (0, math.nan, math.nan))[2]

    length = mean("Train/mean_episode_length")
    support = mean("Episode_Reward/single_support_stability")
    impact_reward = mean("Episode_Reward/feet_impact_velocity")
    impact_proxy = impact_reward / abs(IMPACT_REWARD_WEIGHT[name])
    action_rate = mean("Episode_Reward/action_rate_l2")
    impact_per_step = abs(impact_proxy) / length if math.isfinite(length) and length > 0.0 else math.nan
    action_per_step = abs(action_rate) / length if math.isfinite(length) and length > 0.0 else math.nan
    return (
        f"{name}: step={step} reward_m50={mean('Train/mean_reward'):.5f} "
        f"length_m50={length:.5f} timeout_m50={mean('Episode_Termination/time_out'):.5f} "
        f"ee_term_m50={mean('Episode_Termination/ee_body_pos'):.5f} "
        f"support_reward_m50={support:.5f} "
        f"impact_reward_m50={impact_reward:.5f} impact_proxy_m50={impact_proxy:.5f} "
        f"impact_per_step_m50={impact_per_step:.8f} "
        f"action_rate_m50={action_rate:.5f} action_per_step_m50={action_per_step:.8f}"
    )


def percent_delta(value: float, reference: float) -> float:
    """Relative delta in percent, using magnitude for signed reward rates."""
    denominator = abs(reference)
    return math.nan if denominator < 1.0e-12 else 100.0 * (value - reference) / denominator


def snapshot() -> str:
    lines = [f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] houtaitui monitor"]
    run_data: dict[str, dict[str, tuple[int, float, float]]] = {}
    for name, path in RUNS.items():
        data = read_run(path)
        if not data:
            lines.append(f"{name}: no event data yet")
            continue
        run_data[name] = data
        lines.append(format_run(name, data))

    # The two jobs run at different wall-clock speeds.  Always include a
    # common-step comparison so a faster job is not mistaken for a better one
    # merely because it has trained longer.
    if len(run_data) == len(RUNS):
        common_step = min(max(item[0] for item in data.values()) for data in run_data.values())
        lines.append(f"aligned_at_step={common_step}")
        for name, path in RUNS.items():
            aligned = read_run_at_step(path, common_step)
            if aligned:
                lines.append(format_run(name, aligned))
        normal = read_run_at_step(RUNS["houtaitui"], common_step)
        ema = read_run_at_step(RUNS["houtaituiEMA"], common_step)
        if normal and ema:
            def mean(data: dict[str, tuple[int, float, float]], tag: str) -> float:
                return data.get(tag, (0, math.nan, math.nan))[2]

            normal_length = mean(normal, "Train/mean_episode_length")
            ema_length = mean(ema, "Train/mean_episode_length")
            normal_support = mean(normal, "Episode_Reward/single_support_stability")
            ema_support = mean(ema, "Episode_Reward/single_support_stability")
            normal_impact = mean(normal, "Episode_Reward/feet_impact_velocity")
            ema_impact = mean(ema, "Episode_Reward/feet_impact_velocity")
            normal_impact_proxy = normal_impact / abs(IMPACT_REWARD_WEIGHT["houtaitui"])
            ema_impact_proxy = ema_impact / abs(IMPACT_REWARD_WEIGHT["houtaituiEMA"])
            normal_action = mean(normal, "Episode_Reward/action_rate_l2")
            ema_action = mean(ema, "Episode_Reward/action_rate_l2")
            normal_impact_per_step = abs(normal_impact_proxy) / normal_length if normal_length > 0.0 else math.nan
            ema_impact_per_step = abs(ema_impact_proxy) / ema_length if ema_length > 0.0 else math.nan
            normal_action_per_step = abs(normal_action) / normal_length if normal_length > 0.0 else math.nan
            ema_action_per_step = abs(ema_action) / ema_length if ema_length > 0.0 else math.nan
            normal_ee = mean(normal, "Episode_Termination/ee_body_pos")
            ema_ee = mean(ema, "Episode_Termination/ee_body_pos")
            lines.append(
                "aligned_delta_ema_minus_normal_pct: "
                f"length={percent_delta(ema_length, normal_length):+.2f} "
                f"support_reward={percent_delta(ema_support, normal_support):+.2f} "
                f"impact_proxy_abs={percent_delta(abs(ema_impact_proxy), abs(normal_impact_proxy)):+.2f} "
                f"impact_per_step={percent_delta(ema_impact_per_step, normal_impact_per_step):+.2f} "
                f"action_abs={percent_delta(abs(ema_action), abs(normal_action)):+.2f} "
                f"action_per_step={percent_delta(ema_action_per_step, normal_action_per_step):+.2f} "
                f"ee_term_pp={(ema_ee - normal_ee) * 100.0:+.2f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/houtaitui_monitor_v3.log"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        text = snapshot()
        print(text, flush=True)
        with args.output.open("a", encoding="utf-8") as file:
            file.write(text + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
