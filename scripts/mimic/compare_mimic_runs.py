#!/usr/bin/env python3
"""Compare mimic training runs whose env configs differ.

Most TensorBoard scalars in this repo are not comparable across config changes:

* ``Episode_Reward/*`` is the episodic sum divided by ``episode_length_s``, so it
  scales with how long episodes actually last.
* ``Episode_Termination/*`` is a raw count of resetting environments, so it
  inflates when episodes get shorter.
* ``Train/mean_episode_length`` has a different ceiling per config: with
  ``motion_end_behavior="resample"`` and no ``motion_end`` term the ceiling is
  ``episode_length_s / step_dt``; with ``"hold"`` plus ``motion_end`` it is the
  expected number of reference frames left after the reset phase is sampled.

``Metrics/motion/error_*`` is the exception: it is a per-step mean in metres and
radians and is directly comparable. This script leads with those, converts the
termination counts to shares, and reports reward per step rather than per
episode.

Usage::

    python scripts/mimic/compare_mimic_runs.py logs/rsl_rl/<exp>/<run> [<run> ...]
    python scripts/mimic/compare_mimic_runs.py --window 500 logs/rsl_rl/*/2026-08-1*
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import yaml

# Directly comparable: per-step physical error, no episode-length normalization.
ERROR_TAGS = [
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_body_rot",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_anchor_rot",
    "Metrics/motion/error_joint_vel",
]


def _last_mean(events, tag: str, window: int) -> float | None:
    try:
        scalars = events.Scalars(tag)
    except KeyError:
        return None
    if not scalars:
        return None
    return float(np.mean([s.value for s in scalars[-window:]]))


def _final_step(events) -> int:
    # "*/time" tags are logged against wall-clock seconds, not iterations.
    for tag in ("Train/mean_reward", "Loss/value_function"):
        try:
            scalars = events.Scalars(tag)
        except KeyError:
            continue
        if scalars:
            return int(scalars[-1].step)
    return 0


def _cfg_fingerprint(run: Path) -> dict:
    env_yaml = run / "params" / "env.yaml"
    if not env_yaml.is_file():
        return {}
    cfg = yaml.unsafe_load(env_yaml.read_text())
    motion = (cfg.get("commands") or {}).get("motion") or {}
    events_cfg = cfg.get("events") or {}
    push = events_cfg.get("push_robot") or {}
    push_vel = (push.get("params") or {}).get("velocity_range") or {}
    armature = ((events_cfg.get("scale_joint_parameters") or {}).get("params") or {}).get(
        "armature_distribution_params"
    )
    actuators = (((cfg.get("scene") or {}).get("robot") or {}).get("actuators")) or {}
    delays = sorted({a.get("max_delay") for a in actuators.values() if isinstance(a, dict)})
    gains = {
        name: (a.get("stiffness"), a.get("damping"))
        for name, a in actuators.items()
        if isinstance(a, dict)
    }
    return {
        "motion_file": os.path.basename(str(motion.get("motion_file", "?"))),
        "end_behavior": motion.get("motion_end_behavior"),
        "frame_zero_p": motion.get("frame_zero_probability"),
        "targeted_p": motion.get("targeted_frame_probability"),
        "targeted_range": motion.get("targeted_frame_range"),
        "reset_vel_xy": ((motion.get("velocity_range") or {}).get("x"),
                         (motion.get("velocity_range") or {}).get("y")),
        "push_interval": push.get("interval_range_s"),
        "push_vel_xy": (push_vel.get("x"), push_vel.get("y")),
        "armature": armature,
        "max_delay": delays,
        "gains": gains,
        "episode_length_s": cfg.get("episode_length_s"),
        "decimation": cfg.get("decimation"),
        "sim_dt": ((cfg.get("sim") or {}).get("dt")),
        "terminations": list(cfg.get("terminations") or {}),
        "rewards": list(cfg.get("rewards") or {}),
        "motion_path": str(motion.get("motion_file", "")),
    }


def _episode_length_ceiling(fp: dict) -> tuple[float | None, str]:
    """Expected mean_episode_length for a policy that never fails."""
    step_dt = (fp.get("decimation") or 4) * (fp.get("sim_dt") or 0.005)
    timeout_steps = (fp.get("episode_length_s") or 0.0) / step_dt
    if "motion_end" not in fp.get("terminations", []):
        return timeout_steps, "timeout"
    path = fp.get("motion_path") or ""
    if not os.path.isfile(path):
        return None, "motion npz not found"
    total = int(np.load(path)["joint_pos"].shape[0])
    p0 = fp.get("frame_zero_p") or 0.0
    pt = fp.get("targeted_p") or 0.0
    rng = fp.get("targeted_range")
    mean_targeted = 0.5 * (rng[0] + rng[1]) if rng else 0.0
    # Remaining adaptive share is approximated as uniform over the clip.
    expected_start = p0 * 0.0 + pt * mean_targeted + (1.0 - p0 - pt) * (total - 1) / 2.0
    return min(total - 1 - expected_start, timeout_steps), f"clip end ({total} frames, approx)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directories (globs allowed).")
    parser.add_argument("--window", type=int, default=500, help="Average over the last N logged iterations.")
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dirs: list[Path] = []
    for pattern in args.runs:
        run_dirs.extend(Path(p) for p in sorted(glob.glob(pattern)))
    run_dirs = [d for d in run_dirs if d.is_dir()]
    if not run_dirs:
        raise SystemExit("no run directories matched")

    rows = []
    for run in run_dirs:
        files = sorted(run.glob("events.out.tfevents.*"))
        if not files:
            print(f"!! {run}: no tfevents file, skipped")
            continue
        events = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
        events.Reload()
        tags = set(events.Tags()["scalars"])
        fp = _cfg_fingerprint(run)

        term_tags = [t for t in tags if t.startswith("Episode_Termination/")]
        term_vals = {t.split("/", 1)[1]: _last_mean(events, t, args.window) or 0.0 for t in term_tags}
        term_total = sum(term_vals.values())

        mean_len = _last_mean(events, "Train/mean_episode_length", args.window)
        mean_rew = _last_mean(events, "Train/mean_reward", args.window)
        ceiling, ceiling_kind = _episode_length_ceiling(fp)

        rows.append({
            "run": run.name,
            "exp": run.parent.name,
            "iters": _final_step(events),
            "fp": fp,
            "errors": {t.rsplit("_", 1)[-1] if False else t: _last_mean(events, t, args.window) for t in ERROR_TAGS},
            "terms": term_vals,
            "term_total": term_total,
            "mean_len": mean_len,
            "mean_rew": mean_rew,
            "per_step_rew": (mean_rew / mean_len) if (mean_rew and mean_len) else None,
            "len_frac": (mean_len / ceiling) if (mean_len and ceiling) else None,
            "ceiling": ceiling,
            "ceiling_kind": ceiling_kind,
        })

    def fmt(v, spec=".4f"):
        return "-" if v is None else format(v, spec)

    w = max(len(r["run"]) for r in rows) + 2
    print(f"\n{'='*100}\nCONFIG FINGERPRINT\n{'='*100}")
    keys = ["motion_file", "end_behavior", "frame_zero_p", "targeted_p", "targeted_range",
            "reset_vel_xy", "push_interval", "push_vel_xy", "armature", "max_delay", "episode_length_s"]
    for k in keys:
        print(f"{k:>16} | " + " | ".join(str(r['fp'].get(k, '?')) for r in rows))
    print(f"{'hip_yaw kp/kd':>16} | " + " | ".join(str(r["fp"].get("gains", {}).get("hip_yaw_E8112", "?")) for r in rows))
    all_terms = sorted({t for r in rows for t in r["terms"]})
    for name in all_terms:
        present = ["Y" if name in r["fp"].get("terminations", []) else "-" for r in rows]
        print(f"{'term:'+name:>16} | " + " | ".join(present))

    print(f"\n{'='*100}\nDIRECTLY COMPARABLE  (per-step physical error, mean of last {args.window} iters)\n{'='*100}")
    print(f"{'run':<{w}}{'iters':>8}" + "".join(f"{t.split('error_')[1]:>16}" for t in ERROR_TAGS))
    for r in rows:
        print(f"{r['run']:<{w}}{r['iters']:>8}" + "".join(f"{fmt(r['errors'][t]):>16}" for t in ERROR_TAGS))

    print(f"\n{'='*100}\nTERMINATION SHARES  (absolute counts are NOT comparable; shares are)\n{'='*100}")
    print(f"{'run':<{w}}" + "".join(f"{t[:14]:>16}" for t in all_terms))
    for r in rows:
        cells = []
        for t in all_terms:
            v = r["terms"].get(t)
            cells.append("-" if v is None or not r["term_total"] else f"{100*v/r['term_total']:.1f}%")
        print(f"{r['run']:<{w}}" + "".join(f"{c:>16}" for c in cells))

    print(f"\n{'='*100}\nEPISODE LENGTH AND REWARD  (normalized; raw values are NOT comparable)\n{'='*100}")
    print(f"{'run':<{w}}{'mean_len':>10}{'ceiling':>10}{'len/ceil':>10}{'mean_rew':>12}{'rew/step':>10}  ceiling basis")
    for r in rows:
        print(f"{r['run']:<{w}}{fmt(r['mean_len'],'.0f'):>10}{fmt(r['ceiling'],'.0f'):>10}"
              f"{fmt(r['len_frac'],'.3f'):>10}{fmt(r['mean_rew'],'.1f'):>12}{fmt(r['per_step_rew'],'.4f'):>10}"
              f"  {r['ceiling_kind']}")
    print()


if __name__ == "__main__":
    main()
