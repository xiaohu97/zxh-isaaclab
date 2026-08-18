#!/usr/bin/env python3
"""Export compact, portable analysis data from Isaac Lab mimic runs.

The raw TensorBoard event files for a handful of long runs can be hundreds of
megabytes.  This tool keeps everything needed for training-curve comparison:

* configuration fingerprints and archived run configurations;
* tail-window scalar summaries and termination shares;
* fixed-size window statistics for every non-wall-clock scalar;
* checkpoint inventory and hashes for the newest checkpoint in each run.

The resulting directory can be analysed on a computer that does not have
Isaac Lab or TensorBoard installed.  Only a CSV reader is needed for the curve
file; ``summary.json`` and ``REPORT.md`` cover the usual comparison questions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from compare_mimic_runs import ERROR_TAGS, _cfg_fingerprint, _episode_length_ceiling


DEFAULT_SUMMARY_WINDOW = 500
DEFAULT_CURVE_WINDOW = 100


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_tail(events, tag: str, window: int) -> float | None:
    if tag not in events.Tags()["scalars"]:
        return None
    values = events.Scalars(tag)
    if not values:
        return None
    return float(np.mean([item.value for item in values[-window:]]))


def _iteration_bounds(events) -> tuple[int, int]:
    for tag in ("Train/mean_reward", "Loss/value_function"):
        if tag in events.Tags()["scalars"]:
            values = events.Scalars(tag)
            if values:
                return int(values[0].step), int(values[-1].step)
    return 0, 0


def _checkpoint_iteration(path: Path) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _copy_run_configs(run: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    candidates = [
        run / "params" / "agent.yaml",
        run / "params" / "env.yaml",
        run / "params" / "tracking_env_cfg.py",
        run / "git" / "unitree_rl_lab.diff",
    ]
    for source in candidates:
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(target.name)
    return copied


def _write_report(summaries: list[dict[str, Any]], output: Path) -> None:
    def number(value: float | None, digits: int = 4) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    error_labels = {
        "Metrics/motion/error_body_pos": "body_pos",
        "Metrics/motion/error_body_rot": "body_rot",
        "Metrics/motion/error_joint_pos": "joint_pos",
        "Metrics/motion/error_anchor_pos": "anchor_pos",
        "Metrics/motion/error_anchor_rot": "anchor_rot",
        "Metrics/motion/error_joint_vel": "joint_vel",
    }
    lines = [
        "# Houtaitui training comparison",
        "",
        f"Snapshot generated: `{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}`",
        "",
        "Values are means over the final configured summary window. Physical error tags are",
        "per-step quantities and remain the safest metrics across reward/config changes.",
        "",
        "## Directly comparable physical errors",
        "",
        "| run | final iter | body pos | body rot | joint pos | anchor pos | anchor rot | joint vel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        errors = item["errors"]
        cells = [number(errors.get(tag)) for tag in error_labels]
        lines.append(
            f"| {item['run']} | {item['final_iteration']} | " + " | ".join(cells) + " |"
        )

    termination_names = sorted(
        {name for item in summaries for name in item["termination_shares"]}
    )
    lines.extend(
        [
            "",
            "## Termination shares",
            "",
            "| run | " + " | ".join(termination_names) + " |",
            "|---|" + "---:|" * len(termination_names),
        ]
    )
    for item in summaries:
        shares = item["termination_shares"]
        cells = [
            "-" if name not in shares else f"{100.0 * shares[name]:.1f}%"
            for name in termination_names
        ]
        lines.append(f"| {item['run']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Episode-normalized view",
            "",
            "| run | mean length | estimated ceiling | length/ceiling | mean reward | reward/step |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            f"| {item['run']} | {number(item['mean_episode_length'], 1)} | "
            f"{number(item['episode_length_ceiling'], 1)} | "
            f"{number(item['episode_length_fraction'], 3)} | "
            f"{number(item['mean_reward'], 2)} | {number(item['reward_per_step'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`: machine-readable summaries, fingerprints and checkpoint inventories.",
            "- `curves_windows.csv.gz`: mean/min/max/last for every scalar in fixed iteration windows.",
            "- `configs/<run>/`: immutable configuration snapshots archived by each run.",
            "- `motions/`: reference NPZ files available when this snapshot was created.",
            "",
            "Do not compare raw episode reward or episode length across runs until checking the",
            "motion file, end behavior, reset sampling and termination set in `summary.json`.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="TensorBoard run directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output bundle directory.")
    parser.add_argument(
        "--summary-window", type=int, default=DEFAULT_SUMMARY_WINDOW,
        help="Number of final scalar points used by summary.json and REPORT.md.",
    )
    parser.add_argument(
        "--curve-window", type=int, default=DEFAULT_CURVE_WINDOW,
        help="Number of scalar points represented by each row in curves_windows.csv.gz.",
    )
    parser.add_argument(
        "--motion", action="append", type=Path, default=[],
        help="Reference NPZ to include (repeatable).",
    )
    args = parser.parse_args()
    if args.summary_window <= 0 or args.curve_window <= 0:
        parser.error("window sizes must be positive")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs_dir = output / "configs"
    motions_dir = output / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)

    motion_sources: dict[str, Path] = {}
    for motion_arg in args.motion:
        motion_source = motion_arg.resolve()
        if not motion_source.is_file():
            raise FileNotFoundError(f"motion file not found: {motion_source}")
        motion_sources[motion_source.name] = motion_source

    summaries: list[dict[str, Any]] = []
    curve_path = output / "curves_windows.csv.gz"
    with gzip.open(curve_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["run", "experiment", "tag", "start_step", "end_step", "count", "mean", "min", "max", "last"]
        )

        for run_arg in args.runs:
            run = run_arg.resolve()
            event_files = sorted(run.glob("events.out.tfevents.*"))
            if not event_files:
                raise FileNotFoundError(f"no TensorBoard event file in {run}")
            events = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
            events.Reload()
            tags = sorted(tag for tag in events.Tags()["scalars"] if not tag.endswith("/time"))
            first_iteration, final_iteration = _iteration_bounds(events)
            fingerprint = _cfg_fingerprint(run)
            ceiling, ceiling_basis = _episode_length_ceiling(fingerprint)
            # A run snapshot can point to a motion file that was later renamed or
            # removed locally.  Rebind it to an explicitly supplied copy with the
            # same basename so the portable report still has a valid ceiling.
            if ceiling is None:
                fallback_motion = motion_sources.get(fingerprint.get("motion_file", ""))
                if fallback_motion is not None:
                    portable_fingerprint = dict(fingerprint)
                    portable_fingerprint["motion_path"] = str(fallback_motion)
                    ceiling, ceiling_basis = _episode_length_ceiling(portable_fingerprint)

            tail_means = {
                tag: _mean_tail(events, tag, args.summary_window)
                for tag in tags
            }
            termination_values = {
                tag.split("/", 1)[1]: value
                for tag, value in tail_means.items()
                if tag.startswith("Episode_Termination/") and value is not None
            }
            termination_total = sum(termination_values.values())
            termination_shares = {
                name: value / termination_total
                for name, value in termination_values.items()
                if termination_total > 0.0
            }
            mean_length = tail_means.get("Train/mean_episode_length")
            mean_reward = tail_means.get("Train/mean_reward")

            checkpoints = []
            for checkpoint in sorted(run.glob("model_*.pt"), key=lambda path: _checkpoint_iteration(path) or -1):
                iteration = _checkpoint_iteration(checkpoint)
                if iteration is None:
                    continue
                checkpoints.append(
                    {
                        "name": checkpoint.name,
                        "iteration": iteration,
                        "size_bytes": checkpoint.stat().st_size,
                    }
                )
            if checkpoints:
                newest = run / checkpoints[-1]["name"]
                checkpoints[-1]["sha256"] = _sha256(newest)

            summaries.append(
                {
                    "run": run.name,
                    "experiment": run.parent.name,
                    "source_path": str(run),
                    "event_file": event_files[-1].name,
                    "event_size_bytes": event_files[-1].stat().st_size,
                    "first_iteration": first_iteration,
                    "final_iteration": final_iteration,
                    "summary_window": args.summary_window,
                    "fingerprint": fingerprint,
                    "errors": {tag: tail_means.get(tag) for tag in ERROR_TAGS},
                    "tail_means": tail_means,
                    "termination_shares": termination_shares,
                    "mean_episode_length": mean_length,
                    "episode_length_ceiling": ceiling,
                    "episode_length_ceiling_basis": ceiling_basis,
                    "episode_length_fraction": (
                        mean_length / ceiling if mean_length is not None and ceiling else None
                    ),
                    "mean_reward": mean_reward,
                    "reward_per_step": (
                        mean_reward / mean_length
                        if mean_reward is not None and mean_length not in (None, 0.0)
                        else None
                    ),
                    "checkpoints": checkpoints,
                    "copied_config_files": _copy_run_configs(run, configs_dir / run.name),
                }
            )

            for tag in tags:
                points = events.Scalars(tag)
                for begin in range(0, len(points), args.curve_window):
                    block = points[begin : begin + args.curve_window]
                    values = np.asarray([point.value for point in block], dtype=np.float64)
                    writer.writerow(
                        [
                            run.name,
                            run.parent.name,
                            tag,
                            block[0].step,
                            block[-1].step,
                            len(block),
                            f"{values.mean():.9g}",
                            f"{values.min():.9g}",
                            f"{values.max():.9g}",
                            f"{values[-1]:.9g}",
                        ]
                    )

    motion_inventory = []
    for source in motion_sources.values():
        target = motions_dir / source.name
        if target.exists() and _sha256(target) != _sha256(source):
            target = motions_dir / f"{source.stem}_{_sha256(source)[:8]}{source.suffix}"
        shutil.copy2(source, target)
        with np.load(source) as motion:
            frames = int(motion["joint_pos"].shape[0])
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        motion_inventory.append(
            {
                "name": target.name,
                "source_path": str(source),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "frames": frames,
                "fps": fps,
                "duration_s": frames / fps,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary_window": args.summary_window,
        "curve_window": args.curve_window,
        "runs": summaries,
        "motions": motion_inventory,
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(summaries, output / "REPORT.md")
    print(output)


if __name__ == "__main__":
    main()
