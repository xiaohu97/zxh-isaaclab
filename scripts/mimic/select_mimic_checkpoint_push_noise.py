#!/usr/bin/env python3
"""Rank Mimic checkpoints with reproducible multi-seed push+noise rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TASK = "USTC-Humanoid-Ultra-27dof-Mimic-Taitui-Right-Finetune"
SUMMARY_PATTERNS = {
    "successful_rollouts": re.compile(r"^\s*successful_rollouts:\s*(\d+)\s*$", re.MULTILINE),
    "failed_rollouts": re.compile(r"^\s*failed_rollouts:\s*(\d+)\s*$", re.MULTILINE),
    "full_rollout_success_rate": re.compile(
        r"^\s*full_rollout_success_rate:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "horizontal_displacement_mean": re.compile(
        r"^\s*horizontal_displacement_mean:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Checkpoint .pt files to compare.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--rollouts-per-seed",
        type=int,
        default=100,
        help="Parallel full frame-0 push_noise rollouts for each seed (minimum: 100).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("logs/rsl_rl/checkpoint_selection/taitui_right_finetune_push_noise"),
    )
    parser.add_argument(
        "--wait-for-systemd-unit",
        help="Wait until this user service stops before checking checkpoints (for queued post-training evaluation).",
    )
    parser.add_argument("--wait-timeout-hours", type=float, default=8.0)
    parser.add_argument("--wait-poll-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the measurement matrix only.")
    args = parser.parse_args()

    if len(set(args.seeds)) < 2:
        parser.error("checkpoint selection requires at least two distinct seeds")
    if args.rollouts_per_seed < 100:
        parser.error("--rollouts-per-seed must be at least 100")
    if args.wait_timeout_hours <= 0.0:
        parser.error("--wait-timeout-hours must be positive")
    if args.wait_poll_seconds < 10.0:
        parser.error("--wait-poll-seconds must be at least 10")
    return args


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def checkpoint_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def parse_summary(output: str) -> dict[str, int | float]:
    parsed: dict[str, int | float] = {}
    for name, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(output)
        if match is None:
            raise RuntimeError(f"Measurement output is missing {name!r}.")
        parsed[name] = int(match.group(1)) if name.endswith("rollouts") else float(match.group(1))
    return parsed


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 0.0
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = rate + z * z / (2.0 * trials)
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return (center - margin) / denominator


def wait_for_training_and_checkpoints(args: argparse.Namespace, checkpoints: list[Path]) -> None:
    deadline = time.monotonic() + args.wait_timeout_hours * 3600.0
    if args.wait_for_systemd_unit:
        print(f"[INFO] Waiting for user service {args.wait_for_systemd_unit!r} to stop.", flush=True)
        while True:
            status = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", args.wait_for_systemd_unit],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if status.returncode != 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {args.wait_for_systemd_unit!r}.")
            time.sleep(args.wait_poll_seconds)
        # Give Isaac Sim time to release its CUDA context after the final
        # checkpoint has been flushed and the service becomes inactive.
        time.sleep(15.0)

    missing = [path for path in checkpoints if not path.is_file()]
    if missing and args.wait_for_systemd_unit:
        raise FileNotFoundError(
            "Training stopped without producing all requested checkpoints: "
            + ", ".join(str(path) for path in missing)
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    play_script = repo_root / "scripts" / "rsl_rl" / "play.py"
    checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    if args.wait_for_systemd_unit and not args.dry_run:
        wait_for_training_and_checkpoints(args, checkpoints)
    for checkpoint in checkpoints:
        if not checkpoint.is_file() and not args.dry_run:
            raise FileNotFoundError(checkpoint)
        if checkpoint.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint: {checkpoint}")

    unique_seeds = list(dict.fromkeys(args.seeds))
    matrix_size = len(checkpoints) * len(unique_seeds)
    total_rollouts = matrix_size * args.rollouts_per_seed
    print(
        f"[INFO] Planned matrix: {len(checkpoints)} checkpoints x {len(unique_seeds)} seeds "
        f"x {args.rollouts_per_seed} rollouts = {total_rollouts} full push_noise rollouts."
    )
    if args.dry_run:
        for checkpoint in checkpoints:
            for seed in unique_seeds:
                print(f"  {checkpoint} seed={seed} rollouts={args.rollouts_per_seed}")
        return

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    individual_results: list[dict] = []
    for checkpoint in checkpoints:
        unique_id = checkpoint_id(checkpoint)
        for seed in unique_seeds:
            log_path = results_dir / f"{unique_id}_seed{seed}.log"
            command = [
                sys.executable,
                str(play_script),
                "--task",
                args.task,
                "--checkpoint",
                str(checkpoint),
                "--headless",
                "--device",
                args.device,
                "--measure_root_displacement",
                "--measurement_profile",
                "push_noise",
                "--measurement_rollouts",
                str(args.rollouts_per_seed),
                "--measurement_seed",
                str(seed),
                "--skip_export",
            ]
            print(f"[INFO] Measuring {checkpoint.name}, seed={seed} ...", flush=True)
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Measurement failed for {checkpoint}, seed={seed} (exit {completed.returncode}). "
                    f"See {log_path}."
                )
            summary = parse_summary(completed.stdout)
            reported_total = summary["successful_rollouts"] + summary["failed_rollouts"]
            if reported_total != args.rollouts_per_seed:
                raise RuntimeError(
                    f"Measurement reported {reported_total} rollouts, expected {args.rollouts_per_seed}: {log_path}"
                )
            individual_results.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_id": unique_id,
                    "iteration": checkpoint_iteration(checkpoint),
                    "seed": seed,
                    "rollouts": args.rollouts_per_seed,
                    **summary,
                    "log": str(log_path),
                }
            )

    aggregates: list[dict] = []
    for checkpoint in checkpoints:
        rows = [row for row in individual_results if row["checkpoint"] == str(checkpoint)]
        successes = sum(row["successful_rollouts"] for row in rows)
        trials = sum(row["rollouts"] for row in rows)
        aggregates.append(
            {
                "checkpoint": str(checkpoint),
                "iteration": checkpoint_iteration(checkpoint),
                "seeds": [row["seed"] for row in rows],
                "rollouts_per_seed": args.rollouts_per_seed,
                "successful_rollouts": successes,
                "total_rollouts": trials,
                "aggregate_success_rate": successes / trials,
                "success_rate_wilson_lower_95": wilson_lower_bound(successes, trials),
                "worst_seed_success_rate": min(row["full_rollout_success_rate"] for row in rows),
                "mean_horizontal_displacement": sum(row["horizontal_displacement_mean"] for row in rows)
                / len(rows),
            }
        )

    aggregates.sort(
        key=lambda row: (
            -row["success_rate_wilson_lower_95"],
            -row["worst_seed_success_rate"],
            row["mean_horizontal_displacement"],
            -row["iteration"],
        )
    )
    payload = {
        "task": args.task,
        "profile": "push_noise",
        "ranking_rule": [
            "highest aggregate 95% Wilson lower success bound",
            "highest worst-seed success rate",
            "lowest mean horizontal displacement",
            "newest iteration",
        ],
        "selected_checkpoint": aggregates[0]["checkpoint"],
        "ranking": aggregates,
        "individual_results": individual_results,
    }
    json_path = results_dir / "selection_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = results_dir / "selection_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)

    print("[RESULT] Checkpoint ranking")
    for rank, row in enumerate(aggregates, start=1):
        print(
            f"  {rank}. {Path(row['checkpoint']).name}: success={row['aggregate_success_rate']:.4f}, "
            f"wilson95_lower={row['success_rate_wilson_lower_95']:.4f}, "
            f"worst_seed={row['worst_seed_success_rate']:.4f}, "
            f"horizontal_mean={row['mean_horizontal_displacement']:.4f}"
        )
    print(f"[RESULT] Selected checkpoint: {aggregates[0]['checkpoint']}")
    print(f"[INFO] Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
