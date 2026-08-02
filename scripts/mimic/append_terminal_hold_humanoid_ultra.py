#!/usr/bin/env python3
"""Append a static terminal hold to a Humanoid Ultra Mimic NPZ.

This is intended for an already-converted reference whose final frame is the
desired standing pose.  Pose fields repeat the final converted rigid-body
state exactly; newly appended velocity fields are zero.  The original prefix
is not resampled or recomputed, which keeps an existing checkpoint's reference
unchanged before the new hold segment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


POSE_FIELDS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_FIELDS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
FRAME_FIELDS = POSE_FIELDS + VELOCITY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--hold-seconds", type=float, default=2.5)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing output file; the input is never overwritten.",
    )
    args = parser.parse_args()
    if args.hold_seconds <= 0.0:
        parser.error("--hold-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_path = args.input_npz.resolve()
    output_path = args.output_npz.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Input and output NPZ paths must differ")
    if output_path.suffix != ".npz":
        raise ValueError("--output-npz must end in .npz")
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(f"Output already exists: {output_path}")

    with np.load(input_path, allow_pickle=False) as source:
        missing = set(("fps",) + FRAME_FIELDS).difference(source.files)
        if missing:
            raise ValueError(f"Input NPZ is missing fields: {sorted(missing)}")
        arrays = {name: np.asarray(source[name]) for name in source.files}

    fps_values = np.asarray(arrays["fps"]).reshape(-1)
    if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0.0:
        raise ValueError(f"Expected one positive fps value, got {arrays['fps']!r}")
    fps = float(fps_values[0])
    hold_frames = int(round(args.hold_seconds * fps))
    if hold_frames < 1:
        raise ValueError("Terminal hold must contain at least one frame interval")

    frame_count = arrays["joint_pos"].shape[0]
    for name in FRAME_FIELDS:
        if arrays[name].ndim < 1 or arrays[name].shape[0] != frame_count:
            raise ValueError(
                f"Expected {name} to have {frame_count} frames, got {arrays[name].shape}"
            )
    for name, array in arrays.items():
        if name not in FRAME_FIELDS and array.ndim > 0 and array.shape[0] == frame_count:
            raise ValueError(f"Unknown frame-aligned field cannot be extended safely: {name}")

    output = dict(arrays)
    for name in POSE_FIELDS:
        output[name] = np.concatenate(
            (arrays[name], np.repeat(arrays[name][-1:], hold_frames, axis=0)), axis=0
        )
    for name in VELOCITY_FIELDS:
        output[name] = np.concatenate(
            (
                arrays[name],
                np.zeros((hold_frames,) + arrays[name].shape[1:], dtype=arrays[name].dtype),
            ),
            axis=0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output)

    print(f"Wrote {output_path}")
    print(
        f"Frames: {frame_count} source + {hold_frames} terminal hold = "
        f"{frame_count + hold_frames} at {fps:g} Hz"
    )
    print(
        "Added hold duration: "
        f"{hold_frames / fps:.3f}s; total sample duration: {(frame_count + hold_frames) / fps:.3f}s"
    )


if __name__ == "__main__":
    main()
