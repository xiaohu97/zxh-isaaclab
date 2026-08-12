# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Humanoid Ultra left-arm velocity and acceleration tracking.

The evaluator runs one deterministic, 100%-enabled rollout for each supplied
RSL-RL checkpoint in the same simulator instance.  It records the seven arm
joints at the physics rate and evaluates the pure Fourier segment (4--16 s by
default) at the 50 Hz control rate.

Primary acceleration errors use the same 20 ms finite-difference operator for
actual and reference velocity.  A 5 Hz zero-phase low-pass result and raw 5 ms
physics result are also emitted so low-frequency tracking and high-frequency
vibration are not conflated.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="USTC-Humanoid-Ultra-27dof-Stand-LeftArmTrack",
    help="Registered stand-left-arm task.",
)
parser.add_argument(
    "--checkpoint",
    action="append",
    required=True,
    metavar="LABEL=PATH",
    help="Checkpoint to evaluate. Repeat this option to compare checkpoints.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="artifacts/leftarm_tracking_eval",
    help="Directory for per-checkpoint time series, metrics, and comparison CSV.",
)
parser.add_argument("--duration", type=float, default=18.0, help="Rollout duration in seconds.")
parser.add_argument("--eval_start", type=float, default=4.0, help="Pure-track window start in seconds.")
parser.add_argument("--eval_end", type=float, default=16.0, help="Pure-track window end in seconds.")
parser.add_argument(
    "--lowpass_hz",
    type=float,
    default=5.0,
    help="Zero-phase Butterworth cutoff used for low-frequency acceleration and lag metrics.",
)
parser.add_argument(
    "--max_lag_s",
    type=float,
    default=0.5,
    help="Maximum absolute lag searched by cycle-wise circular cross-correlation.",
)
parser.add_argument("--seed", type=int, default=42, help="Deterministic environment seed.")
parser.add_argument(
    "--skip_timeseries",
    action="store_true",
    help="Do not write the physics-rate time-series CSV files.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.duration <= args_cli.eval_end:
    parser.error("--duration must be greater than --eval_end to provide a post-window filter margin")
if args_cli.eval_start < 0.0 or args_cli.eval_end <= args_cli.eval_start:
    parser.error("evaluation bounds must satisfy 0 <= eval_start < eval_end")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below requires the simulator to be launched first."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from scipy.signal import butter, sosfiltfilt, welch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


ARM_JOINT_COUNT = 7
CONTROL_PERIOD_S = 0.02
PHYSICS_BAND_HZ = (20.0, 35.0)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


def _parse_checkpoint_specs(values: list[str]) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            label, path_text = value.split("=", 1)
        else:
            path = Path(value)
            label = f"{path.parent.name}_{path.stem}"
            path_text = value
        label = label.strip()
        path = Path(path_text).expanduser().resolve()
        if not label:
            raise ValueError(f"Empty checkpoint label in {value!r}")
        if label in seen:
            raise ValueError(f"Duplicate checkpoint label: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(label)
        specs.append(CheckpointSpec(label=label, path=path))
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_nominal_rollout(env_cfg, seed: int) -> None:
    """Make every checkpoint see the same nominal initial state and dynamics."""
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 1.0
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.debug_vis = False
    env_cfg.arm_command.rel_enabled_envs = 1.0
    env_cfg.arm_command.randomize_start_phase = False

    # DelayedPDActuator normally samples an independent 0--2 physics-step
    # delay at every reset.  Pin every delayed actuator to zero so checkpoint
    # comparisons are not confounded by different 0/5/10 ms delay draws.
    for actuator_cfg in env_cfg.scene.robot.actuators.values():
        if hasattr(actuator_cfg, "min_delay"):
            actuator_cfg.min_delay = 0
        if hasattr(actuator_cfg, "max_delay"):
            actuator_cfg.max_delay = 0

    # Disable every startup randomizer, including the left-wrist payload.  The
    # reset events stay enabled but are made deterministic so sequential policy
    # evaluations start from exactly the same simulator state.
    for event_name in (
        "physics_material",
        "add_base_mass",
        "randomize_rigid_body_com",
        "scale_link_mass",
        "scale_actuator_gains",
        "scale_joint_parameters",
        "add_left_wrist_payload",
        "push_robot",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {}
        env_cfg.events.reset_base.params["velocity_range"] = {}
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


class PhysicsArmRecorder:
    """Record q/dq/ddq after every InteractiveScene physics update."""

    def __init__(self, base_env):
        self.base_env = base_env
        self._original_update = base_env.scene.update
        self.active = False
        self.q: list[np.ndarray] = []
        self.dq: list[np.ndarray] = []
        self.ddq_sim: list[np.ndarray] = []

        def update_and_record(dt: float):
            self._original_update(dt)
            if self.active:
                self._append_state()

        base_env.scene.update = update_and_record

    def _append_state(self) -> None:
        ids = self.base_env.arm_joint_ids
        robot_data = self.base_env.robot.data
        self.q.append(robot_data.joint_pos[0, ids].detach().cpu().numpy().copy())
        self.dq.append(robot_data.joint_vel[0, ids].detach().cpu().numpy().copy())
        # ArticulationData.update() has already refreshed this finite difference
        # for the current physics step; cloning it here does not advance history.
        self.ddq_sim.append(robot_data.joint_acc[0, ids].detach().cpu().numpy().copy())

    def start(self) -> None:
        self.q = []
        self.dq = []
        self.ddq_sim = []
        self._append_state()  # t = 0 endpoint
        self.active = True

    def stop(self) -> None:
        self.active = False

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.q), np.asarray(self.dq), np.asarray(self.ddq_sim)

    def restore(self) -> None:
        self.active = False
        self.base_env.scene.update = self._original_update


def _smoothstep_kinematics(elapsed: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration <= 0.0:
        return np.ones_like(elapsed), np.zeros_like(elapsed), np.zeros_like(elapsed)
    u = np.clip(elapsed / duration, 0.0, 1.0)
    alpha = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    alpha_dot = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2) / duration
    alpha_ddot = (120.0 * u**3 - 180.0 * u**2 + 60.0 * u) / duration**2
    return alpha, alpha_dot, alpha_ddot


def _evaluate_arm_reference(base_env, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce q_ref/dq_ref and analytically derive ddq_ref at arbitrary times."""
    fa = base_env.arm_fa.detach().cpu().numpy().astype(np.float64)
    fb = base_env.arm_fb.detach().cpu().numpy().astype(np.float64)
    omega = base_env.arm_omega.detach().cpu().numpy().astype(np.float64)
    default_q = base_env.arm_default_q[0].detach().cpu().numpy().astype(np.float64)
    safe_q = base_env.arm_safe_q[0].detach().cpu().numpy().astype(np.float64)
    phase_offset = float(base_env.arm_phase_offset[0].item())
    enabled = float(base_env.arm_enabled[0].item())

    safe_time = float(base_env.arm_safe_time)
    blend_time = float(base_env.arm_blend_time)
    period = float(base_env.arm_period)
    trajectory_elapsed = np.maximum(times - safe_time, 0.0)
    phase = np.remainder(trajectory_elapsed + phase_offset, period)
    angle = phase[:, None] * omega[None, :]
    cos = np.cos(angle)
    sin = np.sin(angle)
    q_traj = cos @ fa + sin @ fb
    dq_traj = (-(sin * omega[None, :])) @ fa + (cos * omega[None, :]) @ fb
    omega_sq = omega**2
    ddq_traj = (-(cos * omega_sq[None, :])) @ fa - (sin * omega_sq[None, :]) @ fb

    q_ref = np.broadcast_to(default_q, q_traj.shape).copy()
    dq_ref = np.zeros_like(q_ref)
    ddq_ref = np.zeros_like(q_ref)

    to_safe = times < safe_time
    alpha, alpha_dot, alpha_ddot = _smoothstep_kinematics(times, safe_time)
    safe_delta = safe_q - default_q
    q_ref[to_safe] = default_q + alpha[to_safe, None] * safe_delta
    dq_ref[to_safe] = alpha_dot[to_safe, None] * safe_delta
    ddq_ref[to_safe] = alpha_ddot[to_safe, None] * safe_delta

    blend_elapsed = times - safe_time
    blend_end = safe_time + blend_time
    to_track = (times >= safe_time) & (times < blend_end)
    beta, beta_dot, beta_ddot = _smoothstep_kinematics(blend_elapsed, blend_time)
    track_delta = q_traj - safe_q
    q_ref[to_track] = safe_q + beta[to_track, None] * track_delta[to_track]
    dq_ref[to_track] = (
        beta_dot[to_track, None] * track_delta[to_track]
        + beta[to_track, None] * dq_traj[to_track]
    )
    ddq_ref[to_track] = (
        beta_ddot[to_track, None] * track_delta[to_track]
        + 2.0 * beta_dot[to_track, None] * dq_traj[to_track]
        + beta[to_track, None] * ddq_traj[to_track]
    )

    if base_env.arm_auto_fade_out:
        fade_start = float(base_env.max_episode_length_s) - safe_time - blend_time
    else:
        fade_start = math.inf
    pure_track = (times >= blend_end) & (times < fade_start)
    q_ref[pure_track] = q_traj[pure_track]
    dq_ref[pure_track] = dq_traj[pure_track]
    ddq_ref[pure_track] = ddq_traj[pure_track]

    if base_env.arm_auto_fade_out:
        fade_elapsed = times - fade_start
        fade_end = fade_start + blend_time
        from_track = (times >= fade_start) & (times < fade_end)
        gamma, gamma_dot, gamma_ddot = _smoothstep_kinematics(fade_elapsed, blend_time)
        q_ref[from_track] = (
            q_traj[from_track]
            + gamma[from_track, None] * (safe_q - q_traj[from_track])
        )
        dq_ref[from_track] = (
            (1.0 - gamma[from_track, None]) * dq_traj[from_track]
            + gamma_dot[from_track, None] * (safe_q - q_traj[from_track])
        )
        ddq_ref[from_track] = (
            (1.0 - gamma[from_track, None]) * ddq_traj[from_track]
            - 2.0 * gamma_dot[from_track, None] * dq_traj[from_track]
            + gamma_ddot[from_track, None] * (safe_q - q_traj[from_track])
        )

        return_elapsed = times - fade_end
        to_default = times >= fade_end
        eta, eta_dot, eta_ddot = _smoothstep_kinematics(return_elapsed, safe_time)
        default_delta = default_q - safe_q
        q_ref[to_default] = safe_q + eta[to_default, None] * default_delta
        dq_ref[to_default] = eta_dot[to_default, None] * default_delta
        ddq_ref[to_default] = eta_ddot[to_default, None] * default_delta

    q_ref = default_q + enabled * (q_ref - default_q)
    dq_ref *= enabled
    ddq_ref *= enabled
    return q_ref, dq_ref, ddq_ref


def _lowpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    if not 0.0 < cutoff_hz < 0.5 * sample_rate_hz:
        raise ValueError(f"low-pass cutoff must lie in (0, {0.5 * sample_rate_hz}) Hz")
    sos = butter(4, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def _circular_lag(
    actual: np.ndarray,
    reference: np.ndarray,
    sample_period: float,
    max_lag_s: float,
) -> tuple[float, float, bool]:
    """Return lag where positive means actual lags the reference."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    actual = actual - np.mean(actual)
    reference = reference - np.mean(reference)
    actual_norm = np.linalg.norm(actual)
    reference_norm = np.linalg.norm(reference)
    if actual_norm < 1.0e-12 or reference_norm < 1.0e-12:
        return math.nan, math.nan, False

    max_lag = min(int(round(max_lag_s / sample_period)), len(actual) // 2)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.asarray(
        [np.dot(actual, np.roll(reference, int(lag))) / (actual_norm * reference_norm) for lag in lags]
    )
    peak_index = int(np.argmax(correlations))
    lag = float(lags[peak_index])
    peak = float(correlations[peak_index])
    if 0 < peak_index < len(correlations) - 1:
        left, center, right = correlations[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1.0e-12:
            lag += float(0.5 * (left - right) / denominator)
    identifiable = peak >= 0.7 and peak_index not in (0, len(correlations) - 1)
    return lag * sample_period, peak, identifiable


def _cycle_lags(
    actual: np.ndarray,
    reference: np.ndarray,
    sample_period: float,
    period_s: float,
    max_lag_s: float,
) -> tuple[float, float, int]:
    samples_per_cycle = int(round(period_s / sample_period))
    lags: list[float] = []
    all_peaks: list[float] = []
    for start in range(0, len(actual), samples_per_cycle):
        end = start + samples_per_cycle
        if end > len(actual):
            break
        lag, peak, identifiable = _circular_lag(
            actual[start:end], reference[start:end], sample_period, max_lag_s
        )
        if math.isfinite(peak):
            all_peaks.append(peak)
        if identifiable:
            lags.append(lag)
    mean_peak = float(np.mean(all_peaks)) if all_peaks else math.nan
    if not lags:
        return math.nan, mean_peak, 0
    return float(np.median(lags)), mean_peak, len(lags)


def _nanmedian_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.nan


def _error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    error = actual - reference
    ref_rms = np.sqrt(np.mean(reference**2, axis=0))
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return {
        "reference_rms": ref_rms,
        "actual_rms": np.sqrt(np.mean(actual**2, axis=0)),
        "rmse": rmse,
        "p95_abs": np.quantile(np.abs(error), 0.95, axis=0),
        "bias": np.mean(error, axis=0),
        "nrmse": np.divide(rmse, ref_rms, out=np.full_like(rmse, np.nan), where=ref_rms > 1.0e-12),
        "error": error,
    }


def _arm_scalar_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = actual - reference
    frame_rms = np.sqrt(np.mean(error**2, axis=1))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "rmse": rmse,
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
        "p95_frame_rms": float(np.quantile(frame_rms, 0.95)),
        "reference_rms": reference_rms,
        "actual_rms": float(np.sqrt(np.mean(actual**2))),
        "nrmse": rmse / reference_rms if reference_rms > 1.0e-12 else math.nan,
        "bias": float(np.mean(error)),
    }


def _physics_band_power(error: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    nperseg = min(1024, len(error))
    frequencies, psd = welch(error, fs=sample_rate_hz, nperseg=nperseg, axis=0)
    mask = (frequencies >= PHYSICS_BAND_HZ[0]) & (frequencies <= PHYSICS_BAND_HZ[1])
    return np.trapz(psd[mask], frequencies[mask], axis=0)


def _analyze_rollout(
    base_env,
    spec: CheckpointSpec,
    q_physics: np.ndarray,
    dq_physics: np.ndarray,
    ddq_sim_physics: np.ndarray,
    survival_time_s: float,
    early_done: bool,
    env_reference_samples: list[tuple[float, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> dict:
    physics_dt = float(base_env.physics_dt)
    control_dt = float(base_env.step_dt)
    decimation = int(round(control_dt / physics_dt))
    physics_times = np.arange(len(q_physics), dtype=np.float64) * physics_dt
    q_ref_physics, dq_ref_physics, ddq_ref_analytic = _evaluate_arm_reference(base_env, physics_times)

    start_index = int(round(args_cli.eval_start / physics_dt))
    end_index = int(round(args_cli.eval_end / physics_dt))
    complete_window = len(physics_times) > end_index
    if not complete_window:
        raise RuntimeError(
            f"{spec.label}: rollout ended at {physics_times[-1]:.3f}s before eval_end={args_cli.eval_end:.3f}s"
        )

    endpoint_indices = np.arange(start_index, end_index + 1, decimation)
    expected_intervals = int(round((args_cli.eval_end - args_cli.eval_start) / control_dt))
    if len(endpoint_indices) != expected_intervals + 1:
        raise RuntimeError("Evaluation window does not align with the control grid")

    velocity_actual = dq_physics[endpoint_indices[:-1]]
    velocity_reference = dq_ref_physics[endpoint_indices[:-1]]
    acceleration_actual = np.diff(dq_physics[endpoint_indices], axis=0) / control_dt
    acceleration_reference = np.diff(dq_ref_physics[endpoint_indices], axis=0) / control_dt

    # Filter the complete control-rate sequence first, then crop to 4--16 s.
    control_indices = np.arange(0, len(physics_times), decimation)
    control_times = physics_times[control_indices]
    dq_actual_control = dq_physics[control_indices]
    dq_ref_control = dq_ref_physics[control_indices]
    dq_actual_filtered = _lowpass(dq_actual_control, 1.0 / control_dt, args_cli.lowpass_hz)
    dq_ref_filtered = _lowpass(dq_ref_control, 1.0 / control_dt, args_cli.lowpass_hz)
    control_start = int(round(args_cli.eval_start / control_dt))
    control_end = int(round(args_cli.eval_end / control_dt))
    velocity_actual_filtered = dq_actual_filtered[control_start:control_end]
    velocity_reference_filtered = dq_ref_filtered[control_start:control_end]
    acceleration_actual_filtered = (
        dq_actual_filtered[control_start + 1 : control_end + 1]
        - dq_actual_filtered[control_start:control_end]
    ) / control_dt
    acceleration_reference_filtered = (
        dq_ref_filtered[control_start + 1 : control_end + 1]
        - dq_ref_filtered[control_start:control_end]
    ) / control_dt

    # Raw 5 ms physics error and a structural-band velocity-error diagnostic.
    physics_window = slice(start_index, end_index)
    acceleration_actual_5ms = np.diff(dq_physics[start_index : end_index + 1], axis=0) / physics_dt
    acceleration_reference_5ms = np.diff(
        dq_ref_physics[start_index : end_index + 1], axis=0
    ) / physics_dt
    velocity_error_physics = dq_physics[physics_window] - dq_ref_physics[physics_window]
    band_power = _physics_band_power(velocity_error_physics, 1.0 / physics_dt)

    velocity = _error_stats(velocity_actual, velocity_reference)
    acceleration = _error_stats(acceleration_actual, acceleration_reference)
    acceleration_filtered = _error_stats(acceleration_actual_filtered, acceleration_reference_filtered)
    acceleration_5ms = _error_stats(acceleration_actual_5ms, acceleration_reference_5ms)
    reference_period_s = float(base_env.arm_period)

    joint_names = list(base_env.cfg.arm_command.joint_names)
    if len(joint_names) != ARM_JOINT_COUNT:
        raise RuntimeError(f"Expected {ARM_JOINT_COUNT} arm joints, got {len(joint_names)}")

    per_joint: list[dict] = []
    for joint_index, joint_name in enumerate(joint_names):
        vel_lag, vel_peak, vel_cycles = _cycle_lags(
            velocity_actual_filtered[:, joint_index],
            velocity_reference_filtered[:, joint_index],
            control_dt,
            reference_period_s,
            args_cli.max_lag_s,
        )
        acc_lag, acc_peak, acc_cycles = _cycle_lags(
            acceleration_actual_filtered[:, joint_index],
            acceleration_reference_filtered[:, joint_index],
            control_dt,
            reference_period_s,
            args_cli.max_lag_s,
        )
        row = {
            "joint": joint_name,
            "velocity_reference_rms_rad_s": float(velocity["reference_rms"][joint_index]),
            "velocity_actual_rms_rad_s": float(velocity["actual_rms"][joint_index]),
            "velocity_rmse_rad_s": float(velocity["rmse"][joint_index]),
            "velocity_p95_abs_rad_s": float(velocity["p95_abs"][joint_index]),
            "velocity_bias_rad_s": float(velocity["bias"][joint_index]),
            "velocity_nrmse": float(velocity["nrmse"][joint_index]),
            "velocity_lag_s": vel_lag,
            "velocity_phase_lag_deg": vel_lag * 360.0 / reference_period_s,
            "velocity_lag_peak_correlation": vel_peak,
            "velocity_identifiable_cycles": vel_cycles,
            "acceleration_reference_rms_rad_s2": float(acceleration["reference_rms"][joint_index]),
            "acceleration_actual_rms_rad_s2": float(acceleration["actual_rms"][joint_index]),
            "acceleration_rmse_rad_s2": float(acceleration["rmse"][joint_index]),
            "acceleration_p95_abs_rad_s2": float(acceleration["p95_abs"][joint_index]),
            "acceleration_bias_rad_s2": float(acceleration["bias"][joint_index]),
            "acceleration_nrmse": float(acceleration["nrmse"][joint_index]),
            "acceleration_lowpass_rmse_rad_s2": float(acceleration_filtered["rmse"][joint_index]),
            "acceleration_lowpass_p95_abs_rad_s2": float(
                acceleration_filtered["p95_abs"][joint_index]
            ),
            "acceleration_lag_s": acc_lag,
            "acceleration_phase_lag_deg": acc_lag * 360.0 / reference_period_s,
            "acceleration_lag_peak_correlation": acc_peak,
            "acceleration_identifiable_cycles": acc_cycles,
            "acceleration_5ms_rmse_rad_s2": float(acceleration_5ms["rmse"][joint_index]),
            "acceleration_5ms_p95_abs_rad_s2": float(acceleration_5ms["p95_abs"][joint_index]),
            "velocity_error_band_power_20_35hz": float(band_power[joint_index]),
        }
        per_joint.append(row)

    env_ref_times = np.asarray([sample[0] for sample in env_reference_samples], dtype=np.float64)
    env_q_ref = np.asarray([sample[1] for sample in env_reference_samples])
    env_dq_ref = np.asarray([sample[2] for sample in env_reference_samples])
    analytic_q_ref, analytic_dq_ref, _ = _evaluate_arm_reference(base_env, env_ref_times)
    reference_validation = {
        "q_ref_max_abs_error_rad": float(np.max(np.abs(env_q_ref - analytic_q_ref))),
        "dq_ref_max_abs_error_rad_s": float(np.max(np.abs(env_dq_ref - analytic_dq_ref))),
    }

    velocity_lags = np.asarray([row["velocity_lag_s"] for row in per_joint], dtype=np.float64)
    acceleration_lags = np.asarray([row["acceleration_lag_s"] for row in per_joint], dtype=np.float64)
    velocity_lag_median = _nanmedian_or_nan(velocity_lags)
    acceleration_lag_median = _nanmedian_or_nan(acceleration_lags)
    summary = {
        "label": spec.label,
        "checkpoint": str(spec.path),
        "checkpoint_sha256": _sha256(spec.path),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "nominal_profile": {
            "num_envs": 1,
            "arm_enabled_fraction": 1.0,
            "phase_offset_s": float(base_env.arm_phase_offset[0].item()),
            "observation_noise": False,
            "startup_randomization": False,
            "left_wrist_payload_kg": 0.0,
            "actuator_delay_steps": 0,
            "base_command": [0.0, 0.0, 0.0],
        },
        "physics_dt_s": physics_dt,
        "control_dt_s": control_dt,
        "rollout_duration_requested_s": args_cli.duration,
        "survival_time_s": survival_time_s,
        "early_done": early_done,
        "complete_eval_window": complete_window,
        "eval_window_s": [args_cli.eval_start, args_cli.eval_end],
        "eval_control_samples": expected_intervals,
        "reference_period_s": reference_period_s,
        "lowpass_hz": args_cli.lowpass_hz,
        "max_lag_s": args_cli.max_lag_s,
        "reference_validation": reference_validation,
        "velocity": _arm_scalar_stats(velocity_actual, velocity_reference),
        "acceleration_control_interval": _arm_scalar_stats(
            acceleration_actual, acceleration_reference
        ),
        "acceleration_lowpass": _arm_scalar_stats(
            acceleration_actual_filtered, acceleration_reference_filtered
        ),
        "acceleration_physics_5ms": _arm_scalar_stats(
            acceleration_actual_5ms, acceleration_reference_5ms
        ),
        "velocity_lag_median_s": velocity_lag_median,
        "velocity_phase_lag_median_deg": velocity_lag_median * 360.0 / reference_period_s,
        "acceleration_lag_median_s": acceleration_lag_median,
        "acceleration_phase_lag_median_deg": acceleration_lag_median
        * 360.0
        / reference_period_s,
        "velocity_error_band_power_20_35hz_mean": float(np.mean(band_power)),
        "per_joint": per_joint,
    }

    metrics_path = output_dir / f"{spec.label}_metrics.csv"
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_joint[0]))
        writer.writeheader()
        writer.writerows(per_joint)

    summary_path = output_dir / f"{spec.label}_summary.json"
    with summary_path.open("w") as stream:
        json.dump(_json_safe(summary), stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    if not args_cli.skip_timeseries:
        timeseries_path = output_dir / f"{spec.label}_timeseries.csv"
        ddq_actual_fd_5ms = np.full_like(dq_physics, np.nan, dtype=np.float64)
        ddq_reference_fd_5ms = np.full_like(dq_ref_physics, np.nan, dtype=np.float64)
        ddq_actual_fd_5ms[1:] = np.diff(dq_physics, axis=0) / physics_dt
        ddq_reference_fd_5ms[1:] = np.diff(dq_ref_physics, axis=0) / physics_dt
        with timeseries_path.open("w", newline="") as stream:
            fieldnames = ["time_s", "ddq_interval_center_time_s"]
            for joint_name in joint_names:
                fieldnames.extend(
                    [
                        f"{joint_name}.q_actual_rad",
                        f"{joint_name}.q_ref_rad",
                        f"{joint_name}.dq_actual_rad_s",
                        f"{joint_name}.dq_ref_rad_s",
                        f"{joint_name}.ddq_actual_sim_rad_s2",
                        f"{joint_name}.ddq_ref_analytic_rad_s2",
                        f"{joint_name}.ddq_actual_fd_5ms_rad_s2",
                        f"{joint_name}.ddq_ref_fd_5ms_rad_s2",
                    ]
                )
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for sample_index, time_s in enumerate(physics_times):
                row = {
                    "time_s": float(time_s),
                    "ddq_interval_center_time_s": (
                        float(time_s - 0.5 * physics_dt) if sample_index else math.nan
                    ),
                }
                for joint_index, joint_name in enumerate(joint_names):
                    row[f"{joint_name}.q_actual_rad"] = float(q_physics[sample_index, joint_index])
                    row[f"{joint_name}.q_ref_rad"] = float(q_ref_physics[sample_index, joint_index])
                    row[f"{joint_name}.dq_actual_rad_s"] = float(dq_physics[sample_index, joint_index])
                    row[f"{joint_name}.dq_ref_rad_s"] = float(dq_ref_physics[sample_index, joint_index])
                    row[f"{joint_name}.ddq_actual_sim_rad_s2"] = float(
                        ddq_sim_physics[sample_index, joint_index]
                    )
                    row[f"{joint_name}.ddq_ref_analytic_rad_s2"] = float(
                        ddq_ref_analytic[sample_index, joint_index]
                    )
                    row[f"{joint_name}.ddq_actual_fd_5ms_rad_s2"] = float(
                        ddq_actual_fd_5ms[sample_index, joint_index]
                    )
                    row[f"{joint_name}.ddq_ref_fd_5ms_rad_s2"] = float(
                        ddq_reference_fd_5ms[sample_index, joint_index]
                    )
                writer.writerow(row)

    return summary


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_comparison(summaries: list[dict], output_dir: Path) -> Path:
    rows = []
    for summary in summaries:
        velocity = summary["velocity"]
        acceleration = summary["acceleration_control_interval"]
        acceleration_lowpass = summary["acceleration_lowpass"]
        acceleration_5ms = summary["acceleration_physics_5ms"]
        rows.append(
            {
                "label": summary["label"],
                "checkpoint": summary["checkpoint"],
                "survival_time_s": summary["survival_time_s"],
                "complete_eval_window": summary["complete_eval_window"],
                "velocity_rmse_rad_s": velocity["rmse"],
                "velocity_p95_abs_rad_s": velocity["p95_abs"],
                "velocity_nrmse": velocity["nrmse"],
                "velocity_lag_median_s": summary["velocity_lag_median_s"],
                "velocity_phase_lag_median_deg": summary["velocity_phase_lag_median_deg"],
                "acceleration_rmse_rad_s2": acceleration["rmse"],
                "acceleration_p95_abs_rad_s2": acceleration["p95_abs"],
                "acceleration_nrmse": acceleration["nrmse"],
                "acceleration_lowpass_rmse_rad_s2": acceleration_lowpass["rmse"],
                "acceleration_lowpass_p95_abs_rad_s2": acceleration_lowpass["p95_abs"],
                "acceleration_lag_median_s": summary["acceleration_lag_median_s"],
                "acceleration_phase_lag_median_deg": summary["acceleration_phase_lag_median_deg"],
                "acceleration_5ms_rmse_rad_s2": acceleration_5ms["rmse"],
                "velocity_error_band_power_20_35hz_mean": summary[
                    "velocity_error_band_power_20_35hz_mean"
                ],
            }
        )
    comparison_path = output_dir / "comparison.csv"
    with comparison_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path


def main() -> None:
    specs = _parse_checkpoint_specs(args_cli.checkpoint)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args_cli.device or "cuda:0"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    _configure_nominal_rollout(env_cfg, args_cli.seed)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device

    env = None
    recorder = None
    try:
        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        base_env._ensure_arm()
        if abs(float(base_env.step_dt) - CONTROL_PERIOD_S) > 1.0e-9:
            raise RuntimeError(f"Expected {CONTROL_PERIOD_S}s control period, got {base_env.step_dt}")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        recorder = PhysicsArmRecorder(base_env)
        summaries: list[dict] = []

        for spec in specs:
            print(f"[INFO] Evaluating {spec.label}: {spec.path}")
            runner.load(str(spec.path), load_optimizer=False)
            policy = runner.get_inference_policy(device=base_env.device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic

            obs, _ = env.reset()
            base_env._ensure_arm()
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device))

            env_reference_samples: list[tuple[float, np.ndarray, np.ndarray]] = []
            recorder.start()
            early_done = False
            survival_time_s = 0.0
            num_control_steps = int(round(args_cli.duration / float(base_env.step_dt)))
            try:
                with torch.inference_mode():
                    for control_step in range(num_control_steps):
                        time_s = control_step * float(base_env.step_dt)
                        base_env._refresh_arm_ref()
                        env_reference_samples.append(
                            (
                                time_s,
                                (
                                    base_env.arm_default_q[0] + base_env.arm_q_ref_rel[0]
                                ).detach().cpu().numpy().copy(),
                                base_env.arm_dq_ref[0].detach().cpu().numpy().copy(),
                            )
                        )
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        survival_time_s = (control_step + 1) * float(base_env.step_dt)
                        if int(dones[0].item()) != 0:
                            early_done = survival_time_s + 1.0e-9 < args_cli.duration
                            break
            finally:
                recorder.stop()

            q_physics, dq_physics, ddq_sim_physics = recorder.arrays()
            summary = _analyze_rollout(
                base_env,
                spec,
                q_physics,
                dq_physics,
                ddq_sim_physics,
                survival_time_s,
                early_done,
                env_reference_samples,
                output_dir,
            )
            summaries.append(summary)
            print(
                "[RESULT] "
                f"{spec.label}: dq_rmse={summary['velocity']['rmse']:.6f} rad/s, "
                f"ddq_rmse={summary['acceleration_control_interval']['rmse']:.6f} rad/s^2, "
                f"ddq_lp={summary['acceleration_lowpass']['rmse']:.6f} rad/s^2, "
                f"dq_lag={summary['velocity_lag_median_s']:.6f}s, "
                f"ddq_lag={summary['acceleration_lag_median_s']:.6f}s"
            )

        comparison_path = _write_comparison(summaries, output_dir)
        print(f"[INFO] Wrote comparison: {comparison_path}")
    finally:
        if recorder is not None:
            recorder.restore()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
