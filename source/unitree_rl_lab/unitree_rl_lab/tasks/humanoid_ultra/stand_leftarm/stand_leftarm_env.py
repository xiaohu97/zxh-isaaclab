"""Humanoid Ultra 27-DoF 站立 + 左臂轨迹激励跟踪环境。

在 ``HumanoidUltra27dofStandEnv`` (DirectRLEnv) 基础上注入一条左臂参考轨迹：
- 用 ``np.fft.rfft`` 从 CSV 还原 Fourier 级数，运行时解析求 q_ref/dq_ref（C∞ 连续、周期）。
- 先用五次 smoothstep 展开到安全肩部姿态，再平滑接入轨迹，避免 yaw 大角度旋转时扫过身体。
- 训练 episode 末尾按相反路线渐出，让策略学会实机关闭激励时安全收手。
- 每个 env 在 reset 时按 ``rel_enabled_envs`` 随机开/关跟踪；关闭时目标=默认位姿(收手)。
- 把 15 维 ``[q_ref_rel(7), dq_ref(7)*ref_vel_scale, enabled(1)]`` 追加到 actor/critic 观测末尾，
  奖励项(见 cfg)按此跟踪。

设计与 G1 的 ``Unitree-G1-29dof-Stand-LeftArmTrack-v0`` 对齐，但适配 DirectRLEnv 架构。
"""
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

from unitree_rl_lab.tasks.humanoid_ultra.stand.stand_env import HumanoidUltra27dofStandEnv


class HumanoidUltra27dofStandLeftArmEnv(HumanoidUltra27dofStandEnv):
    """站立 + 左臂 Fourier 轨迹跟踪（含开关与平滑渐入）。"""

    # ------------------------------------------------------------------ arm setup
    def _ensure_arm(self):
        """惰性初始化左臂轨迹状态（需在 robot 初始化后；幂等）。"""
        if getattr(self, "_arm_ready", False):
            return
        cfg = self.cfg.arm_command
        self.arm_period = float(cfg.period)
        self.arm_safe_time = float(cfg.safe_pose_time_s)
        self.arm_blend_time = float(cfg.blend_time_s)
        self.arm_vscale = float(cfg.ref_vel_scale)
        self.arm_rel_enabled = float(cfg.rel_enabled_envs)
        self.arm_randomize_phase = bool(cfg.randomize_start_phase)
        self.arm_auto_fade_out = bool(cfg.auto_fade_out)
        if self.arm_safe_time < 0.0 or self.arm_blend_time < 0.0:
            raise ValueError("Left-arm transition times must be non-negative.")
        transition_time = 2.0 * (self.arm_safe_time + self.arm_blend_time)
        if self.arm_auto_fade_out and transition_time > self.max_episode_length_s:
            raise ValueError("Left-arm safe/blend transitions do not fit in one episode.")

        joint_ids = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)[0]
        self.arm_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.arm_num_joints = len(joint_ids)
        self.arm_default_q = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        if len(cfg.safe_joint_pos) != self.arm_num_joints:
            raise ValueError(
                f"safe_joint_pos has {len(cfg.safe_joint_pos)} values; expected {self.arm_num_joints}."
            )
        self.arm_safe_q = (
            torch.tensor(cfg.safe_joint_pos, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )

        self._build_fourier(cfg.traj_file, list(cfg.joint_names))

        zeros = torch.zeros(self.num_envs, device=self.device)
        self.arm_enabled = zeros.clone()
        self.arm_phase_offset = zeros.clone()
        self.arm_q_ref_rel = torch.zeros(self.num_envs, self.arm_num_joints, device=self.device)
        self.arm_dq_ref = torch.zeros(self.num_envs, self.arm_num_joints, device=self.device)
        self._arm_ready = True

    def _build_fourier(self, traj_file: str, joint_names: list[str]):
        with open(traj_file, "r") as f:
            header = f.readline().strip().split(",")
        raw = np.loadtxt(traj_file, delimiter=",", skiprows=1)
        col_idx = [header.index(name) for name in joint_names]
        q = raw[:, col_idx]
        n_samples = q.shape[0]
        coeff = np.fft.rfft(q, axis=0)
        a = (2.0 / n_samples) * coeff.real
        b = (-2.0 / n_samples) * coeff.imag
        a[0] *= 0.5
        if n_samples % 2 == 0:
            a[-1] *= 0.5
            b[-1] = 0.0
        k = np.arange(coeff.shape[0])
        omega = 2.0 * np.pi * k / self.arm_period
        self.arm_fa = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.arm_fb = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.arm_omega = torch.tensor(omega, dtype=torch.float32, device=self.device)

    def _fourier_eval(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ang = phase.unsqueeze(1) * self.arm_omega.unsqueeze(0)
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        q = cos @ self.arm_fa + sin @ self.arm_fb
        w = self.arm_omega.unsqueeze(0)
        dq = (-(sin * w)) @ self.arm_fa + (cos * w) @ self.arm_fb
        return q, dq

    @staticmethod
    def _smoothstep(elapsed: torch.Tensor, duration: float) -> tuple[torch.Tensor, torch.Tensor]:
        if duration <= 0.0:
            return torch.ones_like(elapsed), torch.zeros_like(elapsed)
        u = torch.clamp(elapsed / duration, 0.0, 1.0)
        alpha = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
        alpha_dot = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2) / duration
        return alpha, alpha_dot

    def _reset_arm(self, env_ids: Sequence[int]):
        self._ensure_arm()
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        self.arm_enabled[env_ids] = (
            torch.rand(n, device=self.device) < self.arm_rel_enabled
        ).float()
        if self.arm_randomize_phase:
            self.arm_phase_offset[env_ids] = torch.rand(n, device=self.device) * self.arm_period
        else:
            self.arm_phase_offset[env_ids] = 0.0

    def _refresh_arm_ref(self):
        """从 episode_length_buf 解析出当前 q_ref_rel / dq_ref（纯函数，可重复调用）。"""
        self._ensure_arm()
        elapsed = self.episode_length_buf.float() * self.step_dt
        # 轨迹时钟在安全姿态到达后才启动，因此 phase_offset 表示真正接轨时的
        # 起始相位；部署端使用相同语义。
        trajectory_elapsed = torch.clamp(elapsed - self.arm_safe_time, min=0.0)
        phase = torch.remainder(trajectory_elapsed + self.arm_phase_offset, self.arm_period)
        q_traj, dq_traj = self._fourier_eval(phase)
        q_ref = self.arm_default_q.clone()
        dq_ref = torch.zeros_like(q_ref)

        # Stage 1: default -> safe pose.  Only shoulder pitch/roll differ in the
        # default configuration, so the large yaw rotations happen outside the body.
        to_safe = elapsed < self.arm_safe_time
        alpha, alpha_dot = self._smoothstep(elapsed, self.arm_safe_time)
        safe_delta = self.arm_safe_q - self.arm_default_q
        q_safe_in = self.arm_default_q + alpha.unsqueeze(1) * safe_delta
        dq_safe_in = alpha_dot.unsqueeze(1) * safe_delta
        q_ref = torch.where(to_safe.unsqueeze(1), q_safe_in, q_ref)
        dq_ref = torch.where(to_safe.unsqueeze(1), dq_safe_in, dq_ref)

        # Stage 2: safe pose -> moving Fourier trajectory.
        blend_elapsed = elapsed - self.arm_safe_time
        blend_end = self.arm_safe_time + self.arm_blend_time
        to_track = torch.logical_and(elapsed >= self.arm_safe_time, elapsed < blend_end)
        beta, beta_dot = self._smoothstep(blend_elapsed, self.arm_blend_time)
        track_delta = q_traj - self.arm_safe_q
        q_track_in = self.arm_safe_q + beta.unsqueeze(1) * track_delta
        dq_track_in = beta_dot.unsqueeze(1) * track_delta + beta.unsqueeze(1) * dq_traj
        q_ref = torch.where(to_track.unsqueeze(1), q_track_in, q_ref)
        dq_ref = torch.where(to_track.unsqueeze(1), dq_track_in, dq_ref)

        if self.arm_auto_fade_out:
            fade_start = self.max_episode_length_s - self.arm_safe_time - self.arm_blend_time
        else:
            fade_start = float("inf")

        tracking = torch.logical_and(elapsed >= blend_end, elapsed < fade_start)
        q_ref = torch.where(tracking.unsqueeze(1), q_traj, q_ref)
        dq_ref = torch.where(tracking.unsqueeze(1), dq_traj, dq_ref)

        if self.arm_auto_fade_out:
            # Stage 3: moving trajectory -> safe pose.
            fade_elapsed = elapsed - fade_start
            fade_end = fade_start + self.arm_blend_time
            from_track = torch.logical_and(elapsed >= fade_start, elapsed < fade_end)
            gamma, gamma_dot = self._smoothstep(fade_elapsed, self.arm_blend_time)
            q_track_out = q_traj + gamma.unsqueeze(1) * (self.arm_safe_q - q_traj)
            dq_track_out = (
                (1.0 - gamma).unsqueeze(1) * dq_traj
                + gamma_dot.unsqueeze(1) * (self.arm_safe_q - q_traj)
            )
            q_ref = torch.where(from_track.unsqueeze(1), q_track_out, q_ref)
            dq_ref = torch.where(from_track.unsqueeze(1), dq_track_out, dq_ref)

            # Stage 4: safe pose -> default pose.
            return_elapsed = elapsed - fade_end
            to_default = elapsed >= fade_end
            eta, eta_dot = self._smoothstep(return_elapsed, self.arm_safe_time)
            q_default_out = self.arm_safe_q + eta.unsqueeze(1) * (
                self.arm_default_q - self.arm_safe_q
            )
            dq_default_out = eta_dot.unsqueeze(1) * (self.arm_default_q - self.arm_safe_q)
            q_ref = torch.where(to_default.unsqueeze(1), q_default_out, q_ref)
            dq_ref = torch.where(to_default.unsqueeze(1), dq_default_out, dq_ref)

        enabled = self.arm_enabled.unsqueeze(1)
        self.arm_q_ref_rel = enabled * (q_ref - self.arm_default_q)
        self.arm_dq_ref = enabled * dq_ref

    # ------------------------------------------------------------------ hooks
    def compute_current_observations(self):
        actor_obs, critic_obs = super().compute_current_observations()
        self._refresh_arm_ref()
        arm_obs = torch.cat(
            [self.arm_q_ref_rel, self.arm_dq_ref * self.arm_vscale, self.arm_enabled.unsqueeze(1)],
            dim=-1,
        )
        actor_obs = torch.cat([actor_obs, arm_obs], dim=-1)
        critic_obs = torch.cat([critic_obs, arm_obs], dim=-1)
        return actor_obs, critic_obs

    def _get_rewards(self):
        # 在算奖励前刷新参考，保证奖励与观测看到同一帧目标
        self._refresh_arm_ref()
        return super()._get_rewards()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._reset_arm(env_ids)
