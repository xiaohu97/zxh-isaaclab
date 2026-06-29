"""Humanoid Ultra 27-DoF 站立 + 左臂轨迹激励跟踪环境。

在 ``HumanoidUltra27dofStandEnv`` (DirectRLEnv) 基础上注入一条左臂参考轨迹：
- 用 ``np.fft.rfft`` 从 CSV 还原 Fourier 级数，运行时解析求 q_ref/dq_ref（C∞ 连续、周期）。
- 五次 smoothstep 把目标从默认位姿平滑渐入（默认位姿与轨迹起点差异很大，需较长 blend）。
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
        self.arm_blend_time = float(cfg.blend_time_s)
        self.arm_vscale = float(cfg.ref_vel_scale)
        self.arm_rel_enabled = float(cfg.rel_enabled_envs)
        self.arm_randomize_phase = bool(cfg.randomize_start_phase)

        joint_ids = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)[0]
        self.arm_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.arm_num_joints = len(joint_ids)
        self.arm_default_q = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()

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
        phase = torch.remainder(elapsed + self.arm_phase_offset, self.arm_period)
        q_traj, dq_traj = self._fourier_eval(phase)
        delta = q_traj - self.arm_default_q

        if self.arm_blend_time > 0.0:
            u = torch.clamp(elapsed / self.arm_blend_time, 0.0, 1.0)
            alpha = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
            dalpha = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2) / self.arm_blend_time
        else:
            alpha = torch.ones_like(elapsed)
            dalpha = torch.zeros_like(elapsed)

        gate = self.arm_enabled * alpha
        gate_dot = self.arm_enabled * dalpha
        self.arm_q_ref_rel = gate.unsqueeze(1) * delta
        self.arm_dq_ref = gate_dot.unsqueeze(1) * delta + gate.unsqueeze(1) * dq_traj

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
