"""左臂关节轨迹跟踪命令项 (Left-arm joint trajectory command).

设计要点
--------
1. **Fourier 还原**：CSV 是辨识激励轨迹在一个周期内的均匀采样（默认 30 点 / 6.0s）。
   线性插值会让参考速度每 0.2s 阶跃，不利于扭矩平滑与 sim2real。这里在 ``__init__``
   用 ``np.fft.rfft`` 还原成有限 Fourier 级数，运行时解析求 ``q_ref``/``dq_ref``，
   位置/速度/加速度全程 C∞ 连续且天然周期。
2. **平滑渐入**：每次 reset 时机器人左臂停在默认位姿，与轨迹起点差异较大。用五次
   smoothstep ``α=6u⁵-15u⁴+10u³`` 把目标从默认位姿混合到轨迹（``α'(0)=α'(T)=0``，
   边界处位置/速度/加速度均连续）。相位在渐入期照常推进；``dq_ref`` 含 blend 导数项。
3. **可开关**：每个 env 一个 ``enabled`` 标志（进观测）。训练时按 ``rel_enabled_envs``
   随机开/关；关闭时 ``q_ref_rel=dq_ref=0``（目标=默认位姿），策略学到“收手”。
   部署时把这一位接到手柄即可实时切换跟踪 / 收手。

观测向量 (``command`` 属性, 维度 = 2*n_joint + 1)::

    [ q_ref_rel(n) , dq_ref(n) * ref_vel_scale , enabled(1) ]

其中 ``q_ref_rel = target_abs - default`` 与 ``joint_pos_rel`` 同口径。
"""
from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LeftArmJointTrajectoryCommand(CommandTerm):
    """输出左臂关节的参考位置/速度（相对默认位姿），支持渐入与开关。"""

    cfg: LeftArmJointTrajectoryCommandCfg

    def __init__(self, cfg: LeftArmJointTrajectoryCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # 按名字解析关节索引（Isaac 内部顺序与 SDK 顺序不同，必须按名字！），保持顺序
        joint_ids, joint_names = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.joint_names = joint_names
        self.num_joints = len(joint_ids)

        # 控制步长（policy dt）与渐入时长
        self.dt = float(env.cfg.sim.dt * env.cfg.decimation)
        self.period = float(cfg.period)
        self.blend_time = float(cfg.blend_time_s)

        # 从 CSV 还原 Fourier 级数系数 a/b 与角频率 omega
        self._build_fourier(cfg.traj_file)

        # default_joint_pos 在 sim 启动后才可用，延迟缓存
        self._default_q: torch.Tensor | None = None

        # 每个 env 的状态
        self.phase = torch.zeros(self.num_envs, device=self.device)     # 轨迹相位 [s]
        self.elapsed = torch.zeros(self.num_envs, device=self.device)   # 自 reset 起 [s] (渐入用)
        self.enabled = torch.ones(self.num_envs, device=self.device)    # 1=跟踪, 0=收手
        # 输出缓存（相对默认位姿）
        self.q_ref_rel = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.dq_ref = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        self.metrics["error_left_arm_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_enabled"] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ Fourier
    def _build_fourier(self, traj_file: str):
        assert os.path.isfile(traj_file), f"Invalid trajectory file: {traj_file}"
        with open(traj_file, "r") as f:
            header = f.readline().strip().split(",")
        raw = np.loadtxt(traj_file, delimiter=",", skiprows=1)  # (N, 1+n)
        # 按 cfg.joint_names 的顺序选列，避免 CSV 列序与关节序不一致
        col_idx = [header.index(name) for name in self.cfg.joint_names]
        q = raw[:, col_idx]                       # (N, n)
        n_samples = q.shape[0]
        coeff = np.fft.rfft(q, axis=0)            # (K, n) complex, K = N//2 + 1
        a = (2.0 / n_samples) * coeff.real        # cos 系数
        b = (-2.0 / n_samples) * coeff.imag       # sin 系数
        a[0] *= 0.5                               # 直流项 -> 均值
        if n_samples % 2 == 0:                    # 偶数采样: Nyquist 项只算一次
            a[-1] *= 0.5
            b[-1] = 0.0
        k = np.arange(coeff.shape[0])
        omega = 2.0 * np.pi * k / self.period     # 各次谐波角频率

        self.fa = torch.tensor(a, dtype=torch.float32, device=self.device)        # (K, n)
        self.fb = torch.tensor(b, dtype=torch.float32, device=self.device)        # (K, n)
        self.omega = torch.tensor(omega, dtype=torch.float32, device=self.device)  # (K,)

    def _fourier_eval(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """解析求轨迹位置与速度。phase: (M,) -> (q (M,n), dq (M,n))。"""
        ang = phase.unsqueeze(1) * self.omega.unsqueeze(0)   # (M, K)
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        q = cos @ self.fa + sin @ self.fb                    # (M, n)
        w = self.omega.unsqueeze(0)
        dq = (-(sin * w)) @ self.fa + (cos * w) @ self.fb    # d/dt
        return q, dq

    # ------------------------------------------------------------------ helpers
    @property
    def default_q(self) -> torch.Tensor:
        if self._default_q is None:
            self._default_q = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        return self._default_q

    @property
    def target_abs(self) -> torch.Tensor:
        """绝对目标关节位置 (num_envs, n)。"""
        return self.default_q + self.q_ref_rel

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.q_ref_rel, self.dq_ref * self.cfg.ref_vel_scale, self.enabled.unsqueeze(1)], dim=1
        )

    # ------------------------------------------------------------------ lifecycle
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self.elapsed[env_ids] = 0.0
        if self.cfg.randomize_start_phase:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * self.period
        else:
            self.phase[env_ids] = 0.0
        # 随机开/关激励
        self.enabled[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.rel_enabled_envs
        ).float()
        # elapsed=0 时 α=α'=0 -> 目标=默认位姿、速度=0
        self.q_ref_rel[env_ids] = 0.0
        self.dq_ref[env_ids] = 0.0

    def _update_command(self):
        self.elapsed += self.dt
        self.phase = torch.remainder(self.phase + self.dt, self.period)

        q_traj, dq_traj = self._fourier_eval(self.phase)     # (N, n)
        delta = q_traj - self.default_q                       # 默认 -> 轨迹 的偏移

        if self.blend_time > 0.0:
            u = torch.clamp(self.elapsed / self.blend_time, 0.0, 1.0)
            alpha = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
            dalpha = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2) / self.blend_time
        else:
            alpha = torch.ones_like(self.elapsed)
            dalpha = torch.zeros_like(self.elapsed)

        en = self.enabled.unsqueeze(1)
        a = alpha.unsqueeze(1)
        da = dalpha.unsqueeze(1)
        self.q_ref_rel = en * (a * delta)
        # 链式法则: d/dt[default + α·δ] = α'·δ + α·δ̇   (δ̇ = dq_traj, phase 实时推进)
        self.dq_ref = en * (da * delta + a * dq_traj)

    def _update_metrics(self):
        cur_rel = self.robot.data.joint_pos[:, self.joint_ids] - self.default_q
        self.metrics["error_left_arm_pos"] = torch.norm(cur_rel - self.q_ref_rel, dim=1)
        self.metrics["tracking_enabled"] = self.enabled


@configclass
class LeftArmJointTrajectoryCommandCfg(CommandTermCfg):
    """``LeftArmJointTrajectoryCommand`` 的配置。"""

    class_type: type = LeftArmJointTrajectoryCommand

    asset_name: str = MISSING
    """机器人 asset 名。"""

    joint_names: list[str] = MISSING
    """被跟踪的关节名（顺序需与 CSV 列对应；内部按名字解析索引）。"""

    traj_file: str = MISSING
    """轨迹 CSV 路径（第一列为时间 t，其余列为各关节角）。"""

    period: float = 6.0
    """轨迹周期 [s]（用于 Fourier 还原）。"""

    blend_time_s: float = 1.5
    """默认位姿 -> 轨迹 的五次 smoothstep 渐入时长 [s]；<=0 表示瞬时切入。"""

    rel_enabled_envs: float = 0.8
    """训练时开启跟踪的 env 比例；其余 env 收手到默认位姿。play 时设 1.0。"""

    randomize_start_phase: bool = True
    """是否随机起始相位（提高覆盖）；play / 调试时设 False 从相位 0 起。"""

    ref_vel_scale: float = 0.25
    """观测中参考速度 dq_ref 的缩放，使其量级 ~O(1)。"""

    toggle_button: str = "RB + A.on_pressed"
    """部署端(C++)切换手臂激励开/关的手柄组合键(joystick DSL 语法)。仅导出到 deploy.yaml，
    供 g1_ctrl 的 arm_command 观测使用；训练侧不读取。"""
