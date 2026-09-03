"""HugWBC 式「速度 + 步态参数」命令项。

设计要点
--------
1. **一个命令项同时输出任务指令与行为指令**（HugWBC 的 command space 思路）::

       [ vx, vy, ωz | 步频 f, 支撑相比例 θ, 摆动腿高度 h_f, 躯干高度 h_z, 躯干俯仰 p ]

   前 3 维与 IsaacLab 原生速度命令完全兼容（``track_lin_vel_xy_yaw_frame_exp`` 取
   ``[:, :2]``、``track_ang_vel_z_exp`` 取 ``[:, 2]``），后 5 维是"想让它怎么走"。
   训练完这 5 个量就是手柄上可以实时拧的旋钮。

2. **θ < 0.5 就有腾空期**。两腿相位固定差半个周期，每腿支撑相占比 θ，于是
   双支撑重叠 = max(0, 2θ-1)、腾空占比 = max(0, 1-2θ)。所以 θ 这一个标量把
   「走 ↔ 跑」连续参数化了 —— 这是本任务能跑起来的核心，不是速度上限。

3. **相位必须积分，不能用 episode 时间反推**。步频逐 env 不同且会被重采样，
   用 ``t % T / T`` 会在改频率的瞬间产生相位跳变，策略学不出稳定步态。
   这里 ``phase += f · dt``，改频率时相位连续。

4. **课程用一个标量 progress∈[0,1] 统一插值**，把区间从 ``cfg.ranges``（好学的走路
   区间）线性拉到 ``cfg.limit_ranges``（跑步区间）。比原地改 ranges 元组干净，
   而且观测归一化始终用 limit_ranges，课程推进时策略看到的数字含义不变。

5. **两条物理自洽约束**（见 ``_apply_feasibility``）。步态参数与速度独立采样会产生
   「1.2 Hz 步频跑 3 m/s」这种不可能的指令，策略只能学会忽略它们，可控性就没了。
   所以采样后按 ``v ≤ f · max_stride_length`` 削速度，并在高速时强制 θ 进入腾空区。
"""
from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# command 张量各分量下标。rewards / observations 一律从这里取，不要写魔数。
IDX_LIN_VEL_X = 0
IDX_LIN_VEL_Y = 1
IDX_ANG_VEL_Z = 2
IDX_GAIT_FREQ = 3
IDX_STANCE_RATIO = 4
IDX_SWING_HEIGHT = 5
IDX_BODY_HEIGHT = 6
IDX_BODY_PITCH = 7

COMMAND_DIM = 8

# 行为指令字段名，顺序必须与上面 IDX_GAIT_FREQ 起的下标一致
GAIT_FIELDS = ("gait_freq", "stance_ratio", "swing_height", "body_height", "body_pitch")


class GaitCommand(CommandTerm):
    """速度 + 步态参数命令，支持课程插值与相位积分。"""

    cfg: GaitCommandCfg

    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.command_b = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        # 全局步态相位 ∈ [0, 1)，左腿为基准
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 课程进度：0 = cfg.ranges（走路），1 = cfg.limit_ranges（跑步）
        self.progress: float = 0.0

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_max_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_min_stance_ratio"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {COMMAND_DIM}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tCurriculum progress: {self.progress:.3f}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """原始物理量纲的指令 (num_envs, 8)。奖励项用这个。"""
        return self.command_b

    @property
    def command_obs(self) -> torch.Tensor:
        """进观测的指令 (num_envs, 8)。

        速度保持物理量纲（与 IsaacLab 惯例一致），5 个步态参数按 **limit_ranges**
        归一到 [-1, 1]。刻意不用当前 ranges：课程推进时观测口径必须不变，否则同一个
        数字在训练前后含义不同，等于中途换了观测空间。
        """
        obs = self.command_b.clone()
        for i, name in enumerate(GAIT_FIELDS):
            lo, hi = getattr(self.cfg.limit_ranges, name)
            span = max(hi - lo, 1e-6)
            obs[:, IDX_GAIT_FREQ + i] = 2.0 * (obs[:, IDX_GAIT_FREQ + i] - lo) / span - 1.0
        return obs

    @property
    def leg_phase(self) -> torch.Tensor:
        """(num_envs, 2) 左/右腿相位 ∈ [0,1)，右腿固定滞后半个周期。"""
        return torch.stack([self.phase, torch.remainder(self.phase + 0.5, 1.0)], dim=1)

    @property
    def is_standing(self) -> torch.Tensor:
        """(num_envs,) bool：该 env 此刻应当站立（双脚落地、不摆臂、回默认位姿）。

        判据是**观测里能看到的速度指令范数** < ``standing_threshold``，而不是内部的
        ``is_standing_env`` 采样标志 —— 策略看不到那个标志。若两个观测完全相同的 env
        （都是零速度指令）一个被要求按时钟踏步、一个被要求站着不动，监督信号自相矛盾，
        零速附近的行为会学成两者的平均。用观测量做判据，站立就是策略可学的规则：
        指令归零 -> 双脚落地。``is_standing_env`` 只负责把速度指令置零。
        """
        return torch.norm(self.command_b[:, IDX_LIN_VEL_X : IDX_ANG_VEL_Z + 1], dim=1) < self.cfg.standing_threshold

    @property
    def desired_contact(self) -> torch.Tensor:
        """(num_envs, 2) bool：该腿此刻是否应当支撑。站立时强制双脚触地。"""
        theta = self.command_b[:, IDX_STANCE_RATIO].unsqueeze(1)
        stance = self.leg_phase < theta
        return torch.where(self.is_standing.unsqueeze(1), torch.ones_like(stance), stance)

    @property
    def desired_foot_height(self) -> torch.Tensor:
        """(num_envs, 2) 参考足高（世界系 z，相对地面）。

        支撑相恒为 ``foot_offset``（足底贴地时 ankle_roll 原点的高度），摆动相走半个
        正弦拱到指令高度。给出完整参考轨迹而不是"最高点"，摆动腿高度才真的可控。
        """
        theta = self.command_b[:, IDX_STANCE_RATIO].unsqueeze(1)
        # u ∈ [0,1] 是摆动相内的进度；支撑相被 clamp 到 0 -> sin(0)=0 -> 目标即贴地
        u = ((self.leg_phase - theta) / (1.0 - theta).clamp(min=1e-3)).clamp(0.0, 1.0)
        u = torch.where(self.is_standing.unsqueeze(1), torch.zeros_like(u), u)
        swing_h = self.command_b[:, IDX_SWING_HEIGHT].unsqueeze(1)
        return self.cfg.foot_offset + swing_h * torch.sin(math.pi * u)

    """
    Implementation specific functions.
    """

    def _range(self, name: str) -> tuple[float, float]:
        """按课程进度在 ranges(走) 与 limit_ranges(跑) 之间线性插值。"""
        lo0, hi0 = getattr(self.cfg.ranges, name)
        lo1, hi1 = getattr(self.cfg.limit_ranges, name)
        p = self.progress
        return lo0 + (lo1 - lo0) * p, hi0 + (hi1 - hi0) * p

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # 相位只在真正 reset 时随机化。中途重采样（resampling_time_range）不能动相位，
        # 否则每次换指令都会有一次相位跳变。
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self.cfg.randomize_start_phase:
            self.phase[ids] = torch.rand_like(self.phase[ids])
        else:
            self.phase[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        self.command_b[env_ids, IDX_LIN_VEL_X] = r.uniform_(*self._range("lin_vel_x"))
        self.command_b[env_ids, IDX_LIN_VEL_Y] = r.uniform_(*self._range("lin_vel_y"))
        self.command_b[env_ids, IDX_ANG_VEL_Z] = r.uniform_(*self._range("ang_vel_z"))
        for i, name in enumerate(GAIT_FIELDS):
            self.command_b[env_ids, IDX_GAIT_FREQ + i] = r.uniform_(*self._range(name))

        self._apply_feasibility(env_ids)

        self.is_standing_env[env_ids] = torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_standing_envs

    def _apply_feasibility(self, env_ids: Sequence[int]):
        """把独立采样出来的速度/步态参数拉回物理上可能实现的组合。

        不做这一步，训练集里会混进大量"步频 1.2 Hz 跑 3 m/s"式的自相矛盾指令，
        策略学到的最优行为是忽略步态参数，可控性直接没了。
        """
        freq = self.command_b[env_ids, IDX_GAIT_FREQ]
        theta = self.command_b[env_ids, IDX_STANCE_RATIO]
        vx = self.command_b[env_ids, IDX_LIN_VEL_X]

        # 1) 一个步态周期最多迈 max_stride_length，于是 |vx| ≤ f · stride
        if self.cfg.max_stride_length > 0.0:
            v_cap = freq * self.cfg.max_stride_length
            vx = torch.clamp(vx, -v_cap, v_cap)

        # 2) 超过 flight_speed_threshold 就必须有腾空期，否则是不可能的"高速快走"
        needs_flight = vx.abs() > self.cfg.flight_speed_threshold
        theta = torch.where(needs_flight, theta.clamp(max=self.cfg.running_stance_ratio), theta)

        self.command_b[env_ids, IDX_LIN_VEL_X] = vx
        self.command_b[env_ids, IDX_STANCE_RATIO] = theta

    def _update_command(self):
        # 站立 env：三个速度分量清零（步态参数保留，desired_contact 已强制双脚支撑）
        self.command_b[self.is_standing_env, IDX_LIN_VEL_X : IDX_ANG_VEL_Z + 1] = 0.0
        # 相位积分（见文件头第 3 点）
        self.phase = torch.remainder(
            self.phase + self.command_b[:, IDX_GAIT_FREQ] * self._env.step_dt, 1.0
        )

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.command_b[:, IDX_ANG_VEL_Z] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        # 课程状态：直接看 TensorBoard 就知道推进到哪了
        self.metrics["cmd_progress"][:] = self.progress
        self.metrics["cmd_max_lin_vel_x"][:] = self._range("lin_vel_x")[1]
        self.metrics["cmd_min_stance_ratio"][:] = self._range("stance_ratio")[0]

    """
    Debug visualization (照搬 UniformVelocityCommand 的速度箭头)。
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class GaitCommandCfg(CommandTermCfg):
    """``GaitCommand`` 的配置。"""

    class_type: type = GaitCommand

    asset_name: str = MISSING
    """机器人 asset 名。"""

    @configclass
    class Ranges:
        """采样区间。``ranges`` 是课程起点（走路），``limit_ranges`` 是终点（跑步）。"""

        lin_vel_x: tuple[float, float] = MISSING
        """前向速度指令 [m/s]。"""

        lin_vel_y: tuple[float, float] = MISSING
        """侧向速度指令 [m/s]。"""

        ang_vel_z: tuple[float, float] = MISSING
        """偏航角速度指令 [rad/s]。"""

        gait_freq: tuple[float, float] = MISSING
        """步态频率 [Hz]，一个完整周期（左右各迈一步）的频率。"""

        stance_ratio: tuple[float, float] = MISSING
        """单腿支撑相占比 ∈ (0,1)。>0.5 有双支撑（走），<0.5 有腾空期（跑）。"""

        swing_height: tuple[float, float] = MISSING
        """摆动腿最高点 [m]，相对地面。"""

        body_height: tuple[float, float] = MISSING
        """骨盆目标高度 [m]。"""

        body_pitch: tuple[float, float] = MISSING
        """躯干俯仰 [rad]，正 = 前倾（G1 关节轴约定下 +pitch 使躯干 x 轴朝下）。"""

    ranges: Ranges = MISSING
    """课程起点区间（progress = 0）。"""

    limit_ranges: Ranges = MISSING
    """课程终点区间（progress = 1）。步态参数上应包含 ranges，观测归一化以它为准。"""

    rel_standing_envs: float = 0.02
    """站立（零速度指令）env 比例。"""

    standing_threshold: float = 0.1
    """速度指令范数低于该值 [m/s] 即按站立处理（双脚落地）。判据必须是观测量，见 ``is_standing``。"""

    randomize_start_phase: bool = True
    """reset 时是否随机初始相位。play / 调试时可设 False 从相位 0 起。"""

    foot_offset: float = 0.035
    """足底贴地时 ankle_roll_link 原点的高度 [m]。

    G1 29dof 的足底碰撞球在 link 系 z = -0.03、半径 0.005，所以是 0.035。
    """

    max_stride_length: float = 1.0
    """单个步态周期的最大步幅 [m]，用于按步频削速度指令。<=0 关闭该约束。"""

    flight_speed_threshold: float = 1.8
    """超过该速度 [m/s] 就强制支撑相比例进入腾空区。"""

    running_stance_ratio: float = 0.48
    """上一条触发时 stance_ratio 的上限（<0.5 才有腾空期）。"""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
