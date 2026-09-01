# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""用平衡策略站立 + 左臂跟踪激励轨迹，采集惯性参数辨识用的 .dat 数据集。

用途：验证"换激励目标函数能否让末端连杆惯量 I_c 变得可辨"。腿/腰交给 RL 平衡策略
(和实机一致)，左臂由环境自带的 LeftArmJointTrajectoryCommand 跟踪外部 CSV 轨迹。

输出与 ustc-humanoid-identification 的 read_data 完全同格式(行=信号, 列=时间)：
    <prefix>_robot_low_q.dat      (7+nj)  [x,y,z, qx,qy,qz,qw, joints]   ← 四元数 **w 在末位**
    <prefix>_robot_dq.dat         (6+nj)  [世界系线速度3, 机体系角速度3, joints]
    <prefix>_robot_ddq.dat        (6+nj)  [IMU线加速度3(**含重力**), 机体系角加速度3, joints]
    <prefix>_robot_tau.dat        (nj)
    <prefix>_robot_contact.dat    (2)     取值 **{1=接触, 2=腾空}**
    <prefix>_robot_ee_force.dat   (12)    双脚 6D(力3+力矩3)，力矩位暂填 0

三个容易搞错、本项目踩过坑的约定（务必别改）：
  · IsaacLab 的 root_quat_w 是 (w,x,y,z)，.dat 要 (x,y,z,w) —— 必须转序。
  · 辨识流水线默认 no_gravity_correction=false，会对 ddq[0:3] 减去 R^T·g；
    所以这里必须**加上**重力，写成 IMU 读数(静止直立时 z≈+9.81)。
  · 关节加速度必须 savgol 限带。MuJoCo/PhysX 的原始 qacc 有接触冲击尖峰，
    直接用会毁掉腿部辨识(见项目 memory: ddq-savgol-vs-raw-qacc)。

用法:
    python scripts/rsl_rl/collect_identification_dat.py \
        --task Unitree-G1-29dof-Stand-LeftArmTrack-v0 \
        --checkpoint logs/rsl_rl/unitree_g1_29dof_stand_leftarm/2026-07-01_00-33-55/exported/policy.pt \
        --traj-file /path/to/traj_dopt.csv --out-dir data/excite_dopt --duration 60 --headless
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

os.sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402  scripts/rsl_rl/cli_args.py（与 play.py 同源）

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Stand-LeftArmTrack-v0")
parser.add_argument("--traj-file", type=str, default=None,
                    help="左臂激励轨迹 CSV（t + 7 个左臂关节列）。不给则用环境自带的")
parser.add_argument("--out-dir", type=str, required=True, help=".dat 输出目录")
parser.add_argument("--prefix", type=str, default="g1", help=".dat 文件名前缀（须与 --robot-name 一致）")
parser.add_argument("--robot-urdf", type=str, default=None,
                    help="覆盖机器人 spawn 的 URDF（造真值扰动用；须与原 URDF 同目录以便 mesh 解析）")
parser.add_argument("--joint-order-file", type=str, default=None,
                    help="纯文本关节序文件(一行一个名字)，据其把 IsaacLab DOF 序重排成流水线序。"
                         "IsaacLab 环境一般没有 pinocchio，优先用这个而不是 --urdf-for-order")
parser.add_argument("--urdf-for-order", type=str, default=None,
                    help="辨识流水线所用 URDF；据其关节序把 IsaacLab DOF 序重排成 URDF 序（强烈建议给）")
parser.add_argument("--duration", type=float, default=60.0, help="采集时长(s)，settle 之后开始计")
parser.add_argument("--settle", type=float, default=3.0, help="开始记录前先让策略稳定的时长(s)")
parser.add_argument("--savgol-win", type=int, default=31, help="关节加速度 savgol 窗口(样本)，须为奇数")
parser.add_argument("--savgol-poly", type=int, default=3)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.num_envs = 1
os.sys.argv = [os.sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 以下需 app 已启动
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


import unitree_rl_lab.tasks  # noqa: F401,E402  注册任务


def _disable_randomizers(env_cfg):
    """关掉所有域随机化：辨识要的是**确定的**动力学，不是鲁棒性。"""
    for name in ("physics_material", "add_base_mass", "randomize_rigid_body_com", "scale_link_mass",
                 "scale_actuator_gains", "scale_joint_parameters", "add_left_wrist_payload", "push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {}
        env_cfg.events.reset_base.params["velocity_range"] = {}
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


class FullStateRecorder:
    """在**物理步频**记录整机状态。挂 scene.update 钩子，与 eval_leftarm_tracking 同一模式。

    只存速度不存加速度：加速度统一离线用 savgol 求导，既能限带又避免依赖
    IsaacLab 各版本不同的 body_lin_acc_w / joint_acc 语义。
    """

    def __init__(self, base_env, foot_body_ids, contact_sensor):
        self.env = base_env
        self.robot = base_env.scene["robot"]        # g1 环境没有 .robot 属性，走标准 scene 访问
        self.foot_ids = foot_body_ids
        self.contact = contact_sensor
        self._orig = base_env.scene.update
        self.active = False
        self.buf = {k: [] for k in
                    ("root_pos", "root_quat", "root_linvel_w", "root_angvel_b", "jpos", "jvel", "tau", "cforce")}

        def update_and_record(dt: float):
            self._orig(dt)
            if self.active:
                self._append()

        base_env.scene.update = update_and_record

    def _append(self):
        d = self.robot.data
        g = lambda t: t.detach().cpu().numpy()[0].copy()
        self.buf["root_pos"].append(g(d.root_pos_w))
        self.buf["root_quat"].append(g(d.root_quat_w))          # (w,x,y,z)
        self.buf["root_linvel_w"].append(g(d.root_lin_vel_w))
        self.buf["root_angvel_b"].append(g(d.root_ang_vel_b))
        self.buf["jpos"].append(g(d.joint_pos))
        self.buf["jvel"].append(g(d.joint_vel))
        tau = getattr(d, "applied_torque", None)
        if tau is None:
            tau = getattr(d, "computed_torque")
        self.buf["tau"].append(g(tau))
        if self.contact is not None:
            f = self.contact.data.net_forces_w.detach().cpu().numpy()[0]   # (nbody,3)
            self.buf["cforce"].append(f[self.foot_ids].copy())
        else:
            self.buf["cforce"].append(np.zeros((2, 3)))

    def start(self):
        for k in self.buf:
            self.buf[k] = []
        self.active = True

    def stop(self):
        self.active = False

    def restore(self):
        self.active = False
        self.env.scene.update = self._orig

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.buf.items()}


def _quat_wxyz_to_R(q):
    """(N,4) wxyz -> (N,3,3)。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def write_dat(out_dir, prefix, rec, dt, win, poly, joint_perm=None):
    os.makedirs(out_dir, exist_ok=True)
    N = len(rec["root_pos"])
    win = min(win if win % 2 == 1 else win + 1, N - 1 if (N - 1) % 2 == 1 else N - 2)
    sg = lambda x, d: savgol_filter(x, win, poly, deriv=d, delta=dt, axis=0)

    # ---- 关节顺序重映射: IsaacLab 广度优先序 -> URDF/pinocchio 序 ----
    # IsaacLab 的 joint_pos/joint_vel/applied_torque 按其内部广度优先 DOF 序排列
    # (left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...)，与流水线读取的
    # URDF 序(整条左腿->整条右腿->腰->左臂->右臂)不同，29dof 上有 27 个关节错位。
    # 不重排会让整个辨识乱套(表现为腿质量崩塌、末端 info 恒为 0)。
    if joint_perm is not None:
        for k in ("jpos", "jvel", "tau"):
            rec[k] = rec[k][:, joint_perm]

    quat_wxyz = rec["root_quat"]
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)   # → (x,y,z,w)
    low_q = np.concatenate([rec["root_pos"], quat_xyzw, rec["jpos"]], axis=1)

    dq = np.concatenate([rec["root_linvel_w"], rec["root_angvel_b"], rec["jvel"]], axis=1)

    # 加速度：统一 savgol 求导（限带 + 抑制接触冲击尖峰）
    lin_acc_w = sg(rec["root_linvel_w"], 1)
    ang_acc_b = sg(rec["root_angvel_b"], 1)
    jacc = sg(rec["jvel"], 1)
    # IMU 读数 = 机体系线加速度 + R^T·g（流水线会减掉 R^T·g，故这里必须加上）
    R = _quat_wxyz_to_R(quat_wxyz)
    g_w = np.array([0.0, 0.0, 9.81])
    lin_acc_b = np.einsum("nji,nj->ni", R, lin_acc_w)          # R^T · a_w
    imu_acc = lin_acc_b + np.einsum("nji,j->ni", R, g_w)       # + R^T · g
    ddq = np.concatenate([imu_acc, ang_acc_b, jacc], axis=1)

    cf = rec["cforce"]                                          # (N,2,3)
    ee = np.zeros((N, 12))
    ee[:, 0:3] = cf[:, 0]; ee[:, 6:9] = cf[:, 1]                # 力矩位留 0
    fz = np.stack([cf[:, 0, 2], cf[:, 1, 2]], axis=1)
    contact = np.where(np.abs(fz) > 20.0, 1.0, 2.0)             # {1=接触, 2=腾空}

    for name, arr in (("low_q", low_q), ("dq", dq), ("ddq", ddq), ("tau", rec["tau"]),
                      ("contact", contact), ("ee_force", ee)):
        np.savetxt(os.path.join(out_dir, f"{prefix}_robot_{name}.dat"), arr.T, delimiter="\t", fmt="%.6f")
    print(f"[collect] 写出 {N} 样本 @dt={dt*1000:.3f}ms -> {out_dir}")
    print(f"[collect] 自检: ddq[2] 均值={ddq[:,2].mean():+.3f} (直立静止应≈+9.8)  "
          f"接触率 左/右={np.mean(contact[:,0]==1):.2f}/{np.mean(contact[:,1]==1):.2f}")
    return N


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    _disable_randomizers(env_cfg)
    if args_cli.robot_urdf:
        spawn = env_cfg.scene.robot.spawn
        if not hasattr(spawn, "asset_path"):
            raise SystemExit(f"该环境的 spawn 不是 URDF 型 ({type(spawn).__name__})，无法用 --robot-urdf 覆盖")
        print(f"[collect] 覆盖机器人 URDF: {spawn.asset_path} -> {args_cli.robot_urdf}")
        spawn.asset_path = os.path.abspath(args_cli.robot_urdf)
    cmd = env_cfg.commands.left_arm
    if args_cli.traj_file:
        cmd.traj_file = os.path.abspath(args_cli.traj_file)
    cmd.rel_enabled_envs = 1.0            # 100% 启用臂激励
    cmd.randomize_start_phase = False     # 确定性，便于跨轨迹对比
    cmd.blend_time_s = 1.0
    print(f"[collect] 左臂轨迹: {cmd.traj_file}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped

    if not args_cli.checkpoint:
        raise SystemExit("必须用 --checkpoint 指定策略 .pt")
    ckpt = os.path.abspath(args_cli.checkpoint)
    # exported/policy.pt 是 TorchScript(已烘入观测归一化)，可直接加载；
    # model_*.pt 是训练 checkpoint，需要走 runner 并从注册表取 agent 配置。
    try:
        policy_jit = torch.jit.load(ckpt, map_location=base_env.device).eval()
        policy = lambda o: policy_jit(o)
        print(f"[collect] 以 TorchScript 加载策略: {ckpt}")
    except (RuntimeError, ValueError):
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=base_env.device)
        print(f"[collect] 以训练 checkpoint 加载策略: {ckpt}")

    contact_sensor = base_env.scene.sensors.get("contact_forces")
    # ContactSensor 有**自己的** body 顺序，与 Articulation.body_names 不一定一致；
    # 用关节体索引去查会读到别的 body（表现为足底力恒为 0）。必须用 sensor.body_names。
    foot_ids = []
    if contact_sensor is not None:
        cnames = list(contact_sensor.body_names)
        for pat in ("left_ankle_roll", "right_ankle_roll"):
            hit = [i for i, n in enumerate(cnames) if pat in n]
            if not hit:
                raise SystemExit(f"接触传感器里找不到 {pat}，可用: {cnames}")
            foot_ids.append(hit[0])
        print(f"[collect] 足部接触体: {[cnames[i] for i in foot_ids]} (传感器共 {len(cnames)} 体)")
    else:
        print("[collect] 场景无 contact_forces 传感器 -> ee_force 全 0，辨识须走零空间投影")

    # IsaacLab DOF 序 -> URDF 序的置换（见 write_dat 内注释）
    joint_perm = None
    if args_cli.joint_order_file or args_cli.urdf_for_order:
        if args_cli.joint_order_file:
            # 纯文本，一行一个关节名（由辨识侧用 pinocchio 导出）。IsaacLab 环境通常没装 pinocchio，
            # 所以这条路径是首选；--urdf-for-order 只在本环境恰好有 pinocchio 时可用。
            with open(args_cli.joint_order_file) as f:
                urdf_names = [l.strip() for l in f if l.strip()]
        else:
            import pinocchio as pin
            pm = pin.buildModelFromUrdf(args_cli.urdf_for_order, pin.JointModelFreeFlyer())
            urdf_names = list(pm.names[2:])
        isaac_names = list(base_env.scene["robot"].data.joint_names)
        if sorted(urdf_names) != sorted(isaac_names):
            raise SystemExit(f"关节名集合不匹配\n  URDF独有: {set(urdf_names)-set(isaac_names)}"
                             f"\n  Isaac独有: {set(isaac_names)-set(urdf_names)}")
        joint_perm = np.array([isaac_names.index(n) for n in urdf_names])
        n_moved = int((joint_perm != np.arange(len(joint_perm))).sum())
        print(f"[collect] 关节序重映射 Isaac->URDF: {len(joint_perm)} 个关节，{n_moved} 个需移位")
    else:
        print("[collect] 警告: 未给 --urdf-for-order，关节按 IsaacLab 原序写出（多半与流水线不一致！）")

    rec = FullStateRecorder(base_env, foot_ids, contact_sensor)
    dt_phys = base_env.physics_dt
    n_settle = int(args_cli.settle / base_env.step_dt)
    n_run = int(args_cli.duration / base_env.step_dt)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(n_settle):
            obs, _, _, _ = env.step(policy(obs))
        rec.start()
        for _ in range(n_run):
            obs, _, _, _ = env.step(policy(obs))
        rec.stop()

    write_dat(args_cli.out_dir, args_cli.prefix, rec.arrays(), dt_phys,
              args_cli.savgol_win, args_cli.savgol_poly, joint_perm)
    rec.restore()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
