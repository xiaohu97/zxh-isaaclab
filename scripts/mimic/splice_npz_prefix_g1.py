#!/usr/bin/env python3
"""把新 NPZ 的前缀帧替换成参考 NPZ 的原始数据（位姿必须一致，只为保住速度字段的出处）。

场景：``add_recovery_tail_g1.py`` + ``csv_to_npz.py`` 重新生成的 NPZ，前缀的关节角/刚体位姿与原参考逐位一致
（1e-6），但刚体**速度**不一致——原参考的上肢 link 速度是位置差分得到的，csv_to_npz 给的是 PhysX 解析速度，
起跳段（41~46 帧）手臂快速摆动时差到 0.95 m/s。热启动微调时不该同时换掉线上 checkpoint 学过的速度目标，
所以前缀直接拷原参考的所有逐帧字段，只在接缝那一帧起用新数据（接缝帧用新数据是因为它的中心差分跨过了接缝，
比原参考末帧的单边差分更平滑）。

用法::

    python scripts/mimic/splice_npz_prefix_g1.py --reference ref.npz --target new.npz --frames 89 [--pos-tol 1e-5]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

PER_FRAME = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--target", required=True, help="就地覆盖")
    ap.add_argument("--frames", type=int, required=True, help="拷贝参考的 [0, frames) 帧")
    ap.add_argument("--pos-tol", type=float, default=1e-5)
    a = ap.parse_args()

    ref = dict(np.load(a.reference))
    new = dict(np.load(a.target))
    n = a.frames
    if not (0 < n <= ref["joint_pos"].shape[0] and n < new["joint_pos"].shape[0]):
        sys.exit(f"frames={n} 超出范围（参考 {ref['joint_pos'].shape[0]} 帧，目标 {new['joint_pos'].shape[0]} 帧）")
    if float(np.asarray(ref["fps"]).reshape(-1)[0]) != float(np.asarray(new["fps"]).reshape(-1)[0]):
        sys.exit("fps 不一致")
    for k in ("joint_pos", "body_pos_w"):
        err = float(np.abs(ref[k][:n] - new[k][:n]).max())
        if err > a.pos_tol:
            sys.exit(f"{k} 前缀不一致（max|Δ|={err:.2e} > {a.pos_tol}），拒绝拼接")
    qerr = np.minimum(np.abs(ref["body_quat_w"][:n] - new["body_quat_w"][:n]), np.abs(ref["body_quat_w"][:n] + new["body_quat_w"][:n]))
    if float(qerr.max()) > a.pos_tol:
        sys.exit(f"body_quat_w 前缀不一致（max|Δ|={qerr.max():.2e}）")
    report = {}
    for k in PER_FRAME:
        report[k] = float(np.abs(ref[k][:n] - new[k][:n]).max())
        new[k] = np.concatenate([ref[k][:n].astype(new[k].dtype), new[k][n:]], axis=0)
    np.savez(a.target, **new)
    print(f"spliced {a.target}: frames [0,{n}) from {a.reference}; pre-splice max|Δ| " + ", ".join(f"{k}={v:.2e}" for k, v in report.items()))


if __name__ == "__main__":
    main()
