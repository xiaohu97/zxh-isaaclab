"""订阅感知 walk 用的高程图话题，打印频率和内容统计。上机前用它确认建图节点 / MuJoCo 发布器的输出。

    python deploy/robots/g1_29dof/tools/height_map_echo.py --interface eth0          # 真机
    python deploy/robots/g1_29dof/tools/height_map_echo.py --interface lo --domain 1  # unitree_mujoco

每秒一行：频率、网格尺寸、分辨率、原点、NaN 格数、最小/最大值、躯干正下方格的值（平地应约 -0.78）。
尺寸 / 分辨率 / 原点和控制器 config.yaml 的 height_map.grid 不一致时控制器会整张图按平地处理，这里会标 MISMATCH。
"""
from __future__ import annotations

import argparse
import math
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_

EXPECTED = {"width": 17, "height": 11, "resolution": 0.1, "origin": (-0.8, -0.5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--topic", default="rt/perceptive/height_map")
    parser.add_argument("--seconds", type=float, default=0.0, help="运行时长，0 = 一直跑")
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)
    lock = threading.Lock()
    state = {"count": 0, "last": None}

    def handler(msg: HeightMap_) -> None:
        with lock:
            state["count"] += 1
            state["last"] = msg

    sub = ChannelSubscriber(args.topic, HeightMap_)
    sub.Init(handler, 10)
    print(f"listening on '{args.topic}' (domain {args.domain}, {args.interface})")
    start = time.time()
    while args.seconds <= 0 or time.time() - start < args.seconds:
        time.sleep(1.0)
        with lock:
            count, msg = state["count"], state["last"]
            state["count"] = 0
        if msg is None:
            print("no message yet")
            continue
        data = list(msg.data)
        finite = [v for v in data if math.isfinite(v)]
        nan_count = len(data) - len(finite)
        mismatch = (
            msg.width != EXPECTED["width"]
            or msg.height != EXPECTED["height"]
            or abs(msg.resolution - EXPECTED["resolution"]) > 1e-4
            or abs(msg.origin[0] - EXPECTED["origin"][0]) > 1e-3
            or abs(msg.origin[1] - EXPECTED["origin"][1]) > 1e-3
            or len(data) != msg.width * msg.height
        )
        center = data[(msg.height // 2) * msg.width + msg.width // 2] if len(data) == msg.width * msg.height else math.nan
        print(
            f"{count:3d} Hz  {msg.width}x{msg.height} @ {msg.resolution:.3f} m origin=({msg.origin[0]:.2f}, {msg.origin[1]:.2f})"
            f"  nan={nan_count:3d}  min={min(finite) if finite else math.nan:.3f}  max={max(finite) if finite else math.nan:.3f}"
            f"  under_torso={center:.3f}  frame_id='{msg.frame_id}'" + ("  MISMATCH" if mismatch else "")
        )


if __name__ == "__main__":
    main()
