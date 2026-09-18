#!/usr/bin/env python3
"""在 dev 机(跑 g1_ctrl 的这台)上跑:收 Jetson 发来的 D435i 点云 + 订阅 rt/dog_odom,
建成躯干 yaw 系 17×11 高程图,按控制器约定发到 rt/perceptive/height_map。

为什么建图在这台而不在 Jetson:Jetson 无 DDS、无外网(cyclonedds 无 aarch64 轮子难装),
dev 机有 cyclonedds、在 DDS 网上、能连真机调试。Jetson 只做深度→光学系点云并 TCP 发出(哑节点)。

几何链(逐帧):
  光学系点(x右,y下,z前)
    → 相机 link 系(x前,y左,z上):   R_opt2link
    → 躯干系:                        R_mount(俯仰 47.6°) · p + t_mount   [外参,来自 URDF/perception_cfg]
    → 世界系:                        R_world(odom 四元数) · p + pos_world
    → odom 系累积(记忆身后/脚下看不到的格)
    → 采样躯干 yaw 系 17×11:         value = 点世界 z − 躯干世界 z ≈ 平地 −0.78

控制器侧(include/height_map_gate.h)拿到后:obs = −value − 0.5,再 clip[-1,1]。网格必须 17×11@0.1m,
origin(-0.8,-0.5),data[iy*17+ix],未知格 NaN。尺寸/分辨率/原点任一不符,控制器整张按平地处理。

★安全:没有本节点或它停了,控制器收不到图 → 按平地填充 → 感知策略摔倒率 21.5%(见 velocity_rough/README)。
所以先用 tools/height_map_echo.py 和 --print 确认图对,再考虑切 Velocity_Rough;之前一直用 Velocity 盲走。
"""
import argparse
import math
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dds_idl import HeightMap_, Odometry_  # noqa: E402


# --- 网格(必须与训练 HEIGHT_SCANNER_CFG / 控制器 config.yaml 一致) ---
GW, GH, RES = 17, 11, 0.1
ORIGIN = (-0.8, -0.5)

# --- 相机外参(torso_link 系,来自 g1_29dof URDF / perception_cfg.py) ---
MOUNT_POS = np.array([0.0576235, 0.01753, 0.42987])
MOUNT_PITCH = 0.8307767239493009  # rad,绕 +y 向下
# 光学系(x右,y下,z前) -> link 系(x前,y左,z上)
R_OPT2LINK = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class OdomState:
    def __init__(self):
        self.lock = threading.Lock()
        self.R = np.eye(3)
        self.pos = np.zeros(3)
        self.yaw = 0.0
        self.stamp = 0.0

    def update(self, od):
        q = od.pose.pose.orientation
        p = od.pose.pose.position
        R = quat_to_R(q.x, q.y, q.z, q.w)
        with self.lock:
            self.R = R
            self.pos = np.array([p.x, p.y, p.z])
            self.yaw = math.atan2(R[1, 0], R[0, 0])
            self.stamp = time.time()

    def get(self):
        with self.lock:
            return self.R.copy(), self.pos.copy(), self.yaw, self.stamp


class Accumulator:
    """odom 系滚动高程累积(记忆身后/脚下)。存每格世界 z 的 EWMA + 最后更新时间。"""

    def __init__(self, span=40.0, res=RES, alpha=0.5, timeout=10.0):
        self.n = int(span / res)
        self.res = res
        self.alpha = alpha
        self.timeout = timeout
        self.z = np.full((self.n, self.n), np.nan, np.float32)
        self.t = np.zeros((self.n, self.n), np.float32)
        self.org = None  # 世界系左下角,首帧 odom 位置 - span/2

    def _idx(self, wx, wy):
        gx = np.floor((wx - self.org[0]) / self.res).astype(np.int32)
        gy = np.floor((wy - self.org[1]) / self.res).astype(np.int32)
        return gx, gy

    def add(self, world_pts, pos, now):
        if self.org is None:
            self.org = np.array([pos[0] - self.n * self.res / 2, pos[1] - self.n * self.res / 2])
        gx, gy = self._idx(world_pts[:, 0], world_pts[:, 1])
        ok = (gx >= 0) & (gx < self.n) & (gy >= 0) & (gy < self.n)
        gx, gy, z = gx[ok], gy[ok], world_pts[ok, 2]
        if gx.size == 0:
            return
        # 每格取本帧中位高度(近似落脚支撑面),再对历史做 EWMA。
        # 不能取最小:跨台阶边沿的格里混有低处地面点,min 会把台阶蚀平、系统性低估台阶高度(危险);
        # 也不取最大:会被深度噪声尖点抬高。中位数抗噪又不蚀边,和训练用"射线打上表面"最接近。
        flat = gy * self.n + gx
        order = np.lexsort((z, flat))  # 先按格、再按 z 升序,使每格内 z 连续递增
        flat, z = flat[order], z[order]
        uniq, start, counts = np.unique(flat, return_index=True, return_counts=True)
        cellval = z[start + counts // 2]  # 每格中位(偶数取上中位)
        cy, cx = np.divmod(uniq, self.n)
        old = self.z[cy, cx]
        fresh = np.isnan(old)
        self.z[cy, cx] = np.where(fresh, cellval, (1 - self.alpha) * old + self.alpha * cellval)
        self.t[cy, cx] = now

    def sample(self, world_xy, now):
        if self.org is None:
            return np.full(world_xy.shape[0], np.nan, np.float32)
        gx, gy = self._idx(world_xy[:, 0], world_xy[:, 1])
        ok = (gx >= 0) & (gx < self.n) & (gy >= 0) & (gy < self.n)
        out = np.full(world_xy.shape[0], np.nan, np.float32)
        gxo, gyo = gx[ok], gy[ok]
        z = self.z[gyo, gxo]
        age = now - self.t[gyo, gxo]
        z = np.where(age <= self.timeout, z, np.nan)
        out[ok] = z
        return out


class Mapper:
    def __init__(self, args):
        self.a = args
        self.odom = OdomState()
        self.acc = Accumulator(span=args.acc_span, alpha=args.ewma, timeout=args.memory_s)
        self.R_mount = Ry(MOUNT_PITCH)
        # 躯干 yaw 系每个策略格中心的 (x,y)
        ix, iy = np.meshgrid(np.arange(GW), np.arange(GH))
        self.cell_x = (ORIGIN[0] + ix * RES).astype(np.float32).ravel()  # 前
        self.cell_y = (ORIGIN[1] + iy * RES).astype(np.float32).ravel()  # 左
        self.last_pub = np.full(GW * GH, np.nan, np.float32)
        self.frames = 0
        self.calib = []  # --calibrate-flat 收集可见格的世界 z - body z

        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.core import Qos, Policy
        dp = DomainParticipant(args.domain)
        be = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self.reader = DataReader(dp, Topic(dp, args.odom_topic, Odometry_, qos=be), qos=be)
        self.writer = DataWriter(dp, Topic(dp, args.out_topic, HeightMap_, qos=be), qos=be)
        threading.Thread(target=self._odom_loop, daemon=True).start()
        threading.Thread(target=self._pub_loop, daemon=True).start()

    def _odom_loop(self):
        while True:
            got = False
            for od in self.reader.take(N=10):
                self.odom.update(od)
                got = True
            time.sleep(0.002 if got else 0.01)

    def process(self, opt_pts, stamp_ns):
        """opt_pts: (N,3) 光学系;更新累积。"""
        R, pos, yaw, ostamp = self.odom.get()
        now = time.time()
        if now - ostamp > self.a.odom_timeout:
            return  # 没有新鲜 odom,不敢建图(位姿未知)
        p_link = opt_pts @ R_OPT2LINK.T
        p_torso = p_link @ self.R_mount.T + MOUNT_POS
        world = p_torso @ R.T + pos
        self.acc.add(world, pos, now)
        self.frames += 1
        if self.a.calibrate_flat:
            # 可见格里点相对 body 的高度(平地应处处相等 = -body高度)
            self.calib.append(world[:, 2] - pos[2])

    def _sample_window(self):
        R, pos, yaw, ostamp = self.odom.get()
        now = time.time()
        if now - ostamp > self.a.odom_timeout:
            return None
        c, s = math.cos(yaw), math.sin(yaw)
        # 格中心(躯干 yaw 系) -> 世界
        wx = pos[0] + c * self.cell_x - s * self.cell_y
        wy = pos[1] + s * self.cell_x + c * self.cell_y
        world_z = self.acc.sample(np.stack([wx, wy], 1), now)
        torso_z = pos[2] + self.a.torso_z_over_body
        value = world_z - torso_z  # 地面 - 躯干,平地 ≈ -0.78
        return value.astype(np.float32)

    def _publish(self, value):
        data = value.copy()
        data[~np.isfinite(data)] = float("nan")  # 未知 -> NaN,控制器按平地填该格
        msg = HeightMap_(stamp=time.time(), frame_id="torso_yaw", resolution=float(RES),
                         width=GW, height=GH, origin=[float(ORIGIN[0]), float(ORIGIN[1])],
                         data=data.astype(np.float32).tolist())
        self.writer.write(msg)
        self.last_pub = value

    def _pub_loop(self):
        period = 1.0 / self.a.rate
        nxt = time.time()
        while True:
            v = self._sample_window()
            if v is not None:
                self._publish(v)
            nxt += period
            time.sleep(max(0, nxt - time.time()))

    def print_map(self):
        v = self.last_pub.reshape(GH, GW)
        known = np.isfinite(v)
        print("\n=== height map (行=y左→右, 列=x后→前; 值=地面-躯干 m) 已知格 %d/%d ===" % (known.sum(), GW * GH))
        for iy in range(GH - 1, -1, -1):
            row = "".join(("%+5.2f " % v[iy, ix]) if known[iy, ix] else "  .   " for ix in range(GW))
            print(row)
        R, pos, yaw, st = self.odom.get()
        print("odom pos (%.2f,%.2f,%.2f) yaw %.1f° 累积帧 %d" % (*pos, math.degrees(yaw), self.frames))

    def report_calib(self):
        if not self.calib:
            print("没收到可见点,无法标定"); return
        allz = np.concatenate(self.calib)
        allz = allz[np.isfinite(allz)]
        print("\n=== 平地标定(机器人须站在已知平地上) ===")
        print("可见点相对 body 高度: 中位 %.3f  p10 %.3f  p90 %.3f  (n=%d)" % (
            np.median(allz), np.percentile(allz, 10), np.percentile(allz, 90), allz.size))
        print("→ 把 --torso-z-over-body 设为 %.3f,使平地 value = -0.78" % (np.median(allz) + 0.78))
        print("  (若 p10/p90 相差 > 0.05,可能是外参俯仰有偏,前后格高度不一致)")


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def serve(mapper, args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print("waiting for Jetson depth sender on :%d ..." % args.port)
    last_print = time.time()
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("depth sender connected from %s" % (addr,))
        try:
            while True:
                hdr = recv_exact(conn, 4)
                if hdr is None:
                    break
                (plen,) = struct.unpack("<I", hdr)
                payload = recv_exact(conn, plen)
                if payload is None:
                    break
                stamp_ns, n = struct.unpack_from("<QI", payload, 0)
                pts = np.frombuffer(payload, dtype="<f4", offset=12, count=n * 3).reshape(n, 3)
                mapper.process(pts, stamp_ns)
                if args.print and time.time() - last_print >= 1.0:
                    mapper.print_map()
                    last_print = time.time()
        except (OSError, struct.error) as e:
            print("connection error: %s" % e)
        finally:
            conn.close()
            print("depth sender disconnected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="enp5s0", help="连机器人的网卡(写进 CYCLONEDDS_URI)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--port", type=int, default=5601, help="接收 Jetson 深度的 TCP 端口")
    ap.add_argument("--odom-topic", default="rt/dog_odom")
    ap.add_argument("--out-topic", default="rt/perceptive/height_map")
    ap.add_argument("--rate", type=float, default=50.0, help="高程图发布频率 Hz")
    ap.add_argument("--memory-s", type=float, default=10.0, help="累积格超过此时长未更新 -> 视为未知")
    ap.add_argument("--ewma", type=float, default=0.5, help="每格世界 z 的 EWMA 系数")
    ap.add_argument("--acc-span", type=float, default=40.0, help="odom 系累积网格边长 m")
    ap.add_argument("--torso-z-over-body", type=float, default=0.111,
                    help="躯干 z 相对 odom body 原点的偏置;先用 --calibrate-flat 标定")
    ap.add_argument("--odom-timeout", type=float, default=0.5, help="odom 超时不建图")
    ap.add_argument("--print", action="store_true", help="每秒打印 ASCII 高程图")
    ap.add_argument("--calibrate-flat", action="store_true", help="站平地上收集数据并给出 torso-z 偏置,不必长跑")
    a = ap.parse_args()

    if a.iface and "CYCLONEDDS_URI" not in os.environ:
        xml = "/tmp/cdds_mapper_%d.xml" % os.getpid()
        open(xml, "w").write(
            "<CycloneDDS><Domain><General><Interfaces>"
            "<NetworkInterface name=\"%s\"/></Interfaces></General></Domain></CycloneDDS>" % a.iface)
        os.environ["CYCLONEDDS_URI"] = "file://" + xml

    mapper = Mapper(a)
    print("mapper up: odom<-%s  out->%s  rate %.0fHz  torso_z_over_body %.3f" % (
        a.odom_topic, a.out_topic, a.rate, a.torso_z_over_body))
    try:
        serve(mapper, a)
    except KeyboardInterrupt:
        if a.calibrate_flat:
            mapper.report_calib()


if __name__ == "__main__":
    main()
