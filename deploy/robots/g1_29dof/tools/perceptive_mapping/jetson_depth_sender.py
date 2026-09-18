#!/usr/bin/env python3
"""在 Jetson(192.168.123.164)上跑:ROS1 Noetic 取 D435i 深度,投影降采样成点云,TCP 发给 dev 机。

这台没有 DDS、没有外网,所以只用 rospy + numpy + 标准库 socket(都已具备),不装任何东西。
本节点是"哑"的:只做深度→相机光学系 3D 点的投影和降采样,所有位姿相关的几何(调平、yaw 对齐、
odom 累积、采样成 17×11)都在 dev 机的 height_map_mapper.py 里,那样可调参数集中在一处、且在能连真机调试的机器上。

先启动相机(本目录 run_camera.sh 或):
  source /opt/ros/noetic/setup.bash
  roslaunch realsense2_camera rs_camera.launch enable_color:=false enable_infra1:=false \
      enable_infra2:=false enable_gyro:=false enable_accel:=false \
      depth_width:=640 depth_height:=480 depth_fps:=30
再跑本节点:
  python3 jetson_depth_sender.py --dev-ip 192.168.123.222 --port 5601

线格式(小端,TCP 流):每帧 = uint32 payload 长度 + payload;payload = uint64 stamp_ns + uint32 n + n×3×float32(x,y,z 米,深度光学系)。
"""
import argparse
import socket
import struct
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo


class DepthSender:
    def __init__(self, dev_ip, port, stride, zmin, zmax):
        self.dev = (dev_ip, port)
        self.stride, self.zmin, self.zmax = stride, zmin, zmax
        self.K = None
        self.sock = None
        self.sent = 0
        self.last_log = rospy.Time.now()
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self._info, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self._depth, queue_size=1, buff_size=2 ** 24)

    def _info(self, m):
        if self.K is None:
            self.K = (m.K[0], m.K[4], m.K[2], m.K[5])  # fx, fy, cx, cy
            rospy.loginfo("depth intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f (%dx%d)", *self.K, m.width, m.height)

    def _connect(self):
        try:
            s = socket.create_connection(self.dev, timeout=2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            rospy.loginfo("connected to dev %s:%d", *self.dev)
        except OSError as e:
            self.sock = None
            rospy.logwarn_throttle(5.0, "dev not reachable (%s), retrying" % e)

    def _depth(self, m):
        if self.K is None:
            return
        if m.encoding not in ("16UC1", "mono16"):
            rospy.logwarn_throttle(5.0, "unexpected depth encoding %s" % m.encoding)
        fx, fy, cx, cy = self.K
        d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
        d = d[:: self.stride, :: self.stride].astype(np.float32) * 0.001  # mm -> m
        h, w = d.shape
        vs, us = np.nonzero((d > self.zmin) & (d < self.zmax))
        if vs.size == 0:
            return
        z = d[vs, us]
        u = us.astype(np.float32) * self.stride
        v = vs.astype(np.float32) * self.stride
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts = np.stack([x, y, z], axis=1).astype("<f4")
        stamp = m.header.stamp.to_nsec() if m.header.stamp else rospy.Time.now().to_nsec()
        payload = struct.pack("<QI", int(stamp), pts.shape[0]) + pts.tobytes()
        if self.sock is None:
            self._connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall(struct.pack("<I", len(payload)) + payload)
            self.sent += 1
        except OSError as e:
            rospy.logwarn("send failed (%s), will reconnect" % e)
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            return
        now = rospy.Time.now()
        if (now - self.last_log).to_sec() >= 2.0:
            rospy.loginfo("sent %d frames, last %d pts" % (self.sent, pts.shape[0]))
            self.last_log = now


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-ip", default="192.168.123.222")
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--stride", type=int, default=4, help="像素降采样步长(4 -> 160x120)")
    ap.add_argument("--zmin", type=float, default=0.2)
    ap.add_argument("--zmax", type=float, default=4.0, help="超过此距离的深度丢弃(远场噪声大)")
    a = ap.parse_args(rospy.myargv()[1:])
    rospy.init_node("jetson_depth_sender", anonymous=True)
    DepthSender(a.dev_ip, a.port, a.stride, a.zmin, a.zmax)
    rospy.loginfo("depth sender up: -> %s:%d stride=%d z=[%.1f,%.1f]", a.dev_ip, a.port, a.stride, a.zmin, a.zmax)
    rospy.spin()


if __name__ == "__main__":
    main()
