"""感知建图链路要用的 DDS 消息类型(内联声明,不依赖 unitree_sdk2py)。

字段与线上完全一致:
* HeightMap_  == unitree_go::msg::dds_::HeightMap_(控制器 include/dds_height_map_source.h 订阅的类型)
* Odometry_ / 依赖的 std_msgs/geometry_msgs == 机器人 rt/dog_odom 发布的 nav_msgs::msg::dds_::Odometry_

只用 cyclonedds,dev 机已装。名字里的 typename 必须和对端一字不差,否则 DDS 类型不匹配收不到/发不出。
"""
from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as t


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HeightMap_(idl.IdlStruct, typename="unitree_go.msg.dds_.HeightMap_"):
    stamp: t.float64
    frame_id: str
    resolution: t.float32
    width: t.uint32
    height: t.uint32
    origin: t.array[t.float32, 2]
    data: t.sequence[t.float32]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Time_(idl.IdlStruct, typename="builtin_interfaces.msg.dds_.Time_"):
    sec: t.int32
    nanosec: t.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Header_(idl.IdlStruct, typename="std_msgs.msg.dds_.Header_"):
    stamp: Time_
    frame_id: str


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Point_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Point_"):
    x: t.float64
    y: t.float64
    z: t.float64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Quaternion_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Quaternion_"):
    x: t.float64
    y: t.float64
    z: t.float64
    w: t.float64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Pose_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Pose_"):
    position: Point_
    orientation: Quaternion_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class PoseWithCovariance_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.PoseWithCovariance_"):
    pose: Pose_
    covariance: t.array[t.float64, 36]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Vector3_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Vector3_"):
    x: t.float64
    y: t.float64
    z: t.float64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Twist_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Twist_"):
    linear: Vector3_
    angular: Vector3_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class TwistWithCovariance_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.TwistWithCovariance_"):
    twist: Twist_
    covariance: t.array[t.float64, 36]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Odometry_(idl.IdlStruct, typename="nav_msgs.msg.dds_.Odometry_"):
    header: Header_
    child_frame_id: str
    pose: PoseWithCovariance_
    twist: TwistWithCovariance_
