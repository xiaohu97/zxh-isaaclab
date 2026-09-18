#!/bin/bash
# 在 Jetson(192.168.123.164)上启动 D435i 深度流(只开深度,省 USB 带宽)。
# 用法: bash run_camera.sh
set -e
source /opt/ros/noetic/setup.bash
echo "启动 realsense2_camera(仅深度 640x480@30)..."
exec roslaunch realsense2_camera rs_camera.launch \
    enable_color:=false enable_infra1:=false enable_infra2:=false \
    enable_gyro:=false enable_accel:=false \
    depth_width:=640 depth_height:=480 depth_fps:=30 \
    initial_reset:=true
