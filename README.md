# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1**, **G1-29dof**, and the integrated
**Humanoid Ultra 12/27-DOF** reference robot.

Humanoid Ultra assets, task names, training commands, and the procedure for
adding another robot are documented in
[doc/humanoid_ultra_training.md](doc/humanoid_ultra_training.md).

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Set `UNITREE_MODEL_DIR` in your shell or conda activation script.

    ```bash
    export UNITREE_MODEL_DIR="/home/user/projects/unitree_model"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Set `UNITREE_ROS_DIR` in your shell or conda activation script.
    ```bash
    export UNITREE_ROS_DIR="/home/user/projects/unitree_ros"
    ```
  - To keep machine-specific paths out of Git, you can put them in your conda environment activation script:
    ```bash
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/unitree_paths.sh" <<'EOF'
    export UNITREE_ROS_DIR="/home/user/projects/unitree_ros"
    export UNITREE_MODEL_DIR="/home/user/projects/unitree_model"
    EOF
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Stand-v2 --headless
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Humanoid Ultra 27 自由度站立任务：

    ```bash
    conda activate ustc_isaaclab
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Stand
    ```
    同时跑两个task  
    ```bash
    source ~/.mps/env.sh
    pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d   # 没跑才启
    # 然后在这个 shell 里起训练；第二个终端重复同样两步
    ```
    使用辨识后 URDF 和 Mimic 关节 armature 的独立 Walk/Stand 任务：

    ```bash
    # Flat 即平地 Walk
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Identified-Flat
    # Flat-LeftArm2kg 使用左臂负载 2 kg 时重新辨识的手臂惯性参数
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm2kg
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm4kg
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Identified-Flat-LeftArm5kg
    # Stand 包含左臂周期激励轨迹跟踪
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Identified-Stand
    ```

    `Identified-Flat` 和 `Identified-Stand` 使用
    `humanoid_ultra_27dof_description_identified.urdf`；`Flat-LeftArm2kg` 使用
    `humanoid_ultra_27dof_description_identified_leftarm2kg.urdf`，`Flat-LeftArm4kg`和
    `Flat-LeftArm5kg` 分别使用对应的 `leftarm4kg`/`leftarm5kg` URDF。这些 identified task
    共享的 hip roll armature 为 0.15；Humanoid Ultra Mimic 的 hip roll actuator
    也直接读取这个共享值。
    hip yaw 为 0.01、hip pitch 为 0.10、knee 为 0.12，脚踝、腰部和手臂沿用 0.01。
    所有 `Flat-LeftArm*kg` 任务因负载破坏左右动力学对称性，
    单独关闭镜像数据增强和 mirror loss。Identified-Stand 继承
    `Stand-LeftArmTrack` 的左臂轨迹、15 维参考观测、跟踪奖励、安全渐入/渐出和
    手腕碰撞惩罚。日志分别写入
    `humanoidultra27dof_identified_flat`、
    `humanoidultra27dof_identified_flat_leftarm2kg`、
    `humanoidultra27dof_identified_flat_leftarm4kg`、
    `humanoidultra27dof_identified_flat_leftarm5kg` 和
    `humanoidultra27dof_identified_stand_leftarm`，不会混入原任务目录。

    2.5 kg 左臂负载的 Mimic 任务使用
    `humanoid_ultra_27dof_description_identified_leftarm2-5kg.urdf`：

    ```bash
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Mimic-Pick-2-5kg
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Mimic-Taitui-Right-2-5kg
    ./unitree_rl_lab.sh -t --task USTC-Humanoid-Ultra-27dof-Mimic-houtaitui-2-5kg
    ```

    原有 `USTC-Humanoid-Ultra-27dof-Mimic-Pick` 仍使用无负载 identified URDF。

    从已有站立策略继续进行抗扰动训练：

    ```bash
    ./unitree_rl_lab.sh -t \
      --task USTC-Humanoid-Ultra-27dof-Stand \
      --resume \
      --load_run 2026-06-13_22-58-22 \
      --checkpoint model_20000.pt \
      --max_iterations 15000 \
      --run_name robust_v2
    ```

    抗扰动训练会在前 10000 次迭代内逐步增强水平冲击和躯干角速度冲击，
    同时允许策略通过迈步和摆臂恢复平衡。原有 27 关节动作顺序保持不变。

    可视化训练结果：

    ```bash
    ./unitree_rl_lab.sh -p \
      --task USTC-Humanoid-Ultra-27dof-Stand \
      --checkpoint /absolute/path/to/model_XXXX.pt
    ```

    该任务保持 Humanoid Ultra 现有的 27 维动作、90 维单帧观测和 10 帧历史。
    所有腿部、腰部和手臂奖励均按关节名称解析；环境启动时还会校验 Isaac Lab
    实际关节顺序，防止策略动作与关节映射错位。

  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity

    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity --load_run "2026-04-30_17-41-04" --checkpoint "model_1500.pt"
    ```
  
  - 查看训练结果
  ```bash
    
  # 训练效果查看

  pip install tensorflow
  # 进入训练策略的文件夹,找到当前训练结果文件夹

    RL/unitree_rl_gym/logs/ustc1

  # 在终端输入:
    tensorboard --logdir=path  

  #同时查看历史训练情况

  tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_velocity    

  tensorboard --logdir logs文件夹地址  

  # 如果训练跑在 服务器 上，可以用端口转发：

    ssh -L 6006:localhost:6006 user@server
    tensorboard --logdir runs --port 6006


    # 然后在本地浏览器打开 http://localhost:6006。

    # 如果多个实验，可以用：

    tensorboard --logdir_spec run1:runs/exp1,run2:runs/exp2


      在同一个界面对比实验。


  ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```
安装完共享库后，你可能没有更新动态链接器缓存，所以系统运行时找不到libddsc.so.0
只需要运行一条命令修复
```
sudo ldconfig
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
# RB + Y → 进入 Stand_LeftArmTrack(站立策略,左臂初始收手在默认位姿)
# RB + A → 开/关左臂激励(平滑渐入/渐出,1.5s)
# Passive: LT + B.on_pressed
# Mimic_Houtaitui: LT(2s) + up.on_pressed
# Mimic_Pico_Houtaitui4: LT(2s) + left.on_pressed      改变质量脚不稳
# Mimic_Pico_Dun: LT(2s) + down.on_pressed    效果可以
# Mimic_Pico_Taitui: LT(2s) + X.on_pressed    效果可以 改变质量脚不稳
# Mimic_Pico_Houtaitui:         脚抬的不高
# Mimic_Jump1: LT(2s) + A.on_pressed : LT(2s) + A.on_pressed   跳得远的
# Mimic_Pico_Righttaitui: LT(2s) + right.on_pressed   改变质量效果可以
# Mimic_Dun: LT(2s) + LB.on_pressed           效果可以
# Mimic_Neutral_Walk_Forward: LT(2s) + Y.on_pressed
# Mimic_Jump:                  LT(2s) + RB.on_pressed
# Mimic_Banyun: LT(2s) + RT.on_pressed
# Mimic_Jump3: LT(2s) + start.on_pressed
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build

cd unitree_rl_lab/deploy/robots/g1_27dof/build

./g1_ctrl --network enp5s0 # eth0 is the network interface name.

# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
# Passive: LT + B.on_pressed
# Mimic_Houtaitui: LT(2s) + up.on_pressed
# Mimic_Pico_Houtaitui4: LT(2s) + left.on_pressed      改变质量脚不稳
# Mimic_Pico_Dun: LT(2s) + down.on_pressed    效果可以
# Mimic_Pico_Taitui: LT(2s) + X.on_pressed    效果可以 改变质量脚不稳
# Mimic_Pico_Houtaitui:         脚抬的不高
# Mimic_Jump1: LT(2s) + A.on_pressed : LT(2s) + A.on_pressed   跳得远的
# Mimic_Pico_Righttaitui: LT(2s) + right.on_pressed   改变质量效果可以
# Mimic_Dun: LT(2s) + LB.on_pressed           效果可以
# Mimic_Neutral_Walk_Forward: LT(2s) + Y.on_pressed
# Mimic_Jump:                  LT(2s) + RB.on_pressed
# Mimic_Banyun: LT(2s) + RT.on_pressed
# Mimic_Jump3: LT(2s) + start.on_pressed
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
