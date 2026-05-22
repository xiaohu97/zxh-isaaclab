# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

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
# Passive: LT + B.on_pressed
# Mimic_Dance_102: LT(2s) + up.on_pressed
# Mimic_Pico_Houtaitui4: LT(2s) + left.on_pressed
# Mimic_Pico_Dun: LT(2s) + down.on_pressed
# Mimic_Pico_Taitui: LT(2s) + X.on_pressed
# Mimic_Pico_Houtaitui: LT(2s) + A.on_pressed
# Mimic_Pico_Taitui2: LT(2s) + right.on_pressed
# Mimic_Dun: LT(2s) + LB.on_pressed
# Mimic_Neutral_Walk_Forward: LT(2s) + Y.on_pressed
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
# Mimic_Dance_102: LT(2s) + up.on_pressed
# Mimic_Pico_Houtaitui4: LT(2s) + left.on_pressed      改变质量脚不稳
# Mimic_Pico_Dun: LT(2s) + down.on_pressed    效果可以
# Mimic_Pico_Taitui: LT(2s) + X.on_pressed    效果可以 改变质量脚不稳
# Mimic_Pico_Houtaitui: LT(2s) + A.on_pressed 脚抬的不高
# Mimic_Pico_Taitui2: LT(2s) + right.on_pressed   改变质量效果可以
# Mimic_Dun: LT(2s) + LB.on_pressed           效果可以
# Mimic_Neutral_Walk_Forward: LT(2s) + Y.on_pressed
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
