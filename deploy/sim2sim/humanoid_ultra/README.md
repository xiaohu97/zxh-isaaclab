# Humanoid Ultra MuJoCo Sim2Sim

This standalone runner tests the 12-DOF and 27-DOF Humanoid Ultra locomotion
policies trained in `unitree_rl_lab`. It does not use the Unitree DDS bridge.

## 1. Export the Isaac Lab checkpoint

From `/home/zxh/unitree_rl_lab`:

```bash
conda activate ustc_isaaclab

./unitree_rl_lab.sh -p \
  --task USTC-Humanoid-Ultra-12dof-Flat \
  --checkpoint "/home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/model_1000.pt" \
  --headless \
  --export_only
```

The exported model is written to:

```text
logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/exported/policy.pt
```

Use the matching task and experiment directory for a 27-DOF policy.

## 2. Run MuJoCo

The local `gmr` conda environment already contains both `torch` and `mujoco`:

```bash
cd /home/zxh/ustc_humanoid/unitree_mujoco
conda activate gmr

python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 12 \
  --policy /home/zxh/unitree_rl_lab/logs/rsl_rl/humanoidultra12dof_flat/RUN_DIRECTORY/exported/policy.pt \
  --vx 0.3
```

For 27 DOF:

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 27 \
  --policy /absolute/path/to/exported/policy.pt \
  --vx 0.3
```

Keyboard controls:

```text
W/S       increase/decrease forward velocity
A/D       increase/decrease lateral velocity
Q/E       increase/decrease yaw rate
X/Space   zero all commands
R         reset the robot
```

Headless validation:

```bash
python sim2sim/humanoid_ultra/sim2sim.py \
  --dof 12 \
  --policy /absolute/path/to/exported/policy.pt \
  --vx 0.3 \
  --headless \
  --duration 10
```

The runner reproduces the Isaac Lab policy interface:

- physics time step: `0.005 s`
- policy time step: `0.02 s`
- action scale: `0.25`
- actor history: 10 frames
- 12-DOF policy input/output: 450 / 12
- 27-DOF policy input/output: 900 / 27

Use a policy exported from the matching DOF task. A raw RSL-RL checkpoint such
as `model_1000.pt` cannot be passed directly to this runner.
