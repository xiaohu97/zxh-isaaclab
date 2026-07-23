# Humanoid Ultra houtaitui deployment-matched training

The only registered task for this motion family is:

```text
USTC-Humanoid-Ultra-27dof-Mimic-houtaitui
```

It combines both deployment requirements:

- `ustc1_rightstand_stand_transition.npz`: standing hold, smooth entry,
  original action, smooth recovery, and final standing hold;
- per-joint real-controller position limits;
- a `6 rad/s` position-target slew limit (`0.12 rad` per 50 Hz policy step);
- policy and critic `last_action` observations taken after clipping and slew
  limiting.

The reward keeps the all-joint torque penalty at `-1e-5` and adds separately
logged penalties for both hip-yaw joints (`-5e-5`) and both knee joints
(`-2e-5`). Their total effective torque weights, including the all-joint term,
are therefore `-6e-5` for hip yaw and `-3e-5` for knees.

The legacy RightStand task registrations were removed. Their source NPZ and
saved logs remain available only for provenance and motion regeneration.

## Train from scratch

The trajectory and action boundary differ from the old runs. Do not resume an
old RightStand or DeploySafe checkpoint.

```bash
conda activate ustc_isaaclab
cd /home/zxh/unitree_rl_lab

python scripts/rsl_rl/train.py \
  --task USTC-Humanoid-Ultra-27dof-Mimic-houtaitui \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name houtaitui_stand_transition_v1
```

See `STAND_TRANSITION.md` for the motion timeline, regeneration procedure, and
asset validation boundary.
