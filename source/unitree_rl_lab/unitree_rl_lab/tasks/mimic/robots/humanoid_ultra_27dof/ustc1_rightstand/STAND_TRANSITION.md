# RightStand motion with standing transitions

`ustc1_rightstand_stand_transition.npz` is the deployment-oriented 50 Hz
reference.  The original `ustc1_rightstand.npz` remains unchanged for old-run
provenance.

## Timeline

| Frames | Time | Segment |
| --- | ---: | --- |
| 0-99 | 2.00 s | walk-ready standing hold |
| 100-299 | 4.00 s | quintic stand-to-RightStand transition |
| 299-913 | 12.30 s | original RightStand motion |
| 914-1113 | 4.00 s | quintic RightStand-to-stand recovery |
| 1114-1213 | 2.00 s | final standing hold |

Total: 1214 frames, 24.28 seconds at 50 Hz. Frame 299 is shared by the end of
the prepare transition and the first frame of the original RightStand clip.

## Regenerate

From the repository root:

```bash
conda run -n gmr python \
  scripts/mimic/add_stand_transition_humanoid_ultra.py \
  --input-npz \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand.npz \
  --output-csv /tmp/ustc1_rightstand_stand_transition.csv \
  --stand-hold-seconds 2 \
  --transition-seconds 4

conda run -n ustc_isaaclab python \
  scripts/mimic/csv_to_npz_humanoid_ultra.py \
  --input_file /tmp/ustc1_rightstand_stand_transition.csv \
  --input_fps 50 \
  --output_fps 50 \
  --output_name \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_rightstand/ustc1_rightstand_stand_transition.npz \
  --headless
```

The converter recomputes all 28 rigid-body poses and velocities through the
current Isaac Lab Humanoid Ultra asset. Do not interpolate the NPZ body fields
independently.

## Training boundary

The reference trajectory has materially changed. Train from scratch with the
new task; do not resume `model_14500.pt` or another checkpoint trained against
`ustc1_rightstand.npz`.

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

Pair the exported policy with `ustc1_rightstand_stand_transition.npz` in
sim2sim and sim2real.
