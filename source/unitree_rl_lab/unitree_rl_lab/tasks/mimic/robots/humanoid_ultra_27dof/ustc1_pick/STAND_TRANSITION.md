# Pick motion with standing transitions

`ustc1_pick_stand_transition.npz` is a new 50 Hz training reference. The
original `ustc1_pick.npz` remains unchanged for old-checkpoint provenance.

## Timeline

| Frames | Time | Segment |
| --- | ---: | --- |
| 0-99 | 2.00 s | walk-ready standing hold |
| 100-299 | 4.00 s | quintic stand-to-Pick transition |
| 299-658 | 7.20 s | original Pick motion |
| 659-858 | 4.00 s | quintic Pick-to-stand transition |
| 859-958 | 2.00 s | final standing hold |

Total: 959 frames, 19.18 seconds at 50 Hz. Frame 299 is shared by the end of
the prepare transition and the first frame of the original Pick clip.

## Regenerate

From the repository root:

```bash
conda run -n gmr python \
  scripts/mimic/add_stand_transition_humanoid_ultra.py \
  --input-npz \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_pick/ustc1_pick.npz \
  --output-csv /tmp/ustc1_pick_stand_transition.csv \
  --stand-hold-seconds 2 \
  --transition-seconds 4

conda run -n ustc_isaaclab python \
  scripts/mimic/csv_to_npz_humanoid_ultra.py \
  --input_file /tmp/ustc1_pick_stand_transition.csv \
  --input_fps 50 \
  --output_fps 50 \
  --output_name \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_pick/ustc1_pick_stand_transition.npz \
  --headless
```

The second command must run on a host where Isaac Sim can access a GPU. It
recomputes all 28 rigid-body poses and velocities from the current Humanoid
Ultra asset; do not directly interpolate the NPZ body arrays.

## Training boundary

The trajectory has materially changed, and the actor now uses a 144-dimensional
single-frame observation. Start a new training run; neither checkpoints trained
on `ustc1_pick.npz` nor the former 687-dimensional history checkpoints are
compatible.

```bash
conda activate ustc_isaaclab
cd /home/zxh/unitree_rl_lab
./unitree_rl_lab.sh -t \
  --task USTC-Humanoid-Ultra-27dof-Mimic-Pick \
  --run_name pick_stand_transition_v1
```

The task now points to `ustc1_pick_stand_transition.npz`; the default Mimic
runner trains for 30,000 iterations and saves every 500 iterations.
