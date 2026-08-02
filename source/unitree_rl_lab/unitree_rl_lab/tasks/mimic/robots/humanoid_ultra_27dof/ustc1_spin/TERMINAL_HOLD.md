# Spin terminal standing hold

`ustc1_spin_stand_transition_hold_2p5s.npz` extends the original
`ustc1_spin_stand_transition.npz` without changing its existing trajectory.

## Timeline

| Frames | Duration | Segment |
| --- | ---: | --- |
| 0-305 | 6.12 s | Existing Spin reference, unchanged |
| 306-430 | 2.50 s | Constant terminal/default standing pose |

The new reference contains 431 samples at 50 Hz.  Its first 306 samples are
bit-for-bit identical to the old reference.  The 125 appended pose samples
repeat frame 305, and all appended joint and rigid-body velocities are zero.

## Regenerate

From the repository root:

```bash
conda run -n ustc_isaaclab python \
  scripts/mimic/append_terminal_hold_humanoid_ultra.py \
  --input-npz \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_spin/ustc1_spin_stand_transition.npz \
  --output-npz \
  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof/ustc1_spin/ustc1_spin_stand_transition_hold_2p5s.npz \
  --hold-seconds 2.5
```

The utility rejects identical input/output paths and refuses to replace an
existing output unless `--overwrite-output` is explicitly supplied.

For future regeneration from the original SOMA CSV,
`add_default_pose_transitions_humanoid_ultra.py` also accepts
`--terminal-hold-seconds 2.5` before conversion to NPZ.

## Fine-tuning boundary

The actor/critic architecture and the first 306 reference samples are
unchanged, so loading the current Spin checkpoint is intentional.  Start a new
log run so the old training history and reference provenance remain separate.
The appended segment must be evaluated independently: completion of frame 305
does not demonstrate that the robot can remain standing through frame 430.

For the targeted fine-tuning run, `frame_zero_probability=0.25` forces one
quarter of episode resets to begin at exact frame 0.  The remaining resets keep
the existing adaptive random-phase sampler.  This addresses the measured
full-clip failure around frames 107-115 without discarding phase coverage or
the newly appended terminal hold.
