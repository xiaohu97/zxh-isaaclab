# Phase 1: Lift Learning Curriculum

## Objective

The previous `model_14500` converged to a stable stand-only policy.  Phase 1
uses the original 615-frame, 12.30-second `ustc1_rightstand.npz` clip so the
long lifted-leg hold is represented directly:

- reference left-foot height above 0.30 m: frames 178-505 (6.56 s);
- peak reference height: 0.542 m at frame 204;
- a foot that stays on the floor can be about 0.487 m below the reference.

The deployment-oriented stand-transition clip is deliberately reserved for
Phase 2 fine-tuning.

## Phase 1 configuration

### Sampling

```python
frame_zero_probability = 0.20
targeted_frame_range = (130, 220)
targeted_frame_probability = 0.40
```

Targeted resets cover take-off, not only the already-lifted pose.  A targeted
reset writes the robot into the sampled reference state, so sampling only
frames 178-505 would mostly teach high-leg balance rather than lifting from the
floor.  The remaining 40% of resets retain failure-adaptive sampling.

### Rewards

```text
swing_foot_clearance       weight +2.0
swing_foot_contact_penalty weight -2.0
feet_impact_velocity       weight -0.2
```

`swing_foot_clearance` is a dense linear height-tracking score during the
reference swing phase.  It remains informative when the original exponential
ankle reward is saturated near zero.  It does not depend on the contact flag,
so merely unloading the foot cannot unlock the reward.

`swing_foot_contact_penalty` separately penalizes contact above 10 N while the
reference foot is above 0.30 m.  Landing impact remains weak in Phase 1 so an
initially clumsy landing does not make the policy abandon lifting.

### Terminations

The generic feet-and-hands height-error threshold remains 0.55 m.  A separate
one-sided termination applies only to the left swing foot:

```text
reference left-foot height > 0.30 m
and reference_z - actual_z > 0.25 m
```

This removes stand-only episodes without terminating on hand error, support
foot error, or harmless over-lifting.

## Training commands

Run from scratch for a clean comparison with the failed stand-only runs.

```bash
# houtaitui (no action EMA)
python scripts/rsl_rl/train.py \
  --task USTC-Humanoid-Ultra-27dof-Mimic-houtaitui \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 15000 --seed 42 --run_name phase1_lift_v2

# houtaituiEMA (action EMA alpha=0.85)
python scripts/rsl_rl/train.py \
  --task USTC-Humanoid-Ultra-27dof-Mimic-houtaituiEMA \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 15000 --seed 42 --run_name phase1_lift_ema_v2
```

## Monitoring and Phase 2 gate

Use the exact TensorBoard tags:

- `Episode_Reward/swing_foot_clearance`: should rise;
- `Episode_Reward/swing_foot_contact_penalty`: should approach zero;
- `Episode_Reward/motion_left_ankle_pos`: should rise;
- `Episode_Termination/swing_foot_height`: should fall after the early phase;
- `Episode_Termination/time_out`: should recover above 85%;
- `Episode_Reward/feet_impact_velocity`: record now, optimize in Phase 2.

Do not use a fixed clearance-reward threshold without calibration: Isaac Lab
logs the weighted episodic sum divided by the 30-second maximum episode length.
Enter Phase 2 only after the trends are stable for about 2,000 iterations and a
fixed frame-zero rollout confirms that the actual left foot reaches at least
0.45 m without support loss.

## Phase 2 outline

Phase 2 switches the command back to `StandTransitionCommandsCfg`, resumes the
best Phase 1 checkpoint, and gradually changes:

```text
swing_foot_clearance:       +2.0 -> +0.5
swing_foot_contact_penalty: -2.0 -> -0.5
feet_impact_velocity:       -0.2 -> -1.0
```

Validate the current stand-transition artifact before Phase 2.  The checked-in
file currently contains 941 frames and has reference height above 0.30 m at
frames 468-525; older documentation describing 1,214 frames is stale.
