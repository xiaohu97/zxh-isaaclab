# Taitui-Right targeted fine-tuning

This task resumes the original Taitui-Right `model_29500.pt` without changing
the first 538 reference frames.

- Frames 0-537: bit-identical original reference.
- Frames 538-662: 2.5 seconds of the final standing pose at 50 Hz.
- Reset mixture: 25% exact frame 0, 50% uniform over frames 280-345,
  25% original failure-adaptive sampling.
- Pushes: original random 1-3 second full-range velocity kicks plus one
  optional heading-frame backward-right kick in frames 280-345 (`p=0.5` per
  episode; local `vx=-0.22..-0.10`, `vy=-0.28..-0.14 m/s`).
- Added reward: right ankle relative 3-D position, weight 4.0, `std=0.08`.
- Arm joint-position tracking remains disabled.

Checkpoint selection must use multiple seeds and at least 100 parallel
`push_noise` frame-0 rollouts per seed.  Do not rank checkpoints only by PPO
mean reward.
