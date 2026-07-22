# RightStand deployment-matched training

Use `USTC-Humanoid-Ultra-27dof-Mimic-RightStand-DeploySafe` for policies that
will be tested in the Humanoid Ultra MuJoCo or real-robot controller.

The task preserves the original RightStand motion and rewards, while changing
the action boundary to match deployment:

- position targets are clipped to the per-joint command limits in
  `sim2real_humanoidultra27dof_stand.py`;
- target changes are limited to `6 rad/s`, or `0.12 rad` per 50 Hz policy step;
- the policy and critic observe the normalized target that was actually sent
  after clipping and rate limiting as `last_action`.

The original `USTC-Humanoid-Ultra-27dof-Mimic-RightStand` task remains
unchanged so its saved runs remain reproducible.

## Recommended: train from scratch

```bash
conda activate ustc_isaaclab
cd /home/zxh/unitree_rl_lab

python scripts/rsl_rl/train.py \
  --task USTC-Humanoid-Ultra-27dof-Mimic-RightStand-DeploySafe \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name deploysafe_from_scratch
```

## Optional ablation: fine-tune model_19500

This is an ablation rather than the preferred policy because the old actor was
trained with unclipped, unlimited-rate targets and raw `last_action`.  The
checkpoint iteration is 19500, so 30500 additional iterations reaches 50000.

```bash
python scripts/rsl_rl/train.py \
  --task USTC-Humanoid-Ultra-27dof-Mimic-RightStand-DeploySafe \
  --experiment_name ustc_humanoid_ultra_27dof_mimic_rightstand \
  --resume \
  --load_run 2026-07-21_00-11-18_resume_from_11000 \
  --checkpoint model_19500.pt \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 30500 \
  --run_name deploysafe_finetune_from_19500
```

Do not deploy the old checkpoint merely because it still loads: under the new
action boundary it terminates repeatedly and must be retrained or fine-tuned.
