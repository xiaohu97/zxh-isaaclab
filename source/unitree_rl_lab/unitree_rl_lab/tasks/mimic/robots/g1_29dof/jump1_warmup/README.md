# G1 29DoF Jump1 Warmup

This is the first-stage warmup task for `Unitree-G1-29dof-Mimic-Jump1`.

Task id:

```bash
Unitree-G1-29dof-Mimic-Jump1-Warmup
```

Motion file:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1/jump1.npz
```

It follows the same warmup settings as `jump_warmup`:

```text
static_friction_range: 0.8 - 1.2
dynamic_friction_range: 0.8 - 1.2
restitution_range: 0.0 - 0.1
add_joint_default_pos: None
base_com: None
push_robot: None
anchor_pos threshold: 0.8
anchor_ori threshold: 1.5
ee_body_pos threshold: 1.2
ee_body_pos bodies: left/right ankle only
action_rate_l2 weight: -5e-2
```

Train:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Jump1-Warmup
```
