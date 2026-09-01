# Humanoid Ultra dance mimic tasks

Eight motion-tracking tasks built from the hand-picked BONES-SEED dance clips in
`GR00T-WholeBodyControl/data/挑选舞蹈动作/`.  Each task is a thin override of the
Pick base config (`ustc1_pick/tracking_env_cfg.py`); only the motion file changes.

| Task dir | Gym id suffix | Source clip | Ref. length |
| --- | --- | --- | ---: |
| `dance_padeburee_270` | `Dance-Padeburee-270` | `dance_basic_padeburee_270_R_loop_001__A321` | 21.06 s |
| `dance_slide_360` | `Dance-Slide-360` | `dance_basic_slide_360_R_loop_002__A322` | 11.84 s |
| `dance_turn_180` | `Dance-Turn-180` | `dance_basic_turn_v1_180_R_loop_001__A321` | 17.40 s |
| `dance_barbie` | `Dance-Barbie` | `dance_hiphop_barbie_R_loop_001__A321` | 15.24 s |
| `dance_bounce_to_feet` | `Dance-Bounce-To-Feet` | `dance_hiphop_bounce_to_feet_R_loop_002__A324` | 15.48 s |
| `dance_mambo` | `Dance-Mambo` | `dance_latino_mambo_180_mambo_360_R_001__A321` | 16.14 s |
| `dance_macarena` | `Dance-Macarena` | `victory_dance_macarena_R_001__A323` | 17.08 s |
| `dance_macarena_turn` | `Dance-Macarena-Turn` | `victory_dance_macarena_change_direction_R_001__A321` | 20.84 s |

Full gym id is `USTC-Humanoid-Ultra-27dof-Mimic-<suffix>`.  All references are
50 Hz and fit inside the inherited `episode_length_s = 30.0`.

## Reference layout

Every NPZ is `<clip> = 1 s quintic default-pose blend -> source clip ->
1 s quintic blend back -> 2 s terminal standing hold`, so each reference starts
and ends exactly on the 27-DoF Mimic default standing pose.  Verified: frame 0
and the final frame match `ustc1_walk_stand_transition.npz` frame 0 to 2e-3 rad.

The source clips are already loop-cut and begin/end near standing (root z within
1.00-1.02 m, max non-wrist joint gap to the default pose <= 0.9 rad), which is
why a 1 s blend is enough.

## Config differences from the other Humanoid Ultra mimic tasks

`Walk`, `Wave` and `Spin` set `motion_arm_joint_pos = None`.  The dance tasks
**keep** the base arm joint-position reward (weight 0.75): the arm choreography
is the content of these clips, not incidental.  Everything else — observations,
actions, terminations, events, anchor body, tracked bodies — is inherited
unchanged from Pick.

## Regenerate

Two steps per clip, from the repo root.  Step 1 is pure NumPy; step 2 needs a
GPU host because it recomputes all 28 rigid-body poses and velocities from the
current Humanoid Ultra asset.  Do not interpolate the NPZ body arrays directly.

```bash
SRC=/home/ustczxh/humanoid/GR00T-WholeBodyControl/data/挑选舞蹈动作
DST=source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/humanoid_ultra_27dof

conda run -n ustc_isaaclab python \
  scripts/mimic/add_default_pose_transitions_humanoid_ultra.py \
  --input-csv "$SRC/victory_dance_macarena_R_001__A323.csv" \
  --output-csv /tmp/dance_macarena_stand_transition.csv \
  --input-fps 120 --output-fps 50 \
  --transition-seconds 1.0 --terminal-hold-seconds 2.0

conda run -n ustc_isaaclab python \
  scripts/mimic/csv_to_npz_humanoid_ultra.py \
  -f /tmp/dance_macarena_stand_transition.csv \
  --input_fps 120 --output_fps 50 \
  --output_name $DST/dance_macarena/dance_macarena_stand_transition.npz \
  --headless
```

`--input-fps 120` is the BONES-SEED native rate.  Frame counts are preserved
1:1 by the soma retargeter, confirmed against `move_duration_frames` in
`data/seed_dance/seed_metadata_dance.csv`.

`csv_to_npz_humanoid_ultra.py` hangs in `simulation_app.close()` after writing
the NPZ.  The file is complete once its size stops changing; kill the process.

## Clips deliberately excluded

`DeploymentLimitedJointPositionActionCfg` clips shoulder yaw to +-2.5 rad.
Three of the eleven picked clips ask for more than that, so no policy could ever
reach those frames and the arm tracking reward would saturate there.  They were
built, measured, and then dropped:

| Source clip | Frames outside the clip range | Worst overshoot |
| --- | ---: | --- |
| `dance_latino_latino_sequence_R_002__A324` | 6.1 % | both shoulder yaws, +0.90 rad (52 deg) |
| `victory_dance_asarahe_180_R_001__A321` | 5.3 % | right shoulder yaw, +0.32 rad |
| `victory_dance_brick_brick_slide_slide_R_001__A324` | 1.3 % | right shoulder yaw, +0.16 rad |

Their CSVs are still in `GR00T-WholeBodyControl/data/挑选舞蹈动作/`; rebuild them
with the commands above if the deployment shoulder-yaw limit is ever widened.

The eight shipped clips are entirely inside the deployment limits.
