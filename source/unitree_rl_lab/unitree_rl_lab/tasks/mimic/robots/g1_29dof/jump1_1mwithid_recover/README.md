# jump1_1mwithid_recover：给 Jump3 补落地站稳尾段

目的：让 jump3 参考在 0.8 s 内从落地回弹收到 walk 的默认站姿并保持 1 s，策略微调后能自己站稳，
再交给 walk。背景和改动见 `tracking_env_cfg.py` 顶部说明。

## 生成参考（已生成，改参数时重跑）

```bash
cd /home/ustczxh/humanoid/zxh-isaaclab
# 1. 追加尾段 -> CSV（需要 mujoco：用 ustc_identification 环境）
/home/ustczxh/miniconda3/envs/ustc_identification/bin/python scripts/mimic/add_recovery_tail_g1.py \
    --input-npz  source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid/jump1_1m_waist15.npz \
    --output-csv source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid_recover/jump1_1m_waist15_recover.csv \
    --summary-json source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid_recover/jump1_1m_waist15_recover.summary.json \
    --overwrite
# 2. 用 Isaac Lab 重算全部刚体位姿/速度 -> NPZ（ustc_isaaclab 环境，需 UNITREE_ROS_DIR）
conda activate ustc_isaaclab
python scripts/mimic/csv_to_npz.py \
    -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid_recover/jump1_1m_waist15_recover.csv \
    --input_fps 50 --output_fps 50 --headless --no_ground
#    本机 Isaac 无头启动到 spawn 完机器人约 7 min；打印 "Motion npz file saved" 后 app.close() 会卡住，直接 kill 即可。
# 3. 前 89 帧拷回原参考（位姿逐位一致，但 csv_to_npz 给的是 PhysX 解析速度，原参考上肢是位置差分，
#    起跳段差到 0.95 m/s；热启动不该换掉 checkpoint 学过的速度目标）
/home/ustczxh/miniconda3/envs/ustc_identification/bin/python scripts/mimic/splice_npz_prefix_g1.py \
    --reference source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid/jump1_1m_waist15.npz \
    --target    source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid_recover/jump1_1m_waist15_recover.npz \
    --frames 89
```

已生成的 NPZ 检查结果：180 帧 @50 fps、32 body；前 89 帧与原参考所有字段逐位相同；尾段脚底最低 0.0333 m
（=站立时 ankle_roll 离地高）；末帧根部 z 0.7825、倾角 0°、所有速度 0；尾段关节速度峰值 3.4 rad/s（起跳段 17.9）。

`summary.json` 里要看的数：`expected_npz_frames_after_csv_to_npz`（cfg 读它定 frame 范围）、
`max_ref_foot_drift_m`（左右腿反推根部不一致 = 参考脚位漂移，目前 3.8 cm）、`min_foot_z_in_tail_m`
（应 ≈ 脚底离地高 0.033，不能更低）、`final_root_tilt_deg`（0）。

## 训练 / 验收 / 部署

```bash
# 微调（热启动，experiment_name 与 jump1_1mwithid 相同）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Jump1-1mWithIdRecover --headless \
    --resume --load_run <产出 0915_28000 的 run> --checkpoint model_28000.pt --max_iterations 6000

# play：hold 到最后一帧，看末帧后 |ω|<0.3 rad/s、骨盆倾角<10°、腿 |dq|<1 rad/s；对比起跳高度/距离
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-Jump1-1mWithIdRecover --load_run <run> --checkpoint <ckpt>

# 部署侧参考 CSV（120 fps，电机序），和训练看的是同一个 NPZ
python scripts/mimic/npz_to_deploy_csv.py \
    source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/jump1_1mwithid_recover/jump1_1m_waist15_recover.npz \
    deploy/robots/g1_29dof/config/policy/mimic/jump3/params/deploy.yaml \
    deploy/robots/g1_29dof/config/policy/mimic/jump3/params/jump3_waist15_recover.csv --fps 120
```

上机时把新 `policy.onnx`/`deploy.yaml` 放进 `deploy/robots/g1_29dof/config/policy/mimic/jump3/exported/<日期>/`
并复制到根目录，`config.yaml` 的 `Mimic_Jump3.motion_file` 改成 `jump3_waist15_recover.csv`。
**旧策略 0915_28000 不能配新 CSV 用**：它没学过 hold 段，会一直盯着一个从没见过的参考。
