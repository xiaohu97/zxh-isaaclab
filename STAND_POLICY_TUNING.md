"""
站立策略优化指南 - 解决抖动和高度控制问题

关键修改：

1. 解决抖动问题的三个层面：
   ✓ 奖励函数：降低权重，避免过度纠正
   ✓ 正则化惩罚：增强 action_rate 惩罚，平缓控制
   ✓ 控制频率：增加 decimation 从 4 到 8，降低控制速率

2. 增强高度响应性：
   ✓ 新增 track_height_command 函数：根据速度命令调整目标高度
   ✓ 增大训练命令范围：lin_vel_x (-0.15, 0.15) -> 更大范围
   ✓ 更新高度平缓奖励函数：避免二次函数导致的过度纠正

3. 具体改动值：
   
   a) 奖励函数权重变化：
   ┌─────────────────────────┬─────────┬───────┬──────────┐
   │ 项目                    │ 旧值    │ 新值  │ 说明     │
   ├─────────────────────────┼─────────┼───────┼──────────┤
   │ height_tracking         │ 3.0     │ 1.5   │ 降低纠正 │
   │ posture_tracking        │ 3.0     │ 1.5   │ 降低纠正 │
   │ track_lin_vel_xy_exp    │ 2.0     │ 1.5   │ 降低快速 │
   │ action_rate_l2          │ -0.01   │ -0.05 │ 增强平缓 │
   │ joint_velocity_penalty  │ 无      │ -0.5  │ 新增项   │
   │ dof_torques_l2          │ -0.0001 │ -0.0005│增强     │
   │ waist_penalty           │ -3.0    │ -2.0  │ 放宽约束 │
   │ arm_penalty             │ -2.0    │ -1.5  │ 放宽约束 │
   └─────────────────────────┴─────────┴───────┴──────────┘
   
   b) 控制参数变化：
   - decimation: 4 -> 8 (降低控制频率)
   - rel_standing_envs: 0.8 -> 0.75 (增加变化训练)
   - resampling_time_range: (8.0, 12.0) -> (6.0, 10.0)

4. 新增函数说明：

   track_height_command():
   - 用途：根据速度命令自适应调整目标高度
   - 逻辑：有前进命令时降低高度（俯身），后退时提高高度
   - 效果：前后摇杆时高度会相应变化

   penalize_joint_velocity():
   - 用途：惩罚腿部关节的高速运动
   - 效果：防止关节快速往复运动（抖动的主要原因）

训练建议：

1. 从新配置开始训练：
   python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Stand-v2 --headless

2. 监视以下指标：
   - episode_total_reward: 应该稳步上升
   - reward_track_lin_vel_xy_exp: 应该 > 0.8（速度控制好）
   - reward_action_rate_l2: 应该接近 0（动作平缓）
   - reward_joint_velocity_penalty: 应该接近 0（无高速运动）

3. 如果仍有问题，进一步调整：

   抖动仍然严重：
   - 增大 action_rate_l2 权重到 -0.1
   - 增加 joint_velocity_penalty 权重到 -1.0
   - 检查 joint_pos_rel 观测是否有异常噪声

   高度响应仍然不足：
   - 增大 height_command_tracking 权重到 2.0-3.0
   - 扩大命令范围 lin_vel_x (-0.2, 0.2)
   - 减少 decimation 到 6

4. 部署验证（MuJoCo）：

   使用部署代码时：
   - 摇杆X前后控制高度
   - 摇杆Y左右控制侧向倾斜
   - 摇杆旋转控制身体旋转
   - 所有关节应该平稳，无抖动
"""
