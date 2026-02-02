#!/usr/bin/env python3
"""
站立策略训练监控脚本

监视关键指标：
1. 总奖励是否上升
2. 各个奖励项是否符合预期
3. 动作平缓度（action_rate）
4. 关节速度
"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_log_file(log_file):
    """解析 RSL-RL 日志文件"""
    try:
        with open(log_file, 'r') as f:
            # 读取最后几行
            lines = f.readlines()[-100:]
            
        print("\n" + "="*70)
        print("📊 最近的训练日志 (最后100行)")
        print("="*70)
        
        for line in lines:
            # 查找关键指标
            if 'FPS' in line or 'epoch' in line:
                print(line.rstrip())
            elif 'mean_reward' in line:
                print(f"✓ {line.rstrip()}")
            elif 'action_rate' in line or 'reward_action' in line:
                print(f"⚡ {line.rstrip()}")
            elif 'joint_velocity' in line or 'joint_vel' in line:
                print(f"⚡ {line.rstrip()}")
            elif 'height' in line or 'track_height' in line:
                print(f"📏 {line.rstrip()}")
                
    except FileNotFoundError:
        print(f"❌ 找不到日志文件: {log_file}")


def check_training_stats(log_dir):
    """检查训练统计"""
    log_dir = Path(log_dir)
    
    # 查找最新的运行目录
    run_dirs = sorted(log_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        print("❌ 没有找到训练结果")
        return
    
    latest_run = run_dirs[0]
    print(f"\n📁 检查目录: {latest_run}")
    
    # 查找日志文件
    log_files = list(latest_run.glob("*.log")) + list(latest_run.glob("**/*.log"))
    
    if log_files:
        parse_log_file(log_files[0])
    
    # 检查是否有完整的训练配置
    config_file = latest_run / "config.yaml"
    if config_file.exists():
        print("\n✓ 找到训练配置文件")
        print(f"  位置: {config_file}")


def main():
    parser = argparse.ArgumentParser(description='监控站立策略训练')
    parser.add_argument('--log-dir', type=str, default='logs/rsl_rl',
                        help='训练日志目录')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 G1 站立策略训练监控")
    print("="*70)
    
    print("""
监视关键指标：

1️⃣  总奖励 (mean_reward)
   - 应该稳步上升
   - 最终应该在 15-25 之间

2️⃣  动作平缓度 (action_rate_l2)
   - 对于解决抖动很关键
   - 权重为 -0.05，应该看到这项贡献 0.2-0.5 的负奖励
   - 越接近 0 越好（表示动作变化很小）

3️⃣  关节速度惩罚 (joint_velocity_penalty)
   - 新增的惩罚项
   - 权重为 -0.5，应该看到 0.1-0.3 的负奖励
   - 帮助消除抖动

4️⃣  高度跟踪 (height_tracking, height_command_tracking)
   - 两个奖励项都应该 > 0.5
   - 表示策略能跟踪高度命令

5️⃣  速度命令跟踪 (track_lin_vel_xy_exp)
   - 应该 > 0.8
   - 表示站立策略能准确跟踪零速度（站立）

建议的训练时间：
- 快速预检查：500-1000 episodes (~10-20 min)
- 足够的训练：5000-10000 episodes (~1-2 hours)
- 充分训练：20000+ episodes (~4-8 hours)

开始训练命令：
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Stand-v2 --headless

观看实时训练效果：
  python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stand-v2 \\
    --checkpoint logs/rsl_rl/unitree_g1_29dof_stand_v2/*/model_*.pt
""")
    
    check_training_stats(args.log_dir)


if __name__ == '__main__':
    main()
