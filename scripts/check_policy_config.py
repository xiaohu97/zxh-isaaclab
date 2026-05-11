#!/usr/bin/env python3
"""
诊断脚本：检查训练配置与部署配置是否匹配

使用方法:
    python scripts/check_policy_config.py <policy_dir>

例如:
    python scripts/check_policy_config.py logs/rsl_rl/unitree_g1_29dof_stand/2024-01-01_00-00-00
"""

import argparse
import yaml
import json
from pathlib import Path


def load_deploy_yaml(policy_dir: Path) -> dict:
    """加载 deploy.yaml 配置"""
    deploy_path = policy_dir / "params" / "deploy.yaml"
    if not deploy_path.exists():
        raise FileNotFoundError(f"找不到 deploy.yaml: {deploy_path}")
    with open(deploy_path, 'r') as f:
        return yaml.safe_load(f)


def check_observation_dimensions(cfg: dict) -> dict:
    """检查观测维度"""
    obs_cfg = cfg.get('observations', {})
    total_dim = 0
    history_length = 1
    details = {}
    
    for obs_name, obs_params in obs_cfg.items():
        if obs_params is None:
            continue
        scale = obs_params.get('scale', [1.0])
        if isinstance(scale, list):
            dim = len(scale)
        else:
            dim = 1
        hist = obs_params.get('history_length', 1)
        history_length = max(history_length, hist)
        details[obs_name] = {
            'dim': dim,
            'history': hist,
            'params': obs_params.get('params', {})
        }
        total_dim += dim
    
    return {
        'total_single_step': total_dim,
        'history_length': history_length,
        'total_with_history': total_dim * history_length,
        'details': details
    }


def check_action_dimensions(cfg: dict) -> dict:
    """检查动作维度"""
    actions_cfg = cfg.get('actions', {})
    total_dim = 0
    details = {}
    
    for action_name, action_params in actions_cfg.items():
        if action_params is None:
            continue
        scale = action_params.get('scale', [])
        if isinstance(scale, list):
            dim = len(scale)
        else:
            dim = 29  # 默认29关节
        details[action_name] = {
            'dim': dim,
            'scale': scale[:5] if isinstance(scale, list) else scale,  # 只显示前5个
        }
        total_dim += dim
    
    return {
        'total_dim': total_dim,
        'details': details
    }


def check_command_config(cfg: dict) -> dict:
    """检查命令配置"""
    commands_cfg = cfg.get('commands', {})
    details = {}
    
    for cmd_name, cmd_params in commands_cfg.items():
        if cmd_params is None:
            continue
        ranges = cmd_params.get('ranges', {})
        dim = len([k for k, v in ranges.items() if v is not None])
        details[cmd_name] = {
            'dim': dim,
            'ranges': ranges
        }
    
    return details


def main():
    parser = argparse.ArgumentParser(description='检查策略配置')
    parser.add_argument('policy_dir', type=str, help='策略目录路径')
    args = parser.parse_args()
    
    policy_dir = Path(args.policy_dir)
    
    print("=" * 60)
    print(f"策略配置诊断: {policy_dir}")
    print("=" * 60)
    
    # 加载配置
    try:
        cfg = load_deploy_yaml(policy_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 检查观测
    print("\n📊 观测配置:")
    obs_info = check_observation_dimensions(cfg)
    print(f"  单步维度: {obs_info['total_single_step']}")
    print(f"  历史长度: {obs_info['history_length']}")
    print(f"  总维度 (含历史): {obs_info['total_with_history']}")
    print("\n  详细:")
    for name, detail in obs_info['details'].items():
        params_str = str(detail['params']) if detail['params'] else ''
        print(f"    - {name}: dim={detail['dim']}, history={detail['history']} {params_str}")
    
    # 检查动作
    print("\n🎮 动作配置:")
    action_info = check_action_dimensions(cfg)
    print(f"  总维度: {action_info['total_dim']}")
    for name, detail in action_info['details'].items():
        print(f"    - {name}: dim={detail['dim']}")
    
    # 检查命令
    print("\n📝 命令配置:")
    cmd_info = check_command_config(cfg)
    for name, detail in cmd_info.items():
        print(f"  - {name}: dim={detail['dim']}")
        for range_name, range_val in detail['ranges'].items():
            if range_val is not None:
                print(f"      {range_name}: {range_val}")
    if 'base_velocity' in cmd_info:
        print("  ✅ 检测到 base_velocity：适用于 Stand-v2/Velocity 的部署兼容命令")
    
    # 检查关键问题
    print("\n" + "=" * 60)
    print("⚠️  潜在问题检查:")
    print("=" * 60)
    
    issues = []
    
    # 检查观测中的命令引用
    for obs_name, obs_detail in obs_info['details'].items():
        if 'command_name' in obs_detail['params']:
            cmd_ref = obs_detail['params']['command_name']
            if cmd_ref not in cmd_info:
                issues.append(f"观测 '{obs_name}' 引用了命令 '{cmd_ref}'，但命令配置中没有定义")
    
    # 检查动作维度
    if action_info['total_dim'] != 29:
        issues.append(f"动作维度为 {action_info['total_dim']}，预期为 29 (G1 29DOF)")
    
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ 未发现明显配置问题")
    
    print("\n" + "=" * 60)
    print("💡 如果策略无法站立，请检查:")
    print("=" * 60)
    print("""
  1. 训练时的命令名称与 deploy.yaml 中的 command_name 是否一致
  2. 观测维度是否与 ONNX 模型输入维度匹配
  3. 动作维度是否与关节数量匹配
  4. 关节顺序是否与 joint_ids_map 一致
  
  对于 G1 Stand-v2 策略:
  - 使用 base_velocity 命令是预期配置，便于复用现有部署输入
  - velocity_commands 观测的 command_name 应指向已定义命令，通常为 base_velocity
  - 如果模型输入维度不匹配，优先检查 deploy.yaml 的 observations/actions 导出配置
""")


if __name__ == '__main__':
    main()
