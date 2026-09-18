"""生成离线测试用的合成感知 walk 策略：``fixtures/velocity_rough_synthetic/``。

* ``params/deploy.yaml``：线上 velocity 的 deploy.yaml + 训练侧 PerceptiveHeightScan 会导出的
  ``height_scan`` 观测项（187 维、history 1、clip [-1, 1]、offset 0.5）。
* ``exported/policy.onnx``：667 → 32 → 29 的小 MLP，固定种子；输入名 ``obs``、输出名 ``actions``、
  opset 11、无动态轴，和 isaaclab_rl 的 ``export_policy_as_onnx`` 一致。权重让动作明显依赖高程图，
  测试用它确认高程图真的进了网络。**不是训练出来的策略，不能上机。**

用法（任意带 torch + onnx 的环境，CPU 即可）::

    python deploy/robots/g1_29dof/test/fixtures/make_velocity_rough_fixture.py
"""
from __future__ import annotations

import pathlib

import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DEPLOY_YAML = HERE.parents[1] / "config" / "policy" / "velocity" / "params" / "deploy.yaml"
OUT_DIR = HERE / "velocity_rough_synthetic"
NUM_PROPRIO = 480
NUM_SCAN = 187
NUM_ACTIONS = 29


class SyntheticPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(NUM_PROPRIO + NUM_SCAN, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, NUM_ACTIONS),
        )
        with torch.no_grad():
            first = self.net[0]
            first.weight.mul_(0.05)
            # 高程图列加大权重，动作对地形明显敏感
            first.weight[:, NUM_PROPRIO:].mul_(10.0)
            self.net[2].weight.mul_(0.2)
            self.net[2].bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    cfg = yaml.safe_load(SOURCE_DEPLOY_YAML.read_text())
    expected = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action"]
    if list(cfg["observations"]) != expected:
        raise RuntimeError(f"unexpected source observation layout: {list(cfg['observations'])}")
    cfg["observations"]["height_scan"] = {
        "params": {
            "sensor_cfg": {
                "name": "height_scanner",
                "joint_names": None,
                "joint_ids": None,
                "fixed_tendon_names": None,
                "fixed_tendon_ids": None,
                "body_names": None,
                "body_ids": None,
                "object_collection_names": None,
                "object_collection_ids": None,
                "preserve_order": False,
            },
            "offset": 0.5,
        },
        "clip": [-1.0, 1.0],
        "scale": [1.0] * NUM_SCAN,
        "history_length": 1,
    }
    (OUT_DIR / "params").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exported").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "params" / "deploy.yaml", "w") as f:
        f.write("# 合成测试策略，由 make_velocity_rough_fixture.py 生成；不是训练结果，不能上机。\n")
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    (OUT_DIR / "params" / "source.txt").write_text(
        "synthetic fixture from make_velocity_rough_fixture.py; layout = velocity deploy.yaml + height_scan(187, history 1)\n"
    )

    policy = SyntheticPolicy().eval()
    torch.onnx.export(
        policy,
        torch.zeros(1, NUM_PROPRIO + NUM_SCAN),
        str(OUT_DIR / "exported" / "policy.onnx"),
        export_params=True,
        opset_version=11,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
