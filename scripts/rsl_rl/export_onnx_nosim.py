"""从 rsl_rl checkpoint 直接导出部署用 ONNX，不启动 Isaac Sim。

`export_deploy.py` 要建一个 Isaac 场景才能导出（顺带写 deploy.yaml），在 GPU 被训练占满时会 OOM。
本脚本只用 checkpoint 里的 actor 权重，按 isaaclab_rl 的 `_OnnxPolicyExporter` 逐项复刻：
纯 MLP（无 RNN、无 normalizer，对应 `empirical_normalization=False`）、opset 11、
输入 `obs` [1,N]、输出 `actions` [1,M]。

**只产出 policy.onnx。** deploy.yaml 必须另外来自与该策略配套的 run（或语义等价的既有版本），
本脚本不生成，也不会去猜。

用法:
  python scripts/rsl_rl/export_onnx_nosim.py <checkpoint.pt> <out_dir> [--activation elu]
校验(可选，强烈建议)：用一个已知 checkpoint 导出后与现有官方导出物逐点比对，见 --verify-against。
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("out_dir")
p.add_argument("--activation", default="elu", choices=list(ACT))
p.add_argument("--filename", default="policy.onnx")
p.add_argument("--verify-against", default=None, help="已有 onnx，用随机输入比对最大绝对误差")
a = p.parse_args()

sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
if any(k.startswith("actor_obs_normalizer") or "normalizer" in k.lower() for k in sd):
    raise SystemExit("checkpoint 含 normalizer，本脚本未实现，请用 export_deploy.py")
idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
layers, prev = [], None
for n, i in enumerate(idx):
    w = sd[f"actor.{i}.weight"]
    if n:
        layers.append(ACT[a.activation]())
    lin = nn.Linear(w.shape[1], w.shape[0])
    lin.weight.data, lin.bias.data = w.clone(), sd[f"actor.{i}.bias"].clone()
    layers.append(lin)
    prev = w
actor = nn.Sequential(*layers).eval()
in_dim = sd[f"actor.{idx[0]}.weight"].shape[1]
out_dim = prev.shape[0]

os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, a.filename)
torch.onnx.export(actor, torch.zeros(1, in_dim), out, export_params=True, opset_version=11,
                  input_names=["obs"], output_names=["actions"], dynamic_axes={})
print(f"{out}: obs[1,{in_dim}] -> actions[1,{out_dim}]  (来自 {a.checkpoint})")

if a.verify_against:
    import onnx
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, in_dim)).astype(np.float32)
    mine = ReferenceEvaluator(onnx.load(out)).run(None, {"obs": x})[0]
    ref_model = onnx.load(a.verify_against)
    ref_in = ref_model.graph.input[0].name
    ref = ReferenceEvaluator(ref_model).run(None, {ref_in: x})[0]
    d = float(np.abs(mine - ref).max())
    print(f"  与 {a.verify_against} 的最大绝对误差: {d:.3e}  ({'一致' if d < 1e-5 else '不一致 —— 不要使用本导出物'})")
