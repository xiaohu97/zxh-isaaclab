# Humanoid Ultra 原始参考资料

`original/` 保存从 `/media/zxh/XIAOHU` 复制的原始机器人配置、任务代码、
手柄读取文件和参考策略。正式可运行的 Isaac Lab 集成位于：

```text
source/unitree_rl_lab/unitree_rl_lab/assets/robots/humanoid_ultra.py
source/unitree_rl_lab/unitree_rl_lab/tasks/humanoid_ultra/base
```

原始任务来自 RoboLab 风格工程，包含当前仓库没有的 AMP 数据和定制依赖。
请将此目录作为迁移对照，不要直接加入任务自动发现路径。

`policy.pt` 和 `gamepad_reader.py` 被检测为非标准数据文件，不应在确认来源和
格式前执行或加载。仓库的 `.gitignore` 默认忽略 `*.pt`。
