"""Run 任务的 mdp 聚合模块。

**必须是 .py 模块，不能是包**：``scripts/list_envs.py`` 在 AppLauncher 之前就会
``__import__`` ``locomotion.robots`` 下的每一个**子包**（见其 ``_walk_packages``），
而本模块第一行就会把 isaaclab 拉进来 —— 那时 ``omni.log`` 还不存在，会直接炸掉
train.py / play.py / list_envs.py。普通模块不在那次遍历的导入范围内，所以安全。
同理，这个目录下不要再新建子包。
"""
# 先摊平 locomotion.mdp（含 IsaacLab 原生 mdp），再用本任务的新增项覆盖。
from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F401, F403
from .gait_command import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
