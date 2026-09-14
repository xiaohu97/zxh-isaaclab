"""``VelocityWithIdYaw`` 的 PPO 配置：相对 ``BasePPORunnerCfg`` 只加一条 ``clip_actions``。

2026-09-13：本任务的 run ``2026-09-13_16-32-08_yawfix`` 在 13900 迭代崩于
``RuntimeError: normal expects all elements of std >= 0.0``。value loss 的爆炸过程::

    iter    error_vel_yaw   track_ang_vel_z   value_function
    13500      0.7569           0.5973            0.0065
    13800      0.9426           0.5111        54180.9
    13900      1.2499           0.3854         1.56e21     -> std NaN -> 崩

和 ``run`` 在 2026-09-06_21-49-17 的 30794 迭代那次是同一个故障模式（见
``locomotion/robots/g1/run/rsl_rl_ppo_cfg.py`` 的注释）：偶发的动作尖峰经 ``action_rate``
进奖励，再经 ``last_action`` 观测正反馈，最终 value loss inf -> 参数 NaN。``run`` 当时用
``clip_actions = 10.0`` 修好了，但 velocity 系列复用的 ``BasePPORunnerCfg`` 是
``clip_actions = None``，所以这个保护没有被继承过来。

``JointPositionAction`` 的 scale 是 0.25，10 对应 ±2.5 rad 的关节偏移，远在正常动作幅值之上
（``run`` 实测训好的策略 p99=5.7、p99.9=7.2），只掐尖峰，对正常动作没有任何影响。

其余超参全部继承 ``BasePPORunnerCfg``，改基类会自动同步到这里。``max_iterations`` 保持基类的
50000，实际起训时用 ``--max_iterations 15000`` 指定——三个 run 都证明 10000 迭代后进平台期。
"""
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class VelocityYawPPORunnerCfg(BasePPORunnerCfg):
    """``BasePPORunnerCfg`` + 动作尖峰限幅，防 std NaN。"""

    clip_actions = 10.0
