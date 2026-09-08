#pragma once

#include <chrono>
#include <thread>

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"

/**
 * Run：可控步态跑步策略（Unitree-G1-29dof-Run / RunWithId）。
 *
 * 结构与 State_RLBase 相同（加载 policy_dir 下的 deploy.yaml + policy.onnx，50 Hz 策略线程），多两件事：
 *  1. 把 config.yaml 里 FSM.Run.gait 的手柄映射（限速、档位轴、档位表）喂给 env->gait_command；
 *  2. 退出门槛：除急停（Passive）外，transitions 里的所有切换只在"指令已站立且关节基本静止"时放行。
 *     2.5 m/s 时切走路策略（它没见过这个速度）或切阻尼都是摔倒；正确的停法是松开摇杆，
 *     策略自己减速、支撑相滑到 1.0 站定（训练里 θ→1 的行为），然后再切。
 */
class State_Run : public FSMState
{
public:
    State_Run(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    bool stopped();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::chrono::steady_clock::time_point last_block_log_{};
};

REGISTER_FSM(State_Run)
