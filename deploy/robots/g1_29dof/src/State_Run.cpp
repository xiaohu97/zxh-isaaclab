#include "State_Run.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

State_Run::State_Run(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    if (!env->gait_command) {
        throw std::runtime_error(
            "State_" + state_string + ": deploy.yaml 缺少 commands.base_velocity.gait，"
            "请用 scripts/rsl_rl/export_deploy.py 重新导出");
    }
    env->gait_command->configure(cfg["gait"]);
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 退出门槛：包住 FSMState 构造时按 transitions 注册的检查；急停（Passive）不设门槛
    const int passive_id = FSMStringMap.right.at("Passive");
    for (auto & check : registered_checks)
    {
        if (check.second == passive_id) continue;
        auto cond = check.first;
        check.first = [this, cond]() -> bool {
            if (!cond()) return false;
            if (stopped()) return true;
            auto now = std::chrono::steady_clock::now();
            if (now - last_block_log_ > std::chrono::seconds(1)) {
                last_block_log_ = now;
                spdlog::warn("State_Run: switch blocked, release the sticks and let the robot come to a stop first "
                             "(|v_cmd| < 0.1, stance ratio > 0.95, joints still)");
            }
            return false;
        };
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            passive_id
        )
    );
}

bool State_Run::stopped()
{
    if (!env->gait_command->is_settled()) return false;
    const auto & dq = env->robot->data.joint_vel;
    return dq.size() == 0 || dq.cwiseAbs().maxCoeff() < 1.0f;
}

void State_Run::enter()
{
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->robot->update();
    spdlog::info("State_Run: sticks = velocity, RT = run gear, release sticks to stop; vx cap {:.2f} m/s",
                 env->gait_command->effective_hi(isaaclab::GaitCommand::LIN_VEL_X));

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env->reset();  // 指令从站立值起步（GaitCommand::reset），再按斜率滑向摇杆目标

        while (policy_thread_running)
        {
            env->step();

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Run::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_Run::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
}
