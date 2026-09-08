// 离线对拍：用 Isaac Lab 导出的参考轨迹（export_deploy.py --reference_steps）驱动部署侧的
// 观测拼装 + ONNX 推理 + 动作处理，不需要机器人。验证三件事：
//   1) 观测向量（项顺序、5 帧历史堆叠、裁剪/缩放、gait_commands 归一化、gait_clock）与 Isaac Lab 逐位一致
//   2) ONNX 推理结果与 torch 策略一致
//   3) raw_clip 生效：last_action 与下发的关节目标都基于裁剪后的动作
//
// 用法: test_run_policy <policy_dir>   (policy_dir 下有 exported/policy.onnx, params/deploy.yaml, params/reference.json)

#include <cmath>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

struct MockArticulation : public isaaclab::Articulation
{
    void update() override {}  // 状态由测试直接写进 data
};

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: test_run_policy <policy_dir>\n";
        return 2;
    }
    const std::filesystem::path policy_dir = argv[1];

    auto cfg = YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string());
    auto robot = std::make_shared<MockArticulation>();
    isaaclab::ManagerBasedRLEnv env(cfg, robot);
    if (!env.gait_command) {
        std::cerr << "deploy.yaml has no commands.base_velocity.gait\n";
        return 2;
    }
    env.alg = std::make_unique<isaaclab::OrtRunner>((policy_dir / "exported" / "policy.onnx").string());

    nlohmann::json ref;
    std::ifstream(policy_dir / "params" / "reference.json") >> ref;
    const auto& steps = ref["steps"];
    const int n_joints = robot->data.joint_ids_map.size();
    std::cout << "reference: " << steps.size() << " steps, obs_dim " << ref["obs_dim"] << ", " << n_joints << " joints\n";

    auto set_state = [&](const nlohmann::json& s) {
        for (int i = 0; i < n_joints; ++i) {
            robot->data.joint_pos[i] = s["joint_pos"][i].get<float>();
            robot->data.joint_vel[i] = s["joint_vel"][i].get<float>();
        }
        for (int i = 0; i < 3; ++i) {
            robot->data.root_ang_vel_b[i] = s["root_ang_vel_b"][i].get<float>();
            robot->data.projected_gravity_b[i] = s["projected_gravity_b"][i].get<float>();
        }
        isaaclab::GaitCommand::Vec c;
        for (int i = 0; i < isaaclab::GaitCommand::DIM; ++i) c[i] = s["command"][i].get<float>();
        env.gait_command->set_command(c, s["phase"].get<float>());
    };
    auto ref_vec = [](const nlohmann::json& a) {
        std::vector<float> v;
        for (const auto& x : a) v.push_back(x.get<float>());
        return v;
    };

    // reset：Isaac 在 reset 时把 5 帧历史全部填成 reset 那一帧的观测；这里等价地手动做，
    // 不走 env.reset()，因为它会先用（空的）手柄把指令重置成站立值
    set_state(steps[0]);
    env.action_manager->reset();
    env.observation_manager->reset();

    const float clip = ref.contains("clip_actions") && !ref["clip_actions"].is_null() ? ref["clip_actions"].get<float>() : INFINITY;
    float worst_obs = 0.0f, worst_act = 0.0f;
    int fail = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
        set_state(steps[k]);
        auto obs = env.observation_manager->compute().at("obs");
        auto ref_obs = ref_vec(steps[k]["obs"]);
        float d_obs = max_abs_diff(obs, ref_obs);

        auto act = env.alg->act({{"obs", obs}});
        auto ref_act = ref_vec(steps[k]["action"]);
        float d_act = max_abs_diff(act, ref_act);

        // 用参考动作推进（避免 1e-5 级差异逐步累积），并检查 raw_clip
        env.action_manager->process_action(ref_act);
        auto stored = env.action_manager->action();
        for (int i = 0; i < n_joints; ++i) {
            float expect = std::clamp(ref_act[i], -clip, clip);
            if (std::fabs(stored[i] - expect) > 1e-6f) { fail++; std::cout << "  raw_clip mismatch at step " << k << " joint " << i << "\n"; break; }
        }

        worst_obs = std::max(worst_obs, d_obs);
        worst_act = std::max(worst_act, d_act);
        if (k < 3 || d_obs > 1e-4f || d_act > 2e-3f || k + 1 == steps.size()) {
            std::cout << "step " << k << ": max|obs diff| = " << d_obs << "   max|action diff| = " << d_act
                      << "   phase " << steps[k]["phase"].get<float>() << "\n";
        }
        if (d_obs > 1e-4f || d_act > 2e-3f) fail++;
    }

    // 第一个观测项不对时给出定位：找出第一个超差的分量下标
    {
        set_state(steps[0]);
        env.action_manager->reset();
        env.observation_manager->reset();
        auto obs = env.observation_manager->compute().at("obs");
        auto ref_obs = ref_vec(steps[0]["obs"]);
        for (size_t i = 0; i < std::min(obs.size(), ref_obs.size()); ++i) {
            if (std::fabs(obs[i] - ref_obs[i]) > 1e-4f) {
                std::cout << "first obs mismatch at index " << i << ": cpp " << obs[i] << " vs isaac " << ref_obs[i] << "\n";
                break;
            }
        }
        if (obs.size() != ref_obs.size()) std::cout << "obs size: cpp " << obs.size() << " vs isaac " << ref_obs.size() << "\n";
    }

    std::cout << "\nworst: obs " << worst_obs << ", action " << worst_act << "  -> " << (fail ? "FAIL" : "PASS") << "\n";
    return fail ? 1 : 0;
}
