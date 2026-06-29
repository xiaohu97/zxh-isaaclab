#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "unitree_joystick_dsl.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

// ---------------------------------------------------------------------------
// Left-arm trajectory excitation command (matches Stand-LeftArmTrack-v0).
// Reproduces the Python LeftArmJointTrajectoryCommand: Fourier-series reference
// (relative to default pose) + quintic-smoothstep gate, toggled by a joystick
// button. Output: [q_ref_rel(n), dq_ref(n)*ref_vel_scale, enabled(1)].
// All trajectory data comes from deploy.yaml params (see export_deploy_cfg.py).
// ---------------------------------------------------------------------------
REGISTER_OBSERVATION(arm_command)
{
    // ----- static config, loaded once from params -----
    static const int n = params["n_joints"].as<int>();
    static const float period = params["period"].as<float>();
    static const float blend_time = params["blend_time_s"].as<float>();
    static const float vscale = params["ref_vel_scale"].as<float>();
    static const std::vector<float> omega = params["omega"].as<std::vector<float>>();
    static const std::vector<std::vector<float>> a = params["a_rel"].as<std::vector<std::vector<float>>>();
    static const std::vector<std::vector<float>> b = params["b_rel"].as<std::vector<std::vector<float>>>();
    static const std::string toggle_expr =
        params["toggle"].IsDefined() ? params["toggle"].as<std::string>() : std::string("RB + A.on_pressed");
    static const auto toggle_pred = []() {
        unitree::common::dsl::Parser p(toggle_expr);
        auto ast = p.Parse();
        return unitree::common::dsl::Compile(*ast);
    }();

    // ----- per-episode runtime state -----
    static long last_ep = -1;
    static bool want = false;   // intended on/off (toggled by button)
    static float s = 0.0f;      // ramp parameter in [0,1] feeding the smoothstep
    static float phase = 0.0f;  // trajectory phase [s]

    // reset on (re)entry: episode_length is set to 0 by env->reset()
    if (env->episode_length <= last_ep) { want = false; s = 0.0f; phase = 0.0f; }
    last_ep = env->episode_length;

    // toggle excitation on button rising edge; restart routine when turning on from off
    if (toggle_pred(*env->robot->data.joystick)) {
        want = !want;
        if (want && s <= 0.0f) phase = 0.0f;
    }

    // advance ramp parameter toward target (1=on, 0=off) over blend_time
    const float dir = want ? 1.0f : -1.0f;
    float s_dot = (blend_time > 1e-6f) ? (dir / blend_time) : (dir * 1.0e6f);
    s = std::clamp(s + s_dot * env->step_dt, 0.0f, 1.0f);
    if (s <= 0.0f || s >= 1.0f) s_dot = 0.0f;  // velocity is zero at the rails

    // quintic smoothstep gate and its time derivative
    const float gate = s * s * s * (s * (6.0f * s - 15.0f) + 10.0f);
    const float dgate_ds = 30.0f * s * s * (s - 1.0f) * (s - 1.0f);
    const float gate_dot = dgate_ds * s_dot;

    // advance trajectory phase while active
    if (s > 0.0f) {
        phase += env->step_dt;
        if (phase >= period) phase -= period;
    }

    // evaluate Fourier series (coeffs are already relative to default => q_ref_rel)
    const int K = static_cast<int>(omega.size());
    std::vector<float> obs(2 * n + 1, 0.0f);
    for (int j = 0; j < n; ++j) {
        float q = 0.0f, dq = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float ang = omega[k] * phase;
            const float c = std::cos(ang);
            const float sn = std::sin(ang);
            q += a[k][j] * c + b[k][j] * sn;
            dq += omega[k] * (-a[k][j] * sn + b[k][j] * c);
        }
        obs[j] = gate * q;                                    // q_ref_rel
        obs[n + j] = (gate_dot * q + gate * dq) * vscale;     // dq_ref * ref_vel_scale
    }
    obs[2 * n] = (s > 0.0f) ? 1.0f : 0.0f;                    // enabled flag
    return obs;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}