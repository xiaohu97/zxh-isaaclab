#pragma once

#include "walk_handoff.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <set>

namespace g1
{
// Private, inference-thread-owned state. It never reads the live DDS message,
// the control thread's joystick, or the motor publisher while constructing obs.
class WalkArticulation : public isaaclab::Articulation
{
public:
    WalkArticulation()
    {
        data.root_quat_w = Eigen::Quaternionf::Identity();
        data.root_ang_vel_b.setZero();
        data.projected_gravity_b = data.GRAVITY_VEC_W;
        data.joystick = &joystick_;
        joystick_.ly.smooth = joystick_.lx.smooth = joystick_.rx.smooth = 1;
        joystick_.ly.deadzone = joystick_.lx.deadzone = joystick_.rx.deadzone = 0;
    }
    void update() override
    {
        // Manager construction resizes q/dq before its first update().
        if (!initialized_) {
            data.joint_pos.setZero();
            data.joint_vel.setZero();
        }
    }

    void set(const ControlFrame& frame, const std::array<float, 3>& command)
    {
        if (!frame.valid()) throw std::invalid_argument("Invalid walk control frame");
        for (int i = 0; i < joint_count; ++i) {
            const int motor = data.joint_ids_map[i];
            data.joint_pos[i] = frame.q[motor];
            data.joint_vel[i] = frame.dq[motor];
        }
        data.root_quat_w = frame.quaternion.normalized();
        data.root_ang_vel_b = frame.angular_velocity;
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        data.height_map = frame.height_map;  // ignored by proprioceptive policies
        joystick_.ly(command[0]);
        joystick_.lx(-command[1]);
        joystick_.rx(-command[2]);
        initialized_ = true;
    }
private:
    unitree::common::UnitreeJoystick joystick_;
    bool initialized_ = false;
};

class WalkPolicy
{
public:
    explicit WalkPolicy(const YAML::Node& cfg)
    {
        const double period = cfg["step_dt"].as<double>();
        if (!std::isfinite(period) || std::fabs(period - .02) > 1e-6) {
            throw std::invalid_argument("Walk handoff expects the exported 50 Hz policy");
        }
        const auto ids = cfg["joint_ids_map"].as<std::vector<int>>();
        if (ids.size() != joint_count || std::set<int>(ids.begin(), ids.end()).size() != joint_count
            || *std::min_element(ids.begin(), ids.end()) != 0 || *std::max_element(ids.begin(), ids.end()) != 28) {
            throw std::invalid_argument("Walk requires a permutation of 29 G1 motor IDs");
        }
        const auto actions = cfg["actions"];
        if (actions.size() != 1 || !actions["JointPositionAction"]
            || !actions["JointPositionAction"]["joint_ids"].IsNull()) {
            throw std::invalid_argument("Walk requires one full-body JointPositionAction");
        }
        offset_ = actions["JointPositionAction"]["offset"].as<std::vector<float>>();
        scale_ = actions["JointPositionAction"]["scale"].as<std::vector<float>>();
        if (offset_.size() != joint_count || scale_.size() != joint_count) {
            throw std::invalid_argument("Walk action offset/scale dimensions must be 29");
        }
        for (int i = 0; i < joint_count; ++i) {
            if (!std::isfinite(offset_[i]) || !std::isfinite(scale_[i]) || scale_[i] == 0) {
                throw std::invalid_argument("Invalid walk action offset/scale");
            }
        }
        const std::vector<std::string> names = {"base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos_rel", "joint_vel_rel", "last_action"};
        const auto observations = cfg["observations"];
        if (observations.size() != names.size() && observations.size() != names.size() + 1) {
            throw std::invalid_argument("Unsupported walk observations");
        }
        std::size_t k = 0;
        for (const auto& term : observations) {
            const auto name = term.first.as<std::string>();
            if (k < names.size()) {
                if (name != names[k] || term.second["history_length"].as<int>() != 5) {
                    throw std::invalid_argument("Walk expects the exported six-term, five-frame observation layout");
                }
            } else {
                // Perceptive walk (Unitree-G1-29dof-PerceptiveHeightScan) appends one
                // single-frame height scan after the six proprioceptive terms.
                if (name != "height_scan" || term.second["history_length"].as<int>() != 1) {
                    throw std::invalid_argument("Walk only accepts 'height_scan' (history 1) as a seventh observation");
                }
                height_scan_size_ = term.second["scale"].size();
                if (height_scan_size_ == 0) throw std::invalid_argument("height_scan observation has no dimension");
            }
            ++k;
        }
        robot_ = std::make_shared<WalkArticulation>();
        // Manager construction evaluates every term once; height_scan throws on an empty map.
        if (height_scan_size_) robot_->data.height_map.assign(height_scan_size_, 0.0f);
        env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(cfg, robot_);
        if (robot_->data.joint_stiffness.size() != joint_count || robot_->data.joint_damping.size() != joint_count) {
            throw std::invalid_argument("Walk PD gain dimensions must be 29");
        }
        // Manager construction calls observation terms once; initialize q/dq
        // explicitly before any real inference or history warm-up.
        ControlFrame initial;
        if (height_scan_size_) initial.height_map.assign(height_scan_size_, 0.0f);
        robot_->set(initial, {});
    }

    // Non-zero for perceptive policies: cells expected in ControlFrame::height_map.
    std::size_t height_scan_size() const { return height_scan_size_; }
    bool uses_height_scan() const { return height_scan_size_ > 0; }

    void load(const std::string& path) { env_->alg = std::make_unique<isaaclab::OrtRunner>(path); }
    void cancel_inference() { if (env_->alg) env_->alg->cancel_inference(); }
    void resume_inference() { if (env_->alg) env_->alg->reset_inference_cancel(); }
    double step_dt() const { return env_->step_dt; }
    const std::vector<float>& stiffness() const { return robot_->data.joint_stiffness; }
    const std::vector<float>& damping() const { return robot_->data.joint_damping; }

    // Supply oldest-to-newest physical samples, excluding the sample that will
    // be used by the first step(). No unexecuted policy outputs enter history.
    void reset(const std::vector<ControlFrame>& history)
    {
        if (history.empty()) throw std::invalid_argument("Walk history is empty");
        env_->reset();
        for (size_t i = 0; i < history.size(); ++i) {
            prepare(history[i], 0);
            if (i == 0) env_->observation_manager->reset();
            else env_->observation_manager->compute();
        }
    }

    JointVector step(const ControlFrame& frame, float command_gain)
    {
        if (!env_->alg) throw std::runtime_error("Walk policy not loaded");
        prepare(frame, command_gain);
        env_->step();
        const auto target = env_->action_manager->processed_actions();
        JointVector motor_target{};
        for (int i = 0; i < joint_count; ++i) {
            if (!std::isfinite(target[i])) throw std::runtime_error("Non-finite walk policy output");
            motor_target[static_cast<int>(robot_->data.joint_ids_map[i])] = target[i];
        }
        return motor_target;
    }

    // Same path as step(), exposed to offline observation parity tests.
    void prepare(const ControlFrame& frame, float command_gain)
    {
        if (height_scan_size_ && frame.height_map.size() != height_scan_size_) {
            throw std::invalid_argument("Walk control frame height map does not match the policy's height_scan dimension");
        }
        const auto ranges = env_->cfg["commands"]["base_velocity"]["ranges"];
        const char* keys[] = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
        std::array<float, 3> command{};
        for (int j = 0; j < 3; ++j) {
            command[j] = std::clamp(frame.command[j], ranges[keys[j]][0].as<float>(),
                                   ranges[keys[j]][1].as<float>()) * std::clamp(command_gain, 0.0f, 1.0f);
        }
        robot_->set(frame, command);
        std::vector<float> previous(joint_count);
        for (int i = 0; i < joint_count; ++i) {
            previous[i] = (frame.previous_target[static_cast<int>(robot_->data.joint_ids_map[i])] - offset_[i]) / scale_[i];
        }
        // Applies the model's raw_clip to last_action, just as in training.
        env_->action_manager->process_action(previous);
    }

    isaaclab::ManagerBasedRLEnv& env() { return *env_; }

private:
    std::shared_ptr<WalkArticulation> robot_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
    std::vector<float> offset_, scale_;
    std::size_t height_scan_size_ = 0;
};
}  // namespace g1
