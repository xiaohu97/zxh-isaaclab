#include "State_Walk.h"
#include <chrono>

namespace
{
using Clock = std::chrono::steady_clock;
double seconds_now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
g1::ControlHistory control_history;
}

void State_Walk::record_control_frame()
{
    g1::ControlFrame frame;
    frame.time = seconds_now();
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& state = lowstate->msg_;
        for (int m = 0; m < g1::joint_count; ++m) {
            frame.q[m] = state.motor_state()[m].q();
            frame.dq[m] = state.motor_state()[m].dq();
        }
        const auto& quat = state.imu_state().quaternion();
        frame.quaternion = Eigen::Quaternionf(quat[0], quat[1], quat[2], quat[3]);
        for (int j = 0; j < 3; ++j) frame.angular_velocity[j] = state.imu_state().gyroscope()[j];
    }
    // Both joystick updates and lowcmd writes belong to this control thread.
    frame.command = {lowstate->joystick.ly(), -lowstate->joystick.lx(), -lowstate->joystick.rx()};
    for (int m = 0; m < g1::joint_count; ++m) frame.previous_target[m] = lowcmd->msg_.motor_cmd()[m].q();
    control_history.push(frame);
}

State_Walk::State_Walk(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    const auto cfg = param::config["FSM"][state_string];
    const auto entry = cfg["walk_entry"];
    if (entry) {
        entry_cfg_.blend_time = entry["blend_time_s"].as<double>(0.30);
        entry_cfg_.zero_command_time = entry["zero_command_time_s"].as<double>(0.40);
        entry_cfg_.command_ramp_time = entry["command_ramp_time_s"].as<double>(0.50);
        entry_cfg_.inference_timeout = entry["inference_timeout_s"].as<double>(0.20);
    }
    entry_cfg_.validate();
    handoff_ = std::make_unique<g1::WalkHandoff>(entry_cfg_);
    const auto dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    policy_ = std::make_unique<g1::WalkPolicy>(YAML::LoadFile(dir / "params/deploy.yaml"));
    policy_->load((dir / "exported/policy.onnx").string());

    // Check faults before user transitions; never let a failed/stale inference
    // jump to another active policy. Existing emergency/lowstate checks remain.
    registered_checks.insert(registered_checks.begin(), {
        [this]() {
            const auto frames = control_history.window(1, policy_->step_dt());
            bool bad = handoff_->fault(seconds_now()) || frames.empty();
            if (!frames.empty()) {
                const auto& f = frames.back();
                bad = bad || !f.valid() || seconds_now() - f.time > entry_cfg_.inference_timeout;
                if (f.valid()) {
                    const auto gravity = f.quaternion.normalized().conjugate() * Eigen::Vector3f(0, 0, -1);
                    bad = bad || std::acos(std::clamp(-gravity.z(), -1.0f, 1.0f)) > 1.0f;
                }
            }
            if (bad && !fault_logged_) {
                spdlog::error("Velocity: invalid/stale inference, sensor frame, or bad orientation; entering Passive");
                fault_logged_ = true;
            }
            return bad;
        }, FSMStringMap.right.at("Passive")});
}

void State_Walk::enter()
{
    // Called after the previous state's worker has joined. The q values here
    // are the last targets actually published by that state, in motor order.
    record_control_frame();
    const auto frame = control_history.window(1, policy_->step_dt()).back();
    float max_leg_speed = 0, max_leg_error = 0;
    for (int m = 0; m < 12; ++m) {
        max_leg_speed = std::max(max_leg_speed, std::fabs(frame.dq[m]));
        max_leg_error = std::max(max_leg_error, std::fabs(frame.previous_target[m] - frame.q[m]));
    }
    const Eigen::Vector3f gravity = frame.quaternion.normalized().conjugate() * Eigen::Vector3f(0, 0, -1);
    const float tilt = std::acos(std::clamp(-gravity.z(), -1.0f, 1.0f));
    spdlog::info("Velocity entry measured: pelvis_tilt={:.3f} rad, gyro=[{:.3f},{:.3f},{:.3f}] rad/s, max_leg_dq={:.3f} rad/s, max_leg_target_error={:.3f} rad, waist_pitch_q={:.3f}, waist_pitch_last_target={:.3f}",
                 tilt, frame.angular_velocity.x(), frame.angular_velocity.y(), frame.angular_velocity.z(),
                 max_leg_speed, max_leg_error, frame.q[14], frame.previous_target[14]);
    handoff_->begin(frame.previous_target, seconds_now());
    fault_logged_ = false;
    policy_->resume_inference();
    for (int m = 0; m < g1::joint_count; ++m) {
        auto& cmd = lowcmd->msg_.motor_cmd()[m];
        cmd.kp() = policy_->stiffness()[m];
        cmd.kd() = policy_->damping()[m];
        cmd.dq() = cmd.tau() = 0;
    }
    spdlog::info("Velocity entry: hold last sent target until inference ready; blend {} s, zero command {} s, ramp {} s",
                 entry_cfg_.blend_time, entry_cfg_.zero_command_time, entry_cfg_.command_ramp_time);
    running_ = true;
    policy_thread_ = std::thread([this] {
        try {
            auto history = control_history.window(5, policy_->step_dt());
            if (history.empty()) throw std::runtime_error("No walk entry history");
            auto current = history.back();
            history.pop_back();
            policy_->reset(history);
            const auto dt = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(policy_->step_dt()));
            auto next = Clock::now();
            bool first = true;
            while (running_) {
                const double now = seconds_now();
                if (now - current.time > entry_cfg_.inference_timeout) throw std::runtime_error("Stale walk sensor frame");
                const auto target = policy_->step(current, handoff_->command_gain(now));
                if (seconds_now() - current.time > entry_cfg_.inference_timeout) {
                    throw std::runtime_error("Walk inference exceeded the input age limit");
                }
                handoff_->publish(target, seconds_now());
                if (first) {
                    float delta = 0, leg_delta = 0, kp_delta = 0;
                    int worst_motor = 0;
                    for (int m = 0; m < g1::joint_count; ++m) {
                        const float change = std::fabs(target[m] - current.previous_target[m]);
                        if (change > delta) { delta = change; worst_motor = m; }
                        if (m < 12) leg_delta = std::max(leg_delta, change);
                        kp_delta = std::max(kp_delta, policy_->stiffness()[m] * change);
                    }
                    spdlog::info("Velocity first inference ready: candidate max |q_target - last_sent| = {:.4f} rad at motor {}, max_leg_delta={:.4f} rad, max_kp_delta={:.2f} Nm, waist_pitch_candidate={:.3f} rad (before blend; not applied torque)",
                                 delta, worst_motor, leg_delta, kp_delta, target[14]);
                    first = false;
                }
                next += dt;
                if (next < Clock::now()) next = Clock::now() + dt;  // skip missed deadlines; do not burst
                std::unique_lock<std::mutex> lock(wake_mutex_);
                if (wake_.wait_until(lock, next, [this] { return !running_.load(); })) break;
                lock.unlock();
                current = control_history.window(1, policy_->step_dt()).back();
            }
        } catch (const std::exception& e) {
            if (running_) {  // normal state exit also cancels an active Run
                handoff_->fail();
                spdlog::error("Velocity inference failed: {}", e.what());
            }
        }
    });
}

void State_Walk::run()
{
    const auto target = handoff_->sample(seconds_now());
    for (int m = 0; m < g1::joint_count; ++m) lowcmd->msg_.motor_cmd()[m].q() = target[m];
}

void State_Walk::exit()
{
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        running_ = false;
    }
    if (policy_) policy_->cancel_inference();
    wake_.notify_all();
    if (policy_thread_.joinable()) policy_thread_.join();
}
